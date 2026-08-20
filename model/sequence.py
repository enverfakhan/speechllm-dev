"""Sequence assembly and batched-generation utilities.

Two input conventions, selected by cfg.model.input_convention and dispatched by
assemble_inputs():

  "flat"  (prepare_input)       the original layout
              [audio] [SEP] [instruction] [SEP] [transcript] [SEP]
          The trailing SEP is the EOS target: the model is trained to predict it
          after the last transcript token so greedy decoding terminates.

  "chat"  (prepare_input_chat)  the Llama 3.1 Instruct chat template
              [bos] [start_header] "user" [end_header] "\n\n"
                  AUDIO_BOS [audio] AUDIO_EOS "\n" [instruction]
              [eot]
              [start_header] "assistant" [end_header] "\n\n"
                  [transcript]
              [eot]                          <- EOS target
          Only the TRAILING eot is a label; the user-turn eot and every scaffold
          token are -100.  AUDIO_BOS/AUDIO_EOS are two learned vectors owned by
          the bridge (model/adapter.py) - input-only embeddings, NOT vocabulary
          tokens, so they own no logit row and can never be generated.

Audio-first inside the user turn is deliberate: every token before AUDIO_BOS is
fixed scaffold, so the audio always starts at the SAME offset
(ChatTemplate.audio_offset) in every sample, and swapping instructions in a
later experiment only ever touches the tail.  That constant offset is what lets
prepare_input_chat hand Llama an exact audio mask for the in-layer adapters:
under the chat convention the audio is NOT at [0, audio_lengths[i]).

EvalPrefixBatch handles batched greedy generation under either convention
without causal attention ever seeing padding tokens in the history of real
tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F


# Valid cfg.model.input_convention values; utils/config.py mirrors this set.
INPUT_CONVENTIONS: frozenset[str] = frozenset({"flat", "chat"})


def prepare_input(
    adapter_out: torch.Tensor,
    audio_lengths: torch.Tensor,
    instruction_ids: torch.Tensor,
    instruction_lengths: torch.Tensor,
    transcript_ids: torch.Tensor,
    transcript_lengths: torch.Tensor,
    embed_layer: nn.Embedding,
    sep_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble per-sample embedded sequences and loss masks, then batch-pad.

    Each sample's sequence layout:
        [audio: audio_lengths[i]] [SEP] [instruction: instruction_lengths[i]]
        [SEP] [transcript: transcript_lengths[i]] [SEP]

    The trailing SEP is the EOS target (not masked to -100); the model is
    trained to predict it as the last output token.

    Padding tokens beyond each length are stripped before concatenation, so
    no pad token ever enters the sequence or the loss computation.
    Samples are zero-padded at the end to the longest sequence in the batch.

    The loss mask is -100 everywhere except the real transcript positions.

    This function processes samples one at a time (simple Python loop).
    If it becomes a throughput bottleneck a CUDA kernel can replace the loop
    while keeping the same interface.

    Args:
        adapter_out:          (B, T_audio_max, 4096) — AudioAdapter output
        audio_lengths:        (B,) — real audio tokens per sample, in [1, T_audio_max]
        instruction_ids:      (B, T_inst_max)  — tokenised instruction prompts
        instruction_lengths:  (B,) — real instruction tokens per sample
        transcript_ids:       (B, T_trans_max) — tokenised transcripts
        transcript_lengths:   (B,) — real transcript tokens per sample
        embed_layer:          Llama's token embedding (nn.Embedding)
        sep_token_id:         integer ID of the SEP token

    Returns:
        inputs: (B, L_max, 4096) — batch-padded embedded sequences (zero-padded)
        labels: (B, L_max)       — -100 except at real transcript positions;
                                   end-padding positions are also -100
    """
    B      = adapter_out.shape[0]
    device = adapter_out.device

    # Batch-embed instructions and transcripts up-front (single kernel call each)
    inst_embeds  = embed_layer(instruction_ids)   # (B, T_inst_max,  d_model)
    trans_embeds = embed_layer(transcript_ids)    # (B, T_trans_max, d_model)

    sep_id    = torch.tensor([sep_token_id], dtype=torch.long, device=device)
    sep_embed = embed_layer(sep_id)               # (1, d_model)

    # Build one (sequence, label) pair per sample, then pad to a common length
    seqs:   list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    for i in range(B):
        T_audio = int(audio_lengths[i].item())
        T_inst  = int(instruction_lengths[i].item())
        T_trans = int(transcript_lengths[i].item())

        # Slice to real lengths, discarding pad tokens
        audio_emb = adapter_out[i, :T_audio, :]      # (T_audio, d_model)
        inst_emb  = inst_embeds[i,  :T_inst,  :]     # (T_inst,  d_model)
        trans_emb = trans_embeds[i, :T_trans, :]     # (T_trans, d_model)

        # Final SEP serves as EOS: the model is trained to predict it after the
        # last transcript token so generation naturally terminates.
        seq_i = torch.cat(
            [audio_emb, sep_embed, inst_emb, sep_embed, trans_emb, sep_embed], dim=0
        )
        seqs.append(seq_i)

        T_prefix = T_audio + 1 + T_inst + 1          # audio + SEP + instruction + SEP
        lbl_i = torch.cat([
            torch.full((T_prefix,), -100, dtype=torch.long, device=device),
            transcript_ids[i, :T_trans],
            torch.tensor([sep_token_id], dtype=torch.long, device=device),
        ], dim=0)
        labels.append(lbl_i)

    # Pad every sample to the longest sequence in this batch
    L_max = max(s.shape[0] for s in seqs)

    inputs = torch.stack([
        F.pad(s, (0, 0, 0, L_max - s.shape[0]))      # zero-pad time dim
        for s in seqs
    ], dim=0)  # (B, L_max, d_model)

    padded_labels = torch.stack([
        F.pad(lbl, (0, L_max - lbl.shape[0]), value=-100)
        for lbl in labels
    ], dim=0)  # (B, L_max)

    return inputs, padded_labels


# ── Chat convention ───────────────────────────────────────────────────────────

class _SpecialTokens(Protocol):
    """What ChatTemplate.from_tokenizer needs of a tokenizer.

    Deliberately structural: the same class builds the scaffold in PRUNED id
    space (from data.PrunedTokenizer, for training) and in ORIGINAL id space
    (from a shim over the raw HuggingFace tokenizer, for the
    apply_chat_template anchor check in tools/build_vocab.py).  Keeping model/
    free of a transformers import is the other half of that.
    """

    bos_token_id:    int | None
    start_header_id: int | None
    end_header_id:   int | None
    eot_token_id:    int | None

    def encode(self, text: str) -> list[int]: ...


@dataclass(frozen=True)
class ChatTemplate:
    """The fixed Llama 3.1 Instruct scaffold segments, precomputed once.

    Single source of truth for chat assembly: prepare_input_chat and
    EvalPrefixBatch both build their sequences out of exactly these segments, and
    tools/build_vocab.py anchors them against the tokenizer's own
    ``apply_chat_template`` output at vocabulary-build time.

    Segment layout (▸ = learned marker vector, spliced as an embedding, no id)::

        seg_pre_audio        [bos][start_header]"user"[end_header]"\\n\\n"
        ▸audio_bos  [audio]  ▸audio_eos
        seg_pre_instruction  "\\n"
        {instruction}
        seg_pre_transcript   [eot][start_header]"assistant"[end_header]"\\n\\n"
        {transcript}
        eot_token_id                                        <- EOS target

    All ids are in whatever space the source tokenizer used (pruned at training
    time).  Tuples, not lists, so the dataclass stays hashable and frozen.
    """

    seg_pre_audio:       tuple[int, ...]
    seg_pre_instruction: tuple[int, ...]
    seg_pre_transcript:  tuple[int, ...]
    eot_token_id:        int

    @property
    def audio_offset(self) -> int:
        """Sequence index where the first AUDIO CONTENT token lands.

        Constant across samples — that is the whole point of putting the audio
        first in the user turn.  ``+ 1`` skips the AUDIO_BOS marker, which is a
        delimiter and NOT an audio-content position: the in-layer audio adapters
        must fire on the audio itself, not on its brackets.
        """
        return len(self.seg_pre_audio) + 1

    @classmethod
    def from_tokenizer(cls, tokenizer: _SpecialTokens) -> "ChatTemplate":
        """Precompute the scaffold segments from a tokenizer's ids.

        Fails loudly on anything that would silently mangle the scaffold: a
        missing special id (a vocabulary built without forcing the four chat
        specials in), or an ordinary word that encodes to nothing (pruned-vocab
        encode() DROPS ids it has no mapping for, so a vocabulary that never saw
        "assistant" would quietly yield an empty assistant header).

        Args:
            tokenizer: anything exposing the four special ids and encode();
                       data.PrunedTokenizer is the training-time implementation

        Returns:
            ChatTemplate in the same id space as the tokenizer
        """
        specials = {
            "bos_token_id":    tokenizer.bos_token_id,
            "start_header_id": tokenizer.start_header_id,
            "end_header_id":   tokenizer.end_header_id,
            "eot_token_id":    tokenizer.eot_token_id,
        }
        missing = sorted(k for k, v in specials.items() if v is None)
        if missing:
            raise ValueError(
                f"input_convention 'chat' needs the chat special ids {missing}, "
                "which this tokenizer does not define. Rebuild the vocabulary "
                "with tools/build_vocab.py (it forces the four Llama 3.1 chat "
                "specials into the keep-set and writes their pruned ids into "
                "pruned_config.json)."
            )

        def _encode(text: str) -> tuple[int, ...]:
            ids = tuple(tokenizer.encode(text))
            if not ids:
                raise ValueError(
                    f"chat scaffold text {text!r} encodes to no tokens — the "
                    "pruned vocabulary is missing it. Rebuild the vocabulary; "
                    "build_vocab.py adds every scaffold token to the keep-set."
                )
            return ids

        bos, start, end, eot = (
            specials["bos_token_id"], specials["start_header_id"],
            specials["end_header_id"], specials["eot_token_id"],
        )
        return cls(
            seg_pre_audio       = (bos, start) + _encode("user") + (end,) + _encode("\n\n"),
            # Separates AUDIO_EOS from the instruction text inside the user turn.
            seg_pre_instruction = _encode("\n"),
            seg_pre_transcript  = (eot, start) + _encode("assistant") + (end,) + _encode("\n\n"),
            eot_token_id        = eot,
        )


def prepare_input_chat(
    adapter_out: torch.Tensor,
    audio_lengths: torch.Tensor,
    instruction_ids: torch.Tensor,
    instruction_lengths: torch.Tensor,
    transcript_ids: torch.Tensor,
    transcript_lengths: torch.Tensor,
    embed_layer: nn.Embedding,
    chat: ChatTemplate,
    audio_bos: torch.Tensor,
    audio_eos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble chat-template sequences, loss masks and the audio mask.

    Per-sample layout (see the module docstring for the annotated version)::

        seg_pre_audio  AUDIO_BOS  [audio]  AUDIO_EOS  seg_pre_instruction
        [instruction]  seg_pre_transcript  [transcript]  [eot]

    Padding is stripped from audio / instruction / transcript before
    concatenation exactly as in the flat path, then the batch is zero-padded to
    the longest sequence.

    Labels are -100 everywhere except the assistant-turn transcript tokens and
    the TRAILING eot (the EOS target).  The user-turn eot inside
    seg_pre_transcript is input-only and stays masked.

    Args:
        adapter_out:          (B, T_audio_max, d_model) — bridge output
        audio_lengths:        (B,) — real audio tokens per sample
        instruction_ids:      (B, T_inst_max)  — tokenised instruction prompts
        instruction_lengths:  (B,) — real instruction tokens per sample
        transcript_ids:       (B, T_trans_max) — tokenised transcripts
        transcript_lengths:   (B,) — real transcript tokens per sample
        embed_layer:          Llama's token embedding (nn.Embedding)
        chat:                 precomputed scaffold segments (pruned id space)
        audio_bos:            (d_model,) learned AUDIO_BOS marker (bridge param)
        audio_eos:            (d_model,) learned AUDIO_EOS marker (bridge param)

    Returns:
        inputs:     (B, L_max, d_model) — batch-padded embedded sequences
        labels:     (B, L_max)          — -100 except transcript + trailing eot
        audio_mask: (B, L_max, 1)       — 1.0 on audio-CONTENT positions only
                                          (markers and scaffold excluded), which
                                          is what the in-layer audio adapters gate on
    """
    B      = adapter_out.shape[0]
    device = adapter_out.device

    inst_embeds  = embed_layer(instruction_ids)   # (B, T_inst_max,  d_model)
    trans_embeds = embed_layer(transcript_ids)    # (B, T_trans_max, d_model)

    def _embed_segment(ids: tuple[int, ...]) -> torch.Tensor:
        return embed_layer(torch.tensor(ids, dtype=torch.long, device=device))

    # The three scaffold segments are identical for every sample — embed once.
    pre_audio_emb       = _embed_segment(chat.seg_pre_audio)
    pre_instruction_emb = _embed_segment(chat.seg_pre_instruction)
    pre_transcript_emb  = _embed_segment(chat.seg_pre_transcript)
    eot_emb             = _embed_segment((chat.eot_token_id,))

    marker_bos = audio_bos.unsqueeze(0).to(pre_audio_emb.dtype)   # (1, d_model)
    marker_eos = audio_eos.unsqueeze(0).to(pre_audio_emb.dtype)

    offset = chat.audio_offset
    assert offset == pre_audio_emb.shape[0] + 1, (
        f"audio offset {offset} disagrees with the assembled prefix "
        f"({pre_audio_emb.shape[0]} scaffold tokens + 1 AUDIO_BOS marker)"
    )

    seqs:   list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    n_audio: list[int]         = []

    for i in range(B):
        T_audio = int(audio_lengths[i].item())
        T_inst  = int(instruction_lengths[i].item())
        T_trans = int(transcript_lengths[i].item())

        seq_i = torch.cat([
            pre_audio_emb,                  # [bos][start_header]"user"[end_header]"\n\n"
            marker_bos,                     # AUDIO_BOS
            adapter_out[i, :T_audio, :],    # audio content
            marker_eos,                     # AUDIO_EOS
            pre_instruction_emb,            # "\n"
            inst_embeds[i, :T_inst, :],     # instruction
            pre_transcript_emb,             # [eot][start_header]"assistant"[end_header]"\n\n"
            trans_embeds[i, :T_trans, :],   # transcript
            eot_emb,                        # trailing eot — the EOS target
        ], dim=0)
        seqs.append(seq_i)
        n_audio.append(T_audio)

        T_prefix = (
            pre_audio_emb.shape[0] + 1 + T_audio + 1
            + pre_instruction_emb.shape[0] + T_inst + pre_transcript_emb.shape[0]
        )
        lbl_i = torch.cat([
            torch.full((T_prefix,), -100, dtype=torch.long, device=device),
            transcript_ids[i, :T_trans],
            torch.tensor([chat.eot_token_id], dtype=torch.long, device=device),
        ], dim=0)
        assert lbl_i.shape[0] == seq_i.shape[0], (
            f"chat labels ({lbl_i.shape[0]}) and sequence ({seq_i.shape[0]}) "
            "lengths disagree"
        )
        labels.append(lbl_i)

    L_max = max(s.shape[0] for s in seqs)

    inputs = torch.stack([
        F.pad(s, (0, 0, 0, L_max - s.shape[0]))
        for s in seqs
    ], dim=0)  # (B, L_max, d_model)

    padded_labels = torch.stack([
        F.pad(lbl, (0, L_max - lbl.shape[0]), value=-100)
        for lbl in labels
    ], dim=0)  # (B, L_max)

    # Audio mask: constant start offset, per-sample length.  Built here rather
    # than reconstructed inside Llama.forward, which cannot know the scaffold.
    audio_mask = torch.zeros(B, L_max, 1, dtype=inputs.dtype, device=device)
    for i, T_audio in enumerate(n_audio):
        audio_mask[i, offset : offset + T_audio, 0] = 1.0

    return inputs, padded_labels, audio_mask


def assemble_inputs(
    adapter_out: torch.Tensor,
    audio_lengths: torch.Tensor,
    instruction_ids: torch.Tensor,
    instruction_lengths: torch.Tensor,
    transcript_ids: torch.Tensor,
    transcript_lengths: torch.Tensor,
    embed_layer: nn.Embedding,
    terminator_id: int,
    chat: ChatTemplate | None = None,
    audio_bos: torch.Tensor | None = None,
    audio_eos: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Dispatch to the flat or chat assembler and return a uniform triple.

    ``chat=None`` selects the flat convention and returns audio_mask=None, so
    the caller passes ``audio_lengths`` to Llama.forward and gets exactly the
    pre-chat behaviour.  With a ChatTemplate, the returned audio_mask is the
    authoritative one and Llama.forward uses it verbatim.

    Args:
        (as prepare_input / prepare_input_chat)
        terminator_id: EOS target id — the SEP under "flat", <|eot_id|> under
                       "chat" (where it must equal chat.eot_token_id)
        chat:          ChatTemplate for the chat convention, None for flat
        audio_bos:     (d_model,) AUDIO_BOS marker; required when chat is set
        audio_eos:     (d_model,) AUDIO_EOS marker; required when chat is set

    Returns:
        (inputs, labels, audio_mask | None)
    """
    if chat is None:
        inputs, labels = prepare_input(
            adapter_out, audio_lengths,
            instruction_ids, instruction_lengths,
            transcript_ids, transcript_lengths,
            embed_layer, terminator_id,
        )
        return inputs, labels, None

    if audio_bos is None or audio_eos is None:
        raise ValueError(
            "chat convention needs the bridge's audio_bos/audio_eos markers; "
            "pass adapter.audio_bos / adapter.audio_eos"
        )
    if terminator_id != chat.eot_token_id:
        raise ValueError(
            f"terminator id {terminator_id} != chat.eot_token_id "
            f"{chat.eot_token_id} — the chat convention has exactly one "
            "terminator and it is <|eot_id|>"
        )
    return prepare_input_chat(
        adapter_out, audio_lengths,
        instruction_ids, instruction_lengths,
        transcript_ids, transcript_lengths,
        embed_layer, chat, audio_bos, audio_eos,
    )

class EvalPrefixBatch:
    """Stateful batched-generation context that never exposes padding to causal attention.

    Each sample starts with a fixed prefix ending exactly where the transcript
    begins, so the first generated token is the first transcript token:

        flat  [audio] [SEP] [instruction] [SEP]
        chat  seg_pre_audio AUDIO_BOS [audio] AUDIO_EOS seg_pre_instruction
              [instruction] seg_pre_transcript
              (i.e. ending at the assistant header — see ChatTemplate)

    Under the chat convention the batch also carries an audio mask over the
    prefix's audio-content span, exposed as :attr:`audio_mask` and grown with a
    zero column per generated token (a generated token is never audio).  Pass it
    to Llama.forward; ``audio_lengths`` alone cannot describe the chat layout.

    Prefixes differ in length across samples (different audio durations / instruction
    lengths). To batch them, shorter prefixes are right-padded with zeros to the
    length of the longest prefix. Generated tokens are then inserted at gen_pos[i]
    (the first slot immediately after the real prefix/previously generated tokens),
    NOT appended to the very end.

    This guarantees that causal attention during generation never attends to a
    padding zero that was inserted *before* a real generated token — matching
    the training distribution where no padding exists between sequence sections.

    All sequences in the batch always have the same tensor length (max_prefix_len +
    number of generation steps so far), so stacking into a single batched tensor is
    trivially correct at every step.

    Usage::

        pfx = EvalPrefixBatch(adapter_out, audio_lengths, instruction_ids,
                              instruction_lengths, llama.embed_tokens, sep_token_id)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        generated = [[] for _ in range(B)]
        for _ in range(max_new_tokens):
            context = pfx.get_batch()          # (B, current_len, d)
            # Flat: audio_lengths keeps the gated adapters on the [0, T_a) prefix.
            # Chat: pfx.audio_mask is the authoritative span (audio_mask wins).
            logits, _ = llama(context, None, audio_lengths=audio_lengths,
                              audio_mask=pfx.audio_mask)  # (B, current_len, vocab)
            next_ids = logits[torch.arange(B), pfx.logit_indices, :].argmax(dim=-1)
            for i in range(B):
                if not finished[i]:
                    if int(next_ids[i].item()) == sep_token_id:
                        finished[i] = True
                    else:
                        generated[i].append(int(next_ids[i].item()))
            if finished.all():
                break
            safe_ids    = next_ids.masked_fill(finished, 0)
            next_embeds = llama.embed_tokens(safe_ids.unsqueeze(1))  # (B, 1, d)
            pfx.append(next_embeds, finished)
    """

    def __init__(
        self,
        adapter_out: torch.Tensor,
        audio_lengths: torch.Tensor,
        instruction_ids: torch.Tensor,
        instruction_lengths: torch.Tensor,
        embed_layer: nn.Embedding,
        sep_token_id: int,
        *,
        chat:      "ChatTemplate | None" = None,
        audio_bos: torch.Tensor | None   = None,
        audio_eos: torch.Tensor | None   = None,
    ) -> None:
        """Build per-sample prefix tensors and right-pad to max prefix length.

        Args:
            adapter_out:         (B, T_audio_max, d_model) — AudioAdapter output
            audio_lengths:       (B,) — real audio tokens per sample
            instruction_ids:     (B, T_inst_max) — tokenised instruction
            instruction_lengths: (B,) — real instruction tokens per sample
            embed_layer:         Llama's token embedding nn.Embedding
            sep_token_id:        integer ID of the SEP / EOS token; unused for
                                 assembly under the chat convention, whose
                                 scaffold carries its own eot ids
            chat:                ChatTemplate to assemble the chat prefix, or
                                 None for the flat prefix
            audio_bos/audio_eos: (d_model,) learned bridge markers; required
                                 when chat is set
        """
        B      = adapter_out.shape[0]
        device = adapter_out.device

        inst_embeds = embed_layer(instruction_ids)   # (B, T_inst_max, d)

        if chat is not None and (audio_bos is None or audio_eos is None):
            raise ValueError(
                "chat convention needs the bridge\'s audio_bos/audio_eos markers"
            )

        prefixes: list[torch.Tensor] = []
        if chat is None:
            sep_id  = torch.tensor([sep_token_id], dtype=torch.long, device=device)
            sep_emb = embed_layer(sep_id)            # (1, d)
            for i in range(B):
                T_a = int(audio_lengths[i].item())
                T_i = int(instruction_lengths[i].item())
                pfx = torch.cat([
                    adapter_out[i, :T_a, :],   # audio tokens
                    sep_emb,                    # SEP
                    inst_embeds[i, :T_i, :],   # instruction tokens
                    sep_emb,                    # SEP (before transcript slot)
                ], dim=0)  # (L_prefix_i, d)
                prefixes.append(pfx)
        else:
            def _seg(ids: tuple[int, ...]) -> torch.Tensor:
                return embed_layer(torch.tensor(ids, dtype=torch.long, device=device))

            pre_audio_emb       = _seg(chat.seg_pre_audio)
            pre_instruction_emb = _seg(chat.seg_pre_instruction)
            pre_transcript_emb  = _seg(chat.seg_pre_transcript)
            marker_bos = audio_bos.unsqueeze(0).to(pre_audio_emb.dtype)
            marker_eos = audio_eos.unsqueeze(0).to(pre_audio_emb.dtype)
            for i in range(B):
                T_a = int(audio_lengths[i].item())
                T_i = int(instruction_lengths[i].item())
                pfx = torch.cat([
                    pre_audio_emb,                 # user header
                    marker_bos,                    # AUDIO_BOS
                    adapter_out[i, :T_a, :],       # audio content
                    marker_eos,                    # AUDIO_EOS
                    pre_instruction_emb,           # "\n"
                    inst_embeds[i, :T_i, :],       # instruction
                    pre_transcript_emb,            # eot + assistant header
                ], dim=0)  # (L_prefix_i, d)
                prefixes.append(pfx)

        # gen_pos[i]: index of the next slot to write a generated token into
        self._gen_pos = torch.tensor(
            [p.shape[0] for p in prefixes], dtype=torch.long, device=device
        )

        L_max = int(self._gen_pos.max().item())
        d     = prefixes[0].shape[-1]

        # All contexts start as zero-padded prefix tensors (shape B × L_max × d)
        ctx = torch.zeros(B, L_max, d, device=device, dtype=prefixes[0].dtype)
        for i, p in enumerate(prefixes):
            ctx[i, : p.shape[0]] = p
        self._ctx = ctx   # (B, L_max, d) — grows by 1 column each step

        # Chat convention: the audio sits at a constant offset inside the
        # scaffold, so the gated in-layer adapters need an explicit mask.  Flat
        # convention: None, and the caller keeps passing audio_lengths (audio is
        # the [0, T_a) prefix there).
        self._audio_mask: torch.Tensor | None = None
        if chat is not None:
            offset = chat.audio_offset
            mask   = torch.zeros(B, L_max, 1, device=device, dtype=ctx.dtype)
            for i in range(B):
                T_a = int(audio_lengths[i].item())
                mask[i, offset : offset + T_a, 0] = 1.0
            self._audio_mask = mask

    @property
    def audio_mask(self) -> torch.Tensor | None:
        """(B, current_len, 1) audio-content mask, or None under the flat convention."""
        return self._audio_mask

    @property
    def logit_indices(self) -> torch.Tensor:
        """Per-sample index into the sequence dimension for reading next-token logits.

        At step k, gen_pos[i] is the column where the *next* token will be written.
        The logit for that next token sits at position gen_pos[i] - 1 (the last
        real position currently in the context).

        Returns:
            (B,) int64 tensor of per-sample logit indices
        """
        return self._gen_pos - 1

    def get_batch(self) -> torch.Tensor:
        """Return the current batched context tensor.

        Returns:
            (B, current_len, d_model) — same shape for all sequences
        """
        return self._ctx

    def append(self, token_embeds: torch.Tensor, finished: torch.Tensor) -> None:
        """Grow the context by one step, inserting generated embeddings at gen_pos.

        For finished sequences a zero column is appended and gen_pos is NOT
        advanced, so they maintain correct alignment without polluting history.

        Args:
            token_embeds: (B, 1, d_model) — embedding of the just-chosen token
            finished:     (B,) bool — sequences that have already emitted SEP
        """
        B, _, d = token_embeds.shape
        device  = self._ctx.device

        # Grow every sequence by one zero column (preserves alignment)
        zero_col  = torch.zeros(B, 1, d, device=device, dtype=self._ctx.dtype)
        self._ctx = torch.cat([self._ctx, zero_col], dim=1)

        # Generated tokens are never audio, so the mask grows with zeros — and
        # it must grow, or it would no longer line up with the context.
        if self._audio_mask is not None:
            mask_col = torch.zeros(B, 1, 1, device=device, dtype=self._audio_mask.dtype)
            self._audio_mask = torch.cat([self._audio_mask, mask_col], dim=1)

        # Overwrite the gen_pos slot for each unfinished sequence
        for i in range(B):
            if not finished[i]:
                self._ctx[i, self._gen_pos[i]] = token_embeds[i, 0]
                self._gen_pos[i] += 1


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    torch.manual_seed(0)

    # ── A tiny stand-in for data.PrunedTokenizer ──────────────────────────────
    # Ids are arbitrary but distinct; what matters is that ChatTemplate reads the
    # four specials off the tokenizer and encodes the scaffold words through it.
    # The REAL anchor test — that these segments match the tokenizer's own
    # apply_chat_template rendering — lives in tools/build_vocab.py, where the
    # HuggingFace tokenizer is present (model/ never imports transformers).
    _WORDS = {"user": [11, 12], "assistant": [13], "\n\n": [14], "\n": [15]}

    class _FakeTokenizer:
        bos_token_id    = 1
        start_header_id = 2
        end_header_id   = 3
        eot_token_id    = 4

        def encode(self, text: str) -> list[int]:
            return list(_WORDS[text])

    VOCAB, DIM = 40, 8
    embed = nn.Embedding(VOCAB, DIM)

    chat = ChatTemplate.from_tokenizer(_FakeTokenizer())
    assert chat.seg_pre_audio       == (1, 2, 11, 12, 3, 14), chat.seg_pre_audio
    assert chat.seg_pre_instruction == (15,)
    assert chat.seg_pre_transcript  == (4, 2, 13, 3, 14), chat.seg_pre_transcript
    assert chat.eot_token_id == 4
    assert chat.audio_offset == len(chat.seg_pre_audio) + 1 == 7
    print(f"[OK] ChatTemplate segments built; audio_offset C = {chat.audio_offset}")

    # A tokenizer missing the specials (a vocabulary built before the chat
    # rebuild) must fail loudly rather than assemble a broken scaffold.
    class _LegacyTokenizer(_FakeTokenizer):
        eot_token_id = None

    try:
        ChatTemplate.from_tokenizer(_LegacyTokenizer())
    except ValueError:
        pass
    else:
        raise AssertionError("a tokenizer without eot_token_id must raise")
    print("[OK] missing chat special ids rejected")

    # ── Batch with DIFFERENT audio and instruction lengths ────────────────────
    # Different instruction lengths are the point: the audio offset must NOT move.
    B, T_AUDIO_MAX = 2, 5
    adapter_out    = torch.randn(B, T_AUDIO_MAX, DIM)
    audio_lengths  = torch.tensor([5, 3])
    instruction_ids = torch.tensor([[20, 21, 22, 23],
                                    [24, 25,  0,  0]])
    instruction_lengths = torch.tensor([4, 2])
    transcript_ids  = torch.tensor([[30, 31, 32],
                                    [33,  0,  0]])
    transcript_lengths = torch.tensor([3, 1])

    audio_bos = nn.Parameter(torch.randn(DIM))
    audio_eos = nn.Parameter(torch.randn(DIM))

    inputs, labels, audio_mask = prepare_input_chat(
        adapter_out, audio_lengths,
        instruction_ids, instruction_lengths,
        transcript_ids, transcript_lengths,
        embed, chat, audio_bos, audio_eos,
    )

    # ── Test (a): audio mask — right count, right CONSTANT offset ─────────────
    C = chat.audio_offset
    assert audio_mask.shape == (B, inputs.shape[1], 1), tuple(audio_mask.shape)
    for i in range(B):
        T_a = int(audio_lengths[i].item())
        assert float(audio_mask[i].sum()) == T_a, (
            f"sample {i}: mask sums to {float(audio_mask[i].sum())}, want {T_a}"
        )
        on = audio_mask[i, :, 0].nonzero(as_tuple=False).flatten().tolist()
        assert on == list(range(C, C + T_a)), f"sample {i}: mask at {on}, want [{C},{C+T_a})"
        # The markers bracket the audio and are NOT audio content.
        assert float(audio_mask[i, C - 1, 0]) == 0.0, "AUDIO_BOS must not be masked in"
        assert float(audio_mask[i, C + T_a, 0]) == 0.0, "AUDIO_EOS must not be masked in"
        # The embedded markers really are at those two positions.
        assert torch.allclose(inputs[i, C - 1], audio_bos)
        assert torch.allclose(inputs[i, C + T_a], audio_eos)
        # And the audio content itself landed where the mask says.
        assert torch.allclose(inputs[i, C : C + T_a], adapter_out[i, :T_a])
    print(f"[OK] audio mask: {audio_lengths.tolist()} content positions at the "
          f"constant offset {C} despite instruction lengths "
          f"{instruction_lengths.tolist()}")

    # ── Test (b): labels — transcript + TRAILING eot only ─────────────────────
    for i in range(B):
        T_a, T_i = int(audio_lengths[i].item()), int(instruction_lengths[i].item())
        T_t      = int(transcript_lengths[i].item())
        prefix   = (len(chat.seg_pre_audio) + 1 + T_a + 1
                    + len(chat.seg_pre_instruction) + T_i
                    + len(chat.seg_pre_transcript))
        row = labels[i]
        assert (row[:prefix] == -100).all(), f"sample {i}: scaffold/prompt must be masked"
        assert row[prefix : prefix + T_t].tolist() == transcript_ids[i, :T_t].tolist()
        assert int(row[prefix + T_t].item()) == chat.eot_token_id, "trailing eot must be a label"
        assert (row[prefix + T_t + 1:] == -100).all(), "batch padding must be masked"
        # The user-turn eot (first id of seg_pre_transcript) is input-only.
        user_eot_pos = prefix - len(chat.seg_pre_transcript)
        assert int(labels[i, user_eot_pos].item()) == -100, "user-turn eot must be masked"
        assert torch.allclose(inputs[i, user_eot_pos], embed.weight[chat.eot_token_id])
    n_targets = int((labels != -100).sum())
    assert n_targets == int(transcript_lengths.sum()) + B, n_targets
    print("[OK] labels: transcript + trailing eot only; user-turn eot masked")

    # ── Test (c): assembled scaffold ids, position by position ────────────────
    # Structural check against an independently written-out layout.  The
    # tokenizer-anchored version of this (against apply_chat_template) runs in
    # tools/build_vocab.py at vocabulary-build time.
    for i in range(B):
        T_a, T_i = int(audio_lengths[i].item()), int(instruction_lengths[i].item())
        T_t      = int(transcript_lengths[i].item())
        want_ids = (
            list(chat.seg_pre_audio)
            + [None] * (1 + T_a + 1)                       # AUDIO_BOS, audio, AUDIO_EOS
            + list(chat.seg_pre_instruction)
            + instruction_ids[i, :T_i].tolist()
            + list(chat.seg_pre_transcript)
            + transcript_ids[i, :T_t].tolist()
            + [chat.eot_token_id]
        )
        for pos, tok in enumerate(want_ids):
            if tok is None:
                continue
            assert torch.allclose(inputs[i, pos], embed.weight[tok]), (
                f"sample {i} position {pos}: expected the embedding of id {tok}"
            )
    print("[OK] scaffold ids: every non-audio position matches the intended token")

    # ── Test (d): flat path unchanged (regression guard) ──────────────────────
    SEP = 39
    flat_inputs, flat_labels = prepare_input(
        adapter_out, audio_lengths,
        instruction_ids, instruction_lengths,
        transcript_ids, transcript_lengths,
        embed, SEP,
    )
    sep_emb = embed.weight[SEP]
    for i in range(B):
        T_a, T_i = int(audio_lengths[i].item()), int(instruction_lengths[i].item())
        T_t      = int(transcript_lengths[i].item())
        want = torch.cat([
            adapter_out[i, :T_a], sep_emb[None],
            embed.weight[instruction_ids[i, :T_i]], sep_emb[None],
            embed.weight[transcript_ids[i, :T_t]], sep_emb[None],
        ], dim=0)
        assert torch.allclose(flat_inputs[i, : want.shape[0]], want), f"flat sample {i}"
        assert (flat_inputs[i, want.shape[0]:] == 0).all(), "flat tail must be zero-padded"
        want_lbl = ([-100] * (T_a + 1 + T_i + 1)
                    + transcript_ids[i, :T_t].tolist() + [SEP])
        assert flat_labels[i, : len(want_lbl)].tolist() == want_lbl, f"flat labels {i}"
    # assemble_inputs must route to it unchanged, mask None.
    a_inputs, a_labels, a_mask = assemble_inputs(
        adapter_out, audio_lengths,
        instruction_ids, instruction_lengths,
        transcript_ids, transcript_lengths,
        embed, SEP,
    )
    assert a_mask is None, "flat convention must report no audio mask"
    assert torch.equal(a_inputs, flat_inputs) and torch.equal(a_labels, flat_labels)
    print("[OK] flat path bit-identical, and assemble_inputs dispatches to it")

    # assemble_inputs on the chat path must agree with prepare_input_chat, and
    # must refuse a terminator that is not the chat eot.
    c_inputs, c_labels, c_mask = assemble_inputs(
        adapter_out, audio_lengths,
        instruction_ids, instruction_lengths,
        transcript_ids, transcript_lengths,
        embed, chat.eot_token_id, chat=chat, audio_bos=audio_bos, audio_eos=audio_eos,
    )
    assert torch.equal(c_inputs, inputs) and torch.equal(c_labels, labels)
    assert torch.equal(c_mask, audio_mask)
    for bad_kwargs in ({"chat": chat}, {"chat": chat, "audio_bos": audio_bos}):
        try:
            assemble_inputs(
                adapter_out, audio_lengths, instruction_ids, instruction_lengths,
                transcript_ids, transcript_lengths, embed, chat.eot_token_id,
                **bad_kwargs,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("chat assembly without both markers must raise")
    try:
        assemble_inputs(
            adapter_out, audio_lengths, instruction_ids, instruction_lengths,
            transcript_ids, transcript_lengths, embed, SEP,
            chat=chat, audio_bos=audio_bos, audio_eos=audio_eos,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("a terminator that is not chat.eot_token_id must raise")
    print("[OK] assemble_inputs: chat dispatch, marker and terminator guards")

    # ── EvalPrefixBatch: chat prefix ends at the assistant header ─────────────
    pfx = EvalPrefixBatch(
        adapter_out, audio_lengths, instruction_ids, instruction_lengths,
        embed, chat.eot_token_id, chat=chat, audio_bos=audio_bos, audio_eos=audio_eos,
    )
    ctx = pfx.get_batch()
    for i in range(B):
        T_a, T_i = int(audio_lengths[i].item()), int(instruction_lengths[i].item())
        prefix_len = (len(chat.seg_pre_audio) + 1 + T_a + 1
                      + len(chat.seg_pre_instruction) + T_i
                      + len(chat.seg_pre_transcript))
        assert int(pfx.logit_indices[i].item()) == prefix_len - 1, (
            f"sample {i}: generation must start right after the assistant header"
        )
        # The prefix ends with the assistant header, transcript slot still empty.
        for k, tok in enumerate(chat.seg_pre_transcript):
            assert torch.allclose(
                ctx[i, prefix_len - len(chat.seg_pre_transcript) + k], embed.weight[tok]
            ), f"sample {i}: prefix must end at the assistant header"
        # Same audio span as training.
        on = pfx.audio_mask[i, :, 0].nonzero(as_tuple=False).flatten().tolist()
        assert on == list(range(C, C + T_a)), f"sample {i}: eval mask at {on}"
    assert pfx.audio_mask.shape[:2] == ctx.shape[:2]

    # Appending a generated token grows the mask with a zero column.
    finished = torch.zeros(B, dtype=torch.bool)
    pfx.append(embed.weight[torch.tensor([[30], [31]])], finished)
    assert pfx.audio_mask.shape[:2] == pfx.get_batch().shape[:2], "mask must track context"
    assert float(pfx.audio_mask[:, -1].sum()) == 0.0, "generated tokens are never audio"
    for i in range(B):
        assert float(pfx.audio_mask[i].sum()) == int(audio_lengths[i].item())
    print("[OK] EvalPrefixBatch (chat): prefix ends at the assistant header, "
          "mask covers only the audio span and grows with zeros")

    # ── EvalPrefixBatch: flat mode unchanged ──────────────────────────────────
    flat_pfx = EvalPrefixBatch(
        adapter_out, audio_lengths, instruction_ids, instruction_lengths, embed, SEP,
    )
    assert flat_pfx.audio_mask is None, "flat mode must expose no audio mask"
    for i in range(B):
        want_len = int(audio_lengths[i].item()) + 1 + int(instruction_lengths[i].item()) + 1
        assert int(flat_pfx.logit_indices[i].item()) == want_len - 1
    print("[OK] EvalPrefixBatch (flat): unchanged")

    print("\nPASSED")
    sys.exit(0)
