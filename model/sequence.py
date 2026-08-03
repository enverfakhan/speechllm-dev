"""Sequence assembly and batched-generation utilities.

Provides prepare_input(), which strips per-sample audio padding, assembles
    [audio tokens] [SEP] [instruction] [SEP] [transcript] [SEP]
for each sample individually, then pads the batch to the longest sequence.
The trailing SEP is the EOS target: the model is trained to predict it after the
last transcript token so greedy decoding terminates naturally.

EvalPrefixBatch handles batched greedy generation without causal attention ever
seeing padding tokens in the history of real tokens.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class EvalPrefixBatch:
    """Stateful batched-generation context that never exposes padding to causal attention.

    Each sample starts with a fixed prefix:
        [audio tokens] [SEP] [instruction tokens] [SEP]
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
            # audio_lengths keeps gated audio adapters firing on the prefix only
            logits, _ = llama(context, None, audio_lengths=audio_lengths)  # (B, current_len, vocab)
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
    ) -> None:
        """Build per-sample prefix tensors and right-pad to max prefix length.

        Args:
            adapter_out:         (B, T_audio_max, d_model) — AudioAdapter output
            audio_lengths:       (B,) — real audio tokens per sample
            instruction_ids:     (B, T_inst_max) — tokenised instruction
            instruction_lengths: (B,) — real instruction tokens per sample
            embed_layer:         Llama's token embedding nn.Embedding
            sep_token_id:        integer ID of the SEP / EOS token
        """
        B      = adapter_out.shape[0]
        device = adapter_out.device

        inst_embeds = embed_layer(instruction_ids)   # (B, T_inst_max, d)
        sep_id      = torch.tensor([sep_token_id], dtype=torch.long, device=device)
        sep_emb     = embed_layer(sep_id)            # (1, d)

        prefixes: list[torch.Tensor] = []
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

        # Overwrite the gen_pos slot for each unfinished sequence
        for i in range(B):
            if not finished[i]:
                self._ctx[i, self._gen_pos[i]] = token_embeds[i, 0]
                self._gen_pos[i] += 1
