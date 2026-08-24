"""Autoregressive greedy decoding for speech-llm evaluation."""

from __future__ import annotations

import torch

from model.adapter import BridgeAdapter
from model.sequence import ChatTemplate, EvalPrefixBatch
from model.llama import Llama
from model.whisper_encoder import WhisperEncoder


@torch.no_grad()
def greedy_generate(
    encoder:             WhisperEncoder,
    adapter:             BridgeAdapter,
    llama:               Llama,
    mel:                 torch.Tensor,
    audio_lengths:       torch.Tensor,
    instruction_ids:     torch.Tensor,
    instruction_lengths: torch.Tensor,
    stop_token_id:       int,
    max_new_tokens:      int = 448,
    chat:                ChatTemplate | None = None,
) -> list[list[int]]:
    """Greedy-decode a batch of samples in parallel using EvalPrefixBatch.

    All B sequences generate simultaneously. When a sequence emits the stop
    token — SEP under the flat convention, <|eot_id|> under the chat one — it is
    marked finished and subsequent steps append a zero column to
    maintain tensor alignment without polluting its causal attention history.
    Generation stops once every sequence is finished or max_new_tokens is reached.

    Prefix lengths differ across samples (different audio durations). EvalPrefixBatch
    right-pads shorter prefixes with zeros and inserts each generated token at
    gen_pos[i] rather than at the absolute end, so causal attention never sees
    a padding zero in the history of real tokens — matching the training distribution.

    Args:
        mel:                (B, 80, T_mel)
        audio_lengths:      (B,)
        instruction_ids:    (B, T_inst_max)
        instruction_lengths:(B,)
        stop_token_id:      stop token — generation halts when this is emitted
        max_new_tokens:     hard cap; applied per sequence
        chat:               ChatTemplate to generate under the chat convention
                            (prefix ends at the assistant header), or None for
                            the flat convention

    Returns:
        list of B lists of pruned token IDs (stop token excluded)
    """
    B      = mel.shape[0]
    device = mel.device

    with torch.amp.autocast("cuda", dtype=torch.float16):
        enc_out     = encoder(mel)
        adapter_out = adapter(enc_out)

    pfx = EvalPrefixBatch(
        adapter_out, audio_lengths,
        instruction_ids, instruction_lengths,
        llama.embed_tokens, stop_token_id,
        chat      = chat,
        audio_bos = adapter.audio_bos if chat is not None else None,
        audio_eos = adapter.audio_eos if chat is not None else None,
    )

    finished   = torch.zeros(B, dtype=torch.bool, device=device)
    generated: list[list[int]] = [[] for _ in range(B)]

    for _ in range(max_new_tokens):
        with torch.amp.autocast("cuda", dtype=torch.float16):
            # Flat: audio_lengths is unchanged as the context grows — audio
            # remains the per-sample prefix [0, audio_lengths[i]), so the same
            # mask keeps the gated adapters firing on audio positions only while
            # generated tokens land beyond it.
            # Chat: the audio sits at a scaffold offset, so EvalPrefixBatch owns
            # the mask and grows it; it takes precedence in Llama.forward.
            logits, _ = llama(
                pfx.get_batch(), labels=None,
                audio_lengths=audio_lengths, audio_mask=pfx.audio_mask,
            )  # (B, S, vocab)

        # Read the logit at each sequence's current generation position
        idx_t    = pfx.logit_indices                          # (B,)
        next_ids = logits[torch.arange(B, device=device), idx_t, :].argmax(dim=-1)

        for i in range(B):
            if not finished[i]:
                if int(next_ids[i].item()) == stop_token_id:
                    finished[i] = True
                else:
                    generated[i].append(int(next_ids[i].item()))

        if finished.all():
            break

        safe_ids    = next_ids.masked_fill(finished, 0)
        next_embeds = llama.embed_tokens(safe_ids.unsqueeze(1))  # (B, 1, d)
        pfx.append(next_embeds, finished)

    return generated
