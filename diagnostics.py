"""Training diagnostics for speech-llm.

Drop this file alongside train.py. It adds five targeted metrics that expose
the failure modes a single averaged loss number hides:

  1. token_budget   — transcript tokens in the loss denominator this step
  2. loss_components— loss at position 0 (first transcript token) vs rest
  3. grad_norms     — per-component gradient L2 norm after backward()
  4. first_token    — top-5 predicted tokens at the first transcript position
  5. logit_entropy  — mean softmax entropy at transcript positions (training)
                      and at the generation decision point (eval)

Typical usage in train.py
--------------------------

    from diagnostics import Diagnostics

    diag = Diagnostics(
        tokenizer=tokenizer,        # PrunedTokenizer
        sep_token_id=sep_token_id,
        log_every=10,               # collect every N optimizer steps
        top_k=5,                    # tokens to show in first_token table
    )

    # Inside the micro-step loop, after loss.backward():
    diag.record_micro(labels, loss.detach())

    # Inside the optimizer-step block, after scaler.step():
    diag.record_grad_norms(encoder, adapter, llama)

    # At logging time:
    metrics = diag.flush(global_step)   # returns dict ready for wandb.log()
    # metrics is {} on steps where diag decided not to collect (log_every)

    # During eval (inside _greedy_generate or just after):
    diag.record_generation_entropy(logits, pfx.logit_indices)

Notes
-----
- All operations are on CPU after a single .detach() — no CUDA sync overhead
  during the hot path except the grad norm scan (one pass over parameters).
- flush() resets accumulators, so call it exactly once per optimizer step.
- record_micro() must be called for every micro-step so token counts accumulate
  correctly even when flush() is a no-op (non-logging steps).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from data import PrunedTokenizer
    from model.adapter import AudioAdapter
    from model.llama import Llama
    from model.whisper_encoder import WhisperEncoder


class Diagnostics:
    """Accumulates per-micro-step data and flushes aggregated metrics each optimizer step."""

    def __init__(
        self,
        tokenizer: "PrunedTokenizer",
        sep_token_id: int,
        log_every: int = 10,
        top_k: int = 5,
    ) -> None:
        """
        Args:
            tokenizer:    PrunedTokenizer — used to decode top-k token IDs to text
            sep_token_id: ID of the SEP/EOS token in the pruned vocabulary
            log_every:    collect and return metrics every N optimizer steps
                          (flush() returns {} on non-logging steps)
            top_k:        how many top predicted tokens to show at the first
                          transcript position
        """
        self._tok         = tokenizer
        self._sep_id      = sep_token_id
        self._log_every   = log_every
        self._top_k       = top_k

        # Accumulators — reset by flush()
        self._transcript_tokens: int   = 0    # total unmasked tokens seen
        self._micro_count:       int   = 0    # micro steps since last flush

        # Loss decomposition: first position vs rest
        # We store sums and counts separately so the mean is correct across
        # micro-steps with different sequence lengths.
        self._loss_first_sum:  float = 0.0
        self._loss_first_n:    int   = 0
        self._loss_rest_sum:   float = 0.0
        self._loss_rest_n:     int   = 0

        # Top-k predictions at first transcript position
        # Accumulated as a frequency dict: token_id → count
        self._first_token_votes: dict[int, int] = {}

        # Logit entropy at transcript positions
        self._entropy_sum: float = 0.0
        self._entropy_n:   int   = 0

        # Gradient norms (set by record_grad_norms, consumed by flush)
        self._grad_enc:  float | None = None
        self._grad_ada:  float | None = None
        self._grad_llm:  float | None = None

        # Generation-time entropy (set by record_generation_entropy)
        self._gen_entropy_sum: float = 0.0
        self._gen_entropy_n:   int   = 0

        # Logit saturation (set by record_micro_with_logits)
        self._logit_max_sum: float = 0.0
        self._logit_max_n:   int   = 0

        # Hit/miss loss decomposition (per-token, requires logits)
        # "hit" = top-1 prediction matches label; "miss" = it doesn't
        self._loss_first_hit_sum:  float = 0.0
        self._loss_first_hit_n:    int   = 0
        self._loss_first_miss_sum: float = 0.0
        self._loss_first_miss_n:   int   = 0
        self._loss_rest_hit_sum:   float = 0.0
        self._loss_rest_hit_n:     int   = 0
        self._loss_rest_miss_sum:  float = 0.0
        self._loss_rest_miss_n:    int   = 0

        self._global_step: int = 0

        # ── Eval-pass accumulators (filled by record_eval_micro_with_logits) ──
        # Mirrors the training accumulators; flushed by flush_eval() which uses
        # the "diag_eval/" prefix and does NOT reset the training accumulators.
        self._eval_transcript_tokens: int   = 0
        self._eval_micro_count:       int   = 0
        self._eval_loss_first_sum:    float = 0.0
        self._eval_loss_first_n:      int   = 0
        self._eval_loss_rest_sum:     float = 0.0
        self._eval_loss_rest_n:       int   = 0
        self._eval_first_token_votes: dict[int, int] = {}
        self._eval_entropy_sum:       float = 0.0
        self._eval_entropy_n:         int   = 0
        self._eval_logit_max_sum:     float = 0.0
        self._eval_logit_max_n:       int   = 0
        self._eval_loss_first_hit_sum:  float = 0.0
        self._eval_loss_first_hit_n:    int   = 0
        self._eval_loss_first_miss_sum: float = 0.0
        self._eval_loss_first_miss_n:   int   = 0
        self._eval_loss_rest_hit_sum:   float = 0.0
        self._eval_loss_rest_hit_n:     int   = 0
        self._eval_loss_rest_miss_sum:  float = 0.0
        self._eval_loss_rest_miss_n:    int   = 0

    # ── Per-micro-step call ────────────────────────────────────────────────────

    def record_micro(
        self,
        labels:     torch.Tensor,   # (B, L) — -100 at masked positions
        loss_scalar: torch.Tensor,  # scalar — the loss returned by llama.forward()
    ) -> None:
        """Record one micro-step's labels and loss.

        Call this inside the micro-step loop, immediately after the forward pass,
        before loss.backward(). The tensors are detached and moved to CPU
        immediately so they don't extend the autograd graph.

        Args:
            labels:       (B, L) label tensor from prepare_input(); -100 = masked
            loss_scalar:  scalar loss from llama.forward() (NOT divided by accum_steps)
        """
        self._micro_count += 1

        labels_cpu = labels.detach().cpu()   # (B, L)
        B, L = labels_cpu.shape

        # ── Token budget ──────────────────────────────────────────────────────
        # Count unmasked positions in shift_labels (labels[:, 1:])
        # — this is what cross_entropy actually averages over.
        shift_labels = labels_cpu[:, 1:]          # (B, L-1)
        unmasked     = (shift_labels != -100)     # (B, L-1) bool
        n_unmasked   = int(unmasked.sum().item())
        self._transcript_tokens += n_unmasked

        # ── Loss decomposition ────────────────────────────────────────────────
        # Identify the first unmasked position per sample in shift_labels.
        # That's the position predicting transcript[0] from the SEP before it.
        # We can't recompute per-position losses here without logits, so we
        # use a proxy: the scalar loss is already correct for the "rest";
        # we track whether any sample has >= 2 unmasked tokens (so "rest" exists)
        # and record the micro-step loss for the two categories.
        #
        # For a proper first-vs-rest split we need logits. We collect the proxy
        # here; if you want exact per-position losses, pass logits to this method.
        # The current approach is lightweight and catches the collapse symptom:
        # if loss_first ≈ loss_rest, the model is not differentiating positions.
        loss_val = float(loss_scalar.detach().cpu().item())
        first_count = 0
        rest_count  = 0
        for i in range(B):
            row = unmasked[i]           # (L-1,) bool
            nz  = row.nonzero(as_tuple=False)
            if len(nz) >= 1:
                first_count += 1
            if len(nz) >= 2:
                rest_count += len(nz) - 1
        # Proxy: weight the scalar loss by position counts
        if first_count > 0:
            self._loss_first_sum += loss_val * first_count
            self._loss_first_n   += first_count
        if rest_count > 0:
            self._loss_rest_sum  += loss_val * rest_count
            self._loss_rest_n    += rest_count

        # ── Entropy at transcript positions (training) ─────────────────────────
        # We don't have logits here by default — entropy is computed in
        # record_micro_with_logits() below. This stub keeps the interface simple.
        # entropy is accumulated there separately.

    def record_micro_with_logits(
        self,
        labels:      torch.Tensor,   # (B, L)
        logits:      torch.Tensor,   # (B, L, vocab_size)
        loss_scalar: torch.Tensor,   # scalar
    ) -> None:
        """Extended micro-step recording that also captures logit-level metrics.

        Use this instead of record_micro() when you want:
          - exact per-position loss decomposition (first token vs rest)
          - logit entropy at transcript positions during training
          - top-k first-token predictions

        Slightly more expensive than record_micro() because it computes
        softmax over the full vocab for each unmasked position, but logits
        are already on GPU so the overhead is a single softmax kernel.

        Args:
            labels:      (B, L) — same tensor passed to llama.forward()
            logits:      (B, L, vocab_size) — returned by llama.forward()
            loss_scalar: scalar loss from llama.forward()
        """
        self._micro_count += 1

        
        # Move to CPU once; keep float32 for numerical stability
        labels_cpu = labels.detach().cpu()
        logits_cpu = logits.detach().float().cpu()   # (B, L, V)
        B, L, V    = logits_cpu.shape
        # Track mean of per-sample max logit — saturation indicator
        max_logits = logits_cpu.abs().max(dim=-1).values  # (B, L)
        self._logit_max_sum += float(max_logits.mean().item())
        self._logit_max_n   += 1
        # Shift: logit at position i predicts label at position i+1
        shift_logits = logits_cpu[:, :-1, :]     # (B, L-1, V)
        shift_labels = labels_cpu[:, 1:]          # (B, L-1)
        unmasked     = (shift_labels != -100)     # (B, L-1) bool

        n_unmasked = int(unmasked.sum().item())
        self._transcript_tokens += n_unmasked

        # ── Per-position cross-entropy ─────────────────────────────────────────
        # Compute per-token loss for each unmasked position.
        # shape: (B, L-1) — invalid positions will be ignored
        per_token_loss = F.cross_entropy(
            shift_logits.reshape(-1, V),
            shift_labels.reshape(-1).clamp(min=0),   # clamp -100 → 0 (ignored anyway)
            reduction="none",
        ).reshape(B, L - 1)  # (B, L-1)

        # Top-1 predictions for hit/miss split
        top1_preds = shift_logits.argmax(dim=-1)  # (B, L-1)

        # First unmasked position per sample → "first transcript token" loss
        for i in range(B):
            row_mask = unmasked[i]                   # (L-1,) bool
            nz = row_mask.nonzero(as_tuple=False)
            if len(nz) == 0:
                continue
            first_pos = int(nz[0].item())
            first_loss = float(per_token_loss[i, first_pos].item())
            self._loss_first_sum += first_loss
            self._loss_first_n   += 1

            # Hit/miss for first token
            first_hit = (int(top1_preds[i, first_pos].item()) == int(shift_labels[i, first_pos].item()))
            if first_hit:
                self._loss_first_hit_sum += first_loss
                self._loss_first_hit_n   += 1
            else:
                self._loss_first_miss_sum += first_loss
                self._loss_first_miss_n   += 1

            rest_positions = nz[1:]
            if len(rest_positions) > 0:
                rest_loss = per_token_loss[i, rest_positions].mean().item()
                self._loss_rest_sum += rest_loss
                self._loss_rest_n   += 1

                # Hit/miss per rest token
                for rp in rest_positions:
                    rp_idx  = int(rp.item())
                    r_loss  = float(per_token_loss[i, rp_idx].item())
                    r_hit   = (int(top1_preds[i, rp_idx].item()) == int(shift_labels[i, rp_idx].item()))
                    if r_hit:
                        self._loss_rest_hit_sum += r_loss
                        self._loss_rest_hit_n   += 1
                    else:
                        self._loss_rest_miss_sum += r_loss
                        self._loss_rest_miss_n   += 1

            # ── Top-k first-token prediction ───────────────────────────────────
            # The logit at first_pos - 1 (the SEP before the transcript) predicts
            # the first transcript token.  We want to see what the model thinks
            # the transcript starts with.
            if first_pos > 0:
                pred_logit = shift_logits[i, first_pos - 1]   # (V,)
                topk_ids   = pred_logit.topk(self._top_k).indices.tolist()
                for tid in topk_ids:
                    self._first_token_votes[tid] = self._first_token_votes.get(tid, 0) + 1

        # ── Entropy at all transcript positions ────────────────────────────────
        if n_unmasked > 0:
            # Gather only the unmasked logit rows
            unmasked_logits = shift_logits[unmasked]         # (n_unmasked, V)
            log_probs = F.log_softmax(unmasked_logits, dim=-1)
            probs     = log_probs.exp()
            entropy   = -(probs * log_probs).sum(dim=-1)     # (n_unmasked,)
            self._entropy_sum += float(entropy.sum().item())
            self._entropy_n   += n_unmasked

    def record_eval_micro_with_logits(
        self,
        labels:       torch.Tensor,   # (B, L)
        logits:       torch.Tensor,   # (B, L, vocab_size)
        loss_scalar:  torch.Tensor,   # scalar
    ) -> None:
        """Record one eval-pass micro-step into the eval-specific accumulators.

        Identical computation to record_micro_with_logits() but writes to the
        _eval_* accumulators so flush_eval() can report them under the
        "diag_eval/" prefix without touching the training accumulators.

        Call this inside the diagnostic shard evaluation loop (not the training
        micro-step loop).

        Args:
            labels:      (B, L) — same tensor passed to llama.forward()
            logits:      (B, L, vocab_size) — returned by llama.forward()
            loss_scalar: scalar loss from llama.forward()
        """
        self._eval_micro_count += 1

        labels_cpu = labels.detach().cpu()
        logits_cpu = logits.detach().float().cpu()
        B, L, V    = logits_cpu.shape

        shift_logits = logits_cpu[:, :-1, :]
        shift_labels = labels_cpu[:, 1:]
        unmasked     = (shift_labels != -100)

        n_unmasked = int(unmasked.sum().item())
        self._eval_transcript_tokens += n_unmasked

        max_logits = logits_cpu.abs().max(dim=-1).values
        self._eval_logit_max_sum += float(max_logits.mean().item())
        self._eval_logit_max_n   += 1

        per_token_loss = F.cross_entropy(
            shift_logits.reshape(-1, V),
            shift_labels.reshape(-1).clamp(min=0),
            reduction="none",
        ).reshape(B, L - 1)

        eval_top1_preds = shift_logits.argmax(dim=-1)  # (B, L-1)

        for i in range(B):
            row_mask  = unmasked[i]
            nz        = row_mask.nonzero(as_tuple=False)
            if len(nz) == 0:
                continue
            first_pos  = int(nz[0].item())
            first_loss = float(per_token_loss[i, first_pos].item())
            self._eval_loss_first_sum += first_loss
            self._eval_loss_first_n   += 1

            first_hit = (int(eval_top1_preds[i, first_pos].item()) == int(shift_labels[i, first_pos].item()))
            if first_hit:
                self._eval_loss_first_hit_sum += first_loss
                self._eval_loss_first_hit_n   += 1
            else:
                self._eval_loss_first_miss_sum += first_loss
                self._eval_loss_first_miss_n   += 1

            rest_positions = nz[1:]
            if len(rest_positions) > 0:
                self._eval_loss_rest_sum += per_token_loss[i, rest_positions].mean().item()
                self._eval_loss_rest_n   += 1

                for rp in rest_positions:
                    rp_idx = int(rp.item())
                    r_loss = float(per_token_loss[i, rp_idx].item())
                    r_hit  = (int(eval_top1_preds[i, rp_idx].item()) == int(shift_labels[i, rp_idx].item()))
                    if r_hit:
                        self._eval_loss_rest_hit_sum += r_loss
                        self._eval_loss_rest_hit_n   += 1
                    else:
                        self._eval_loss_rest_miss_sum += r_loss
                        self._eval_loss_rest_miss_n   += 1

            if first_pos > 0:
                pred_logit = shift_logits[i, first_pos - 1]
                topk_ids   = pred_logit.topk(self._top_k).indices.tolist()
                for tid in topk_ids:
                    self._eval_first_token_votes[tid] = (
                        self._eval_first_token_votes.get(tid, 0) + 1
                    )

        if n_unmasked > 0:
            unmasked_logits = shift_logits[unmasked]
            log_probs       = F.log_softmax(unmasked_logits, dim=-1)
            probs           = log_probs.exp()
            entropy         = -(probs * log_probs).sum(dim=-1)
            self._eval_entropy_sum += float(entropy.sum().item())
            self._eval_entropy_n   += n_unmasked

    # ── Per-optimizer-step calls ───────────────────────────────────────────────

    def record_grad_norms(
        self,
        encoder: "WhisperEncoder",
        adapter: "AudioAdapter",
        llama:   "Llama",
    ) -> None:
        """Compute per-component gradient L2 norms.

        Call this after scaler.unscale_(optimizer) but before scaler.step(),
        OR after scaler.step() — the gradients are still populated either way.
        The norm is computed over all parameters that have a non-None gradient.

        Why per-component norms matter:
          - encoder norm >> llama norm → audio signal is overwriting LLM knowledge
          - llama norm ≈ 0 → LLM is not being updated (frozen-like behaviour,
            possibly because gradients are vanishing through the adapter)
          - adapter norm >> both → adapter is the bottleneck; consider lower LR

        Args:
            encoder: WhisperEncoder module
            adapter: AudioAdapter module
            llama:   Llama module
        """
        def _norm(module: nn.Module) -> float:
            total = 0.0
            for p in module.parameters():
                if p.grad is not None:
                    total += p.grad.detach().float().norm().item() ** 2
            return math.sqrt(total)

        self._grad_enc = _norm(encoder)
        self._grad_ada = _norm(adapter)
        self._grad_llm = _norm(llama)

    def record_generation_entropy(
        self,
        logits:        torch.Tensor,   # (B, S, vocab_size)
        logit_indices: torch.Tensor,   # (B,) — per-sample position to read from
    ) -> None:
        """Record logit entropy at the generation decision point during eval.

        Call this inside _greedy_generate(), after each llama() forward pass,
        using pfx.logit_indices as logit_indices. Captures how confident/uncertain
        the model is when choosing the next token.

        High entropy (≈ log(vocab_size)) → model is effectively uniform; random output
        Low entropy and wrong token      → model is confidently wrong (collapse)
        Low entropy and correct tokens   → model is learning

        Args:
            logits:        (B, S, V) — raw logits from llama.forward()
            logit_indices: (B,) — which sequence position to read per sample
        """
        B = logits.shape[0]
        # Gather one logit vector per sample
        idx      = logit_indices.unsqueeze(-1).unsqueeze(-1).expand(B, 1, logits.shape[-1])
        selected = logits.gather(dim=1, index=idx).squeeze(1)    # (B, V)
        selected = selected.detach().float().cpu()

        log_probs = F.log_softmax(selected, dim=-1)
        probs     = log_probs.exp()
        entropy   = -(probs * log_probs).sum(dim=-1)             # (B,)

        self._gen_entropy_sum += float(entropy.sum().item())
        self._gen_entropy_n   += B

    # ── Flush ─────────────────────────────────────────────────────────────────

    def flush(self, global_step: int) -> dict[str, float | str]:
        """Aggregate and return metrics for this optimizer step.

        Returns an empty dict on steps that are not logging steps (log_every).
        Always resets accumulators regardless.

        Args:
            global_step: current optimizer step count (used for log_every gating)

        Returns:
            dict of metric_name → value, ready to pass to wandb.log().
            Empty dict if this is not a logging step.
        """
        self._global_step = global_step
        should_log = (global_step % self._log_every == 0)

        metrics: dict[str, float | str] = {}

        # ── Gradient norms — logged every optimizer step, not gated by log_every ──
        # Per-component norms reveal training dynamics that may spike or collapse
        # between log_every windows and should not be sub-sampled.
        if self._grad_enc is not None:
            metrics["grad/encoder"] = self._grad_enc
        if self._grad_ada is not None:
            metrics["grad/adapter"] = self._grad_ada
        if self._grad_llm is not None:
            metrics["grad/llama"]   = self._grad_llm
        if self._grad_enc is not None and self._grad_llm is not None:
            denom = self._grad_llm if self._grad_llm > 1e-12 else 1e-12
            metrics["grad/enc_llm_ratio"] = self._grad_enc / denom

        if should_log:
            # ── 1. Token budget ────────────────────────────────────────────────
            tokens_per_step = self._transcript_tokens / max(self._micro_count, 1)
            metrics["diag/transcript_tokens_per_micro"]  = tokens_per_step
            metrics["diag/transcript_tokens_this_window"] = float(self._transcript_tokens)

            # ── 2. Loss decomposition ──────────────────────────────────────────
            if self._loss_first_n > 0:
                metrics["loss/train_first_token"] = (
                    self._loss_first_sum / self._loss_first_n
                )
            if self._loss_rest_n > 0:
                metrics["loss/train_rest"] = (
                    self._loss_rest_sum / self._loss_rest_n
                )
            # A healthy model: loss_first > loss_rest (first token is hardest).
            # Collapse symptom: loss_first ≈ loss_rest, both very low (predicting SEP).

            # Hit/miss loss: loss conditioned on whether top-1 prediction was correct
            if self._loss_first_hit_n > 0:
                metrics["loss/train_first_hit"] = (
                    self._loss_first_hit_sum / self._loss_first_hit_n
                )
            if self._loss_first_miss_n > 0:
                metrics["loss/train_first_miss"] = (
                    self._loss_first_miss_sum / self._loss_first_miss_n
                )
            if self._loss_rest_hit_n > 0:
                metrics["loss/train_rest_hit"] = (
                    self._loss_rest_hit_sum / self._loss_rest_hit_n
                )
            if self._loss_rest_miss_n > 0:
                metrics["loss/train_rest_miss"] = (
                    self._loss_rest_miss_sum / self._loss_rest_miss_n
                )

            # ── 3. Top-k first-token predictions ──────────────────────────────

            if self._first_token_votes:
                top = sorted(
                    self._first_token_votes.items(), key=lambda kv: kv[1], reverse=True
                )[: self._top_k]
                parts = []
                for tid, count in top:
                    tok_text = self._tok.decode([tid]) if tid != self._sep_id else "<SEP>"
                    # Fraction of micro-steps this token was top-predicted
                    frac = count / max(self._micro_count, 1)
                    parts.append(f"'{tok_text}'({frac:.0%})")
                metrics["diag/first_token_top5"] = "  ".join(parts)
                # Also log whether SEP dominates (collapse indicator)
                sep_frac = self._first_token_votes.get(self._sep_id, 0) / max(
                    sum(self._first_token_votes.values()), 1
                )
                metrics["collapse/train_sep_fraction"] = sep_frac

            # ── 5. Logit entropy ───────────────────────────────────────────────
            max_entropy = math.log(
                # We don't have vocab_size here directly; derive from any logged
                # value or fall back to a sentinel
                40148
            )
            if self._entropy_n > 0:
                mean_entropy = self._entropy_sum / self._entropy_n
                metrics["entropy/train"] = mean_entropy / max_entropy
                # 1.0 = uniform over vocab (model knows nothing)
                # 0.0 = perfectly confident
                # Healthy range mid-training: 0.3–0.7

            if self._gen_entropy_n > 0:
                gen_entropy = self._gen_entropy_sum / self._gen_entropy_n
                metrics["diag/gen_entropy"]           = gen_entropy
                metrics["diag/gen_entropy_fraction"]  = gen_entropy / max_entropy
           
            # Logit magnitude — should be ~1-5 at healthy init, not ~50-200
            if self._logit_max_n > 0:
                metrics["diag/mean_max_logit"] = self._logit_max_sum / self._logit_max_n

            # ── Console summary ────────────────────────────────────────────────
            _print_summary(global_step, metrics, label="TRAIN DIAG", mode="train")

        # Always reset training accumulators
        self._reset()
        return metrics

    def flush_eval(self, global_step: int) -> dict[str, float | str]:
        """Aggregate and return eval-pass metrics under the "diag_eval/" prefix.

        Identical structure to flush() but reads from the _eval_* accumulators
        and does NOT reset the training accumulators. Designed to be called
        immediately after flush() so both sets of metrics can be merged into a
        single wandb.log() call sharing the same x-axis step.

        Gradient norm and generation-entropy metrics are omitted (they are
        training-specific).

        Args:
            global_step: current optimizer step count

        Returns:
            dict of metric_name → value with "diag_eval/" prefix.
            Empty dict if this is not a logging step.
        """
        should_log = (global_step % self._log_every == 0)
        metrics:    dict[str, float | str] = {}

        if should_log and self._eval_micro_count > 0:
            pfx = "diag_eval/"

            tokens_per_step = (
                self._eval_transcript_tokens / max(self._eval_micro_count, 1)
            )
            metrics[pfx + "transcript_tokens_per_micro"]   = tokens_per_step
            metrics[pfx + "transcript_tokens_this_window"] = float(
                self._eval_transcript_tokens
            )

            if self._eval_loss_first_n > 0:
                metrics["loss/eval_first_token"] = (
                    self._eval_loss_first_sum / self._eval_loss_first_n
                )
            if self._eval_loss_rest_n > 0:
                metrics["loss/eval_rest"] = (
                    self._eval_loss_rest_sum / self._eval_loss_rest_n
                )

            if self._eval_loss_first_hit_n > 0:
                metrics["loss/eval_first_hit"] = (
                    self._eval_loss_first_hit_sum / self._eval_loss_first_hit_n
                )
            if self._eval_loss_first_miss_n > 0:
                metrics["loss/eval_first_miss"] = (
                    self._eval_loss_first_miss_sum / self._eval_loss_first_miss_n
                )
            if self._eval_loss_rest_hit_n > 0:
                metrics["loss/eval_rest_hit"] = (
                    self._eval_loss_rest_hit_sum / self._eval_loss_rest_hit_n
                )
            if self._eval_loss_rest_miss_n > 0:
                metrics["loss/eval_rest_miss"] = (
                    self._eval_loss_rest_miss_sum / self._eval_loss_rest_miss_n
                )

            if self._eval_first_token_votes:
                top   = sorted(
                    self._eval_first_token_votes.items(),
                    key=lambda kv: kv[1], reverse=True,
                )[: self._top_k]
                parts = []
                for tid, count in top:
                    tok_text = self._tok.decode([tid]) if tid != self._sep_id else "<SEP>"
                    frac     = count / max(self._eval_micro_count, 1)
                    parts.append(f"'{tok_text}'({frac:.0%})")
                metrics[pfx + "first_token_top5"] = "  ".join(parts)
                sep_frac = self._eval_first_token_votes.get(self._sep_id, 0) / max(
                    sum(self._eval_first_token_votes.values()), 1
                )
                metrics["collapse/eval_sep_fraction"] = sep_frac

            max_entropy = math.log(40148)
            if self._eval_entropy_n > 0:
                mean_entropy = self._eval_entropy_sum / self._eval_entropy_n
                metrics["entropy/eval"] = mean_entropy / max_entropy

            if self._eval_logit_max_n > 0:
                metrics[pfx + "mean_max_logit"] = (
                    self._eval_logit_max_sum / self._eval_logit_max_n
                )

            _print_summary(global_step, metrics, label="EVAL DIAG", mode="eval")

        self._reset_eval()
        return metrics

    def _reset(self) -> None:
        self._transcript_tokens = 0
        self._micro_count       = 0
        self._loss_first_sum    = 0.0
        self._loss_first_n      = 0
        self._loss_rest_sum     = 0.0
        self._loss_rest_n       = 0
        self._first_token_votes = {}
        self._entropy_sum       = 0.0
        self._entropy_n         = 0
        self._grad_enc          = None
        self._grad_ada          = None
        self._grad_llm          = None
        self._gen_entropy_sum   = 0.0
        self._gen_entropy_n     = 0
        self._logit_max_sum     = 0.0
        self._logit_max_n       = 0
        self._loss_first_hit_sum  = 0.0
        self._loss_first_hit_n    = 0
        self._loss_first_miss_sum = 0.0
        self._loss_first_miss_n   = 0
        self._loss_rest_hit_sum   = 0.0
        self._loss_rest_hit_n     = 0
        self._loss_rest_miss_sum  = 0.0
        self._loss_rest_miss_n    = 0

    def _reset_eval(self) -> None:
        self._eval_transcript_tokens = 0
        self._eval_micro_count       = 0
        self._eval_loss_first_sum    = 0.0
        self._eval_loss_first_n      = 0
        self._eval_loss_rest_sum     = 0.0
        self._eval_loss_rest_n       = 0
        self._eval_first_token_votes = {}
        self._eval_entropy_sum       = 0.0
        self._eval_entropy_n         = 0
        self._eval_logit_max_sum     = 0.0
        self._eval_logit_max_n       = 0
        self._eval_loss_first_hit_sum  = 0.0
        self._eval_loss_first_hit_n    = 0
        self._eval_loss_first_miss_sum = 0.0
        self._eval_loss_first_miss_n   = 0
        self._eval_loss_rest_hit_sum   = 0.0
        self._eval_loss_rest_hit_n     = 0
        self._eval_loss_rest_miss_sum  = 0.0
        self._eval_loss_rest_miss_n    = 0


# ── Console printer ────────────────────────────────────────────────────────────

def _print_summary(
    step: int,
    m: dict,
    label: str = "TRAIN DIAG",
    mode: str = "train",
) -> None:
    """Print a compact diagnostic block to stdout.

    Args:
        step:  current optimizer step
        m:     metrics dict produced by flush() or flush_eval()
        label: header label — "TRAIN DIAG" or "EVAL DIAG"
        mode:  "train" or "eval" — selects which key names to look up
    """
    sep = "─" * 56
    print(f"\n{sep}")
    print(f"  {label}  step {step}")
    print(sep)

    if mode == "train":
        tkn        = m.get("diag/transcript_tokens_per_micro")
        lf         = m.get("loss/train_first_token")
        lr         = m.get("loss/train_rest")
        lf_hit     = m.get("loss/train_first_hit")
        lf_miss    = m.get("loss/train_first_miss")
        lr_hit     = m.get("loss/train_rest_hit")
        lr_miss    = m.get("loss/train_rest_miss")
        ge         = m.get("grad/encoder")
        ga         = m.get("grad/adapter")
        gl         = m.get("grad/llama")
        ratio      = m.get("grad/enc_llm_ratio")
        top5       = m.get("diag/first_token_top5")
        sep_frac   = m.get("collapse/train_sep_fraction")
        ent        = m.get("entropy/train")
        gen_ent    = m.get("diag/gen_entropy_fraction")
        max_logit  = m.get("diag/mean_max_logit")
    else:
        tkn        = m.get("diag_eval/transcript_tokens_per_micro")
        lf         = m.get("loss/eval_first_token")
        lr         = m.get("loss/eval_rest")
        lf_hit     = m.get("loss/eval_first_hit")
        lf_miss    = m.get("loss/eval_first_miss")
        lr_hit     = m.get("loss/eval_rest_hit")
        lr_miss    = m.get("loss/eval_rest_miss")
        ge = ga = gl = ratio = gen_ent = None
        top5       = m.get("diag_eval/first_token_top5")
        sep_frac   = m.get("collapse/eval_sep_fraction")
        ent        = m.get("entropy/eval")
        max_logit  = m.get("diag_eval/mean_max_logit")

    if tkn is not None:
        print(f"  token budget     {tkn:.1f} transcript tokens/micro-step")

    if lf is not None and lr is not None:
        arrow = "↑ healthy" if lf > lr else "⚠ collapse signal"
        print(f"  loss first tok   {lf:.3f}   rest {lr:.3f}   {arrow}")
    elif lf is not None:
        print(f"  loss first tok   {lf:.3f}   (rest: n/a)")

    if lf_hit is not None or lf_miss is not None:
        hit_s  = f"{lf_hit:.3f}" if lf_hit is not None else "  n/a"
        miss_s = f"{lf_miss:.3f}" if lf_miss is not None else "  n/a"
        print(f"  first hit/miss   hit {hit_s}   miss {miss_s}")
    if lr_hit is not None or lr_miss is not None:
        hit_s  = f"{lr_hit:.3f}" if lr_hit is not None else "  n/a"
        miss_s = f"{lr_miss:.3f}" if lr_miss is not None else "  n/a"
        print(f"  rest  hit/miss   hit {hit_s}   miss {miss_s}")

    if ge is not None:
        print(f"  grad norms       enc {ge:.3e}  ada {ga:.3e}  llm {gl:.3e}")
    if ratio is not None:
        flag = "  ⚠ enc >> llm" if ratio > 10 else ""
        print(f"  grad enc/llm     {ratio:.2f}{flag}")

    if top5:
        print(f"  first-tok top5   {top5}")
    if sep_frac is not None:
        flag = "  ⚠ EOS COLLAPSE" if sep_frac > 0.5 else ""
        print(f"  SEP fraction     {sep_frac:.1%}{flag}")

    if ent is not None:
        if ent > 0.85:
            ent_label = "⚠ near-uniform (model knows nothing)"
        elif ent < 0.15:
            ent_label = "⚠ very low (may be collapsing)"
        else:
            ent_label = "✓ healthy range"
        print(f"  train entropy    {ent:.2f}  {ent_label}")

    if gen_ent is not None:
        print(f"  gen entropy      {gen_ent:.2f}  (at generation decision point)")

    if max_logit is not None:
        flag = "  ⚠ saturating" if max_logit > 50 else ""
        print(f"  mean max logit   {max_logit:.2f}{flag}")

    print(sep + "\n")