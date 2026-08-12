"""Metric collector for speech-llm training and evaluation.

Architecture
------------
Metric objects are tagged with an *input family* — the type of data they consume:
  "logits"     — (logits, labels) from a teacher-forced forward pass
  "loss"       — scalar loss only (no-logits fallback; currently no metrics use it)
  "grads"      — model modules read after backward; train phase only
  "generation" — autoregressive decode outputs; train phase only (GenEntropyMetric)

Each metric declares:
  name               list[str]   wandb keys it emits
  input_family       str         one of the four families above
  phases             set[str]    subset of {"train", "eval"}
  emit_period        int         optimizer steps between emissions (train only)
  compute_only_on_emit bool      if True, collector skips update() on non-emit steps
                                 (future optimization seam; default False)

MetricCollector.observe() routes raw inputs to the matching family, computing
logits-family shared intermediates exactly once per call (one GPU→CPU transfer).

MetricCollector.flush(phase, step) emits due metrics, resets them, returns the
merged dict. render(metrics, mode, step) prints the console block; the collector
never prints.

Key-set parity with diagnostics.py
------------------------------------
This file emits the identical key set as diagnostics.py flush() / flush_eval(),
with one addition: diag_eval/transcript_tokens_this_window is present in
diagnostics.py flush_eval() but was omitted from the task spec — it is included
here to preserve parity.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from data import PrunedTokenizer

# Ground truth is data/pruned_tokenizer/pruned_config.json — build.py reads
# vocab_size from there. This copy exists only to normalise entropy into a 0-1
# fraction; update it whenever tools/build_vocab.py rebuilds the vocabulary.
# (40,148 for the vocab built from the BasicTextNormalizer/spaCy labels;
#  40,034 for the LLM-labelled corpus.)
_VOCAB_SIZE  = 40034
_MAX_ENTROPY = math.log(_VOCAB_SIZE)


# ── Shared intermediates for the logits family ────────────────────────────────

class _LogitsInter:
    """All shared intermediates computed once per observe() call."""

    __slots__ = (
        "shift_logits", "shift_labels", "unmasked",
        "per_token_loss", "top1_preds", "entropy",
        "n_unmasked", "max_logit_mean", "B", "V",
    )

    def __init__(
        self,
        shift_logits:   torch.Tensor,
        shift_labels:   torch.Tensor,
        unmasked:       torch.Tensor,
        per_token_loss: torch.Tensor,
        top1_preds:     torch.Tensor,
        entropy:        torch.Tensor,
        n_unmasked:     int,
        max_logit_mean: float,
        B: int,
        V: int,
    ) -> None:
        self.shift_logits   = shift_logits
        self.shift_labels   = shift_labels
        self.unmasked       = unmasked
        self.per_token_loss = per_token_loss
        self.top1_preds     = top1_preds
        self.entropy        = entropy
        self.n_unmasked     = n_unmasked
        self.max_logit_mean = max_logit_mean
        self.B              = B
        self.V              = V


def _compute_inter(logits: torch.Tensor, labels: torch.Tensor) -> _LogitsInter:
    """Single .detach().float().cpu() transfer; all ops after are pure-CPU."""
    logits_cpu = logits.detach().float().cpu()
    labels_cpu = labels.detach().cpu()
    B, L, V    = logits_cpu.shape

    max_logit_mean = float(logits_cpu.abs().max(dim=-1).values.mean().item())

    shift_logits = logits_cpu[:, :-1, :]    # (B, L-1, V)
    shift_labels = labels_cpu[:, 1:]         # (B, L-1)
    unmasked     = (shift_labels != -100)    # (B, L-1) bool
    n_unmasked   = int(unmasked.sum().item())

    per_token_loss = F.cross_entropy(
        shift_logits.reshape(-1, V),
        shift_labels.reshape(-1).clamp(min=0),
        reduction="none",
    ).reshape(B, L - 1)

    top1_preds = shift_logits.argmax(dim=-1)   # (B, L-1)

    if n_unmasked > 0:
        ul        = shift_logits[unmasked]
        log_probs = F.log_softmax(ul, dim=-1)
        probs     = log_probs.exp()
        entropy   = -(probs * log_probs).sum(dim=-1)
    else:
        entropy = torch.zeros(0)

    return _LogitsInter(
        shift_logits=shift_logits,
        shift_labels=shift_labels,
        unmasked=unmasked,
        per_token_loss=per_token_loss,
        top1_preds=top1_preds,
        entropy=entropy,
        n_unmasked=n_unmasked,
        max_logit_mean=max_logit_mean,
        B=B,
        V=V,
    )


# ── Metric base ───────────────────────────────────────────────────────────────

class _Metric:
    """Minimal interface every metric object must satisfy."""

    name:                 list[str] = []
    input_family:         str       = ""
    phases:               set[str]  = set()
    emit_period:          int       = 1
    compute_only_on_emit: bool      = False

    def update(self, **kwargs) -> None:
        raise NotImplementedError

    def compute(self) -> dict[str, float | str]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


# ── Concrete metrics ──────────────────────────────────────────────────────────

class GradNormMetric(_Metric):
    """Per-component gradient L2 norms and encoder/llm ratio.

    llama is split by parameter name into the gated audio adapters
    ('audio_adapter' substring → grad/audio_ada) and everything else
    (grad/llama).  Excluding the adapters from grad/llama means a frozen
    backbone reports grad/llama == 0 even while the adapters train, so the two
    numbers stay independently meaningful.
    """

    name         = ["grad/encoder", "grad/adapter", "grad/llama",
                    "grad/audio_ada", "grad/enc_llm_ratio"]
    input_family = "grads"
    phases       = {"train"}
    emit_period  = 1

    def __init__(self) -> None:
        self._enc:       float | None = None
        self._ada:       float | None = None
        self._llm:       float | None = None
        self._audio_ada: float | None = None

    def update(self, *, encoder: nn.Module, adapter: nn.Module, llama: nn.Module) -> None:
        def _norm(params) -> float:
            total = 0.0
            for p in params:
                if p.grad is not None:
                    total += p.grad.detach().float().norm().item() ** 2
            return math.sqrt(total)

        # Split llama's params by name: gated audio adapters vs pretrained rest.
        aa_params  = [p for n, p in llama.named_parameters() if "audio_adapter" in n]
        llm_params = [p for n, p in llama.named_parameters() if "audio_adapter" not in n]

        self._enc       = _norm(encoder.parameters())
        self._ada       = _norm(adapter.parameters())
        self._llm       = _norm(llm_params)
        self._audio_ada = _norm(aa_params)

    def compute(self) -> dict[str, float | str]:
        if self._enc is None:
            return {}
        denom = self._llm if (self._llm is not None and self._llm > 1e-12) else 1e-12
        return {
            "grad/encoder":       self._enc,
            "grad/adapter":       self._ada,
            "grad/llama":         self._llm,
            "grad/audio_ada":     self._audio_ada,
            "grad/enc_llm_ratio": self._enc / denom,
        }

    def reset(self) -> None:
        self._enc = self._ada = self._llm = self._audio_ada = None


class TokenBudgetMetric(_Metric):
    """Transcript token count: per micro-step average and window total."""

    input_family         = "logits"
    compute_only_on_emit = False

    def __init__(self, phase: str, emit_period: int) -> None:
        self.phases      = {phase}
        self.emit_period = emit_period
        _p               = "diag" if phase == "train" else "diag_eval"
        self._k_per      = f"{_p}/transcript_tokens_per_micro"
        self._k_win      = f"{_p}/transcript_tokens_this_window"
        self.name        = [self._k_per, self._k_win]
        self._total:  int = 0
        self._micros: int = 0

    def update(self, *, inter: _LogitsInter) -> None:
        self._total  += inter.n_unmasked
        self._micros += 1

    def compute(self) -> dict[str, float | str]:
        return {
            self._k_per: self._total / max(self._micros, 1),
            self._k_win: float(self._total),
        }

    def reset(self) -> None:
        self._total = self._micros = 0


class LossDecompositionMetric(_Metric):
    """Per-position loss: first transcript token vs rest, each split by hit/miss."""

    input_family         = "logits"
    compute_only_on_emit = False

    def __init__(self, phase: str, emit_period: int) -> None:
        self.phases      = {phase}
        self.emit_period = emit_period
        _s               = "train" if phase == "train" else "eval"
        self._s          = _s
        self.name        = [
            f"loss/{_s}_first_token", f"loss/{_s}_rest",
            f"loss/{_s}_first_hit",   f"loss/{_s}_first_miss",
            f"loss/{_s}_rest_hit",    f"loss/{_s}_rest_miss",
        ]
        self.reset()

    def update(self, *, inter: _LogitsInter) -> None:
        for i in range(inter.B):
            row_mask = inter.unmasked[i]
            nz = row_mask.nonzero(as_tuple=False)
            if len(nz) == 0:
                continue

            first_pos  = int(nz[0].item())
            first_loss = float(inter.per_token_loss[i, first_pos].item())
            self._first_sum += first_loss
            self._first_n   += 1

            first_hit = (
                int(inter.top1_preds[i, first_pos].item()) ==
                int(inter.shift_labels[i, first_pos].item())
            )
            if first_hit:
                self._fhit_sum += first_loss
                self._fhit_n   += 1
            else:
                self._fmiss_sum += first_loss
                self._fmiss_n   += 1

            rest_positions = nz[1:]
            if len(rest_positions) > 0:
                rest_loss = float(inter.per_token_loss[i, rest_positions].mean().item())
                self._rest_sum += rest_loss
                self._rest_n   += 1

                for rp in rest_positions:
                    rp_idx = int(rp.item())
                    r_loss = float(inter.per_token_loss[i, rp_idx].item())
                    r_hit  = (
                        int(inter.top1_preds[i, rp_idx].item()) ==
                        int(inter.shift_labels[i, rp_idx].item())
                    )
                    if r_hit:
                        self._rhit_sum += r_loss
                        self._rhit_n   += 1
                    else:
                        self._rmiss_sum += r_loss
                        self._rmiss_n   += 1

    def compute(self) -> dict[str, float | str]:
        s   = self._s
        out: dict[str, float | str] = {}
        if self._first_n > 0:
            out[f"loss/{s}_first_token"] = self._first_sum / self._first_n
        if self._rest_n > 0:
            out[f"loss/{s}_rest"]        = self._rest_sum  / self._rest_n
        if self._fhit_n > 0:
            out[f"loss/{s}_first_hit"]   = self._fhit_sum  / self._fhit_n
        if self._fmiss_n > 0:
            out[f"loss/{s}_first_miss"]  = self._fmiss_sum / self._fmiss_n
        if self._rhit_n > 0:
            out[f"loss/{s}_rest_hit"]    = self._rhit_sum  / self._rhit_n
        if self._rmiss_n > 0:
            out[f"loss/{s}_rest_miss"]   = self._rmiss_sum / self._rmiss_n
        return out

    def reset(self) -> None:
        self._first_sum = self._rest_sum  = 0.0
        self._first_n   = self._rest_n    = 0
        self._fhit_sum  = self._fmiss_sum = 0.0
        self._fhit_n    = self._fmiss_n   = 0
        self._rhit_sum  = self._rmiss_sum = 0.0
        self._rhit_n    = self._rmiss_n   = 0


class FirstTokenVotesMetric(_Metric):
    """Top-k vote accumulator for the first transcript token prediction."""

    input_family         = "logits"
    compute_only_on_emit = False

    def __init__(
        self,
        phase:        str,
        emit_period:  int,
        top_k:        int,
        tokenizer:    "PrunedTokenizer",
        sep_token_id: int,
    ) -> None:
        self.phases      = {phase}
        self.emit_period = emit_period
        self._top_k      = top_k
        self._tok        = tokenizer
        self._sep_id     = sep_token_id
        _s               = "train" if phase == "train" else "eval"
        _p               = "diag"  if phase == "train" else "diag_eval"
        self._k_top5     = f"{_p}/first_token_top5"
        self._k_sep      = f"collapse/{_s}_sep_fraction"
        self.name        = [self._k_top5, self._k_sep]
        self._votes:  dict[int, int] = {}
        self._micros: int            = 0

    def update(self, *, inter: _LogitsInter) -> None:
        self._micros += 1
        for i in range(inter.B):
            row_mask = inter.unmasked[i]
            nz = row_mask.nonzero(as_tuple=False)
            if len(nz) == 0:
                continue
            first_pos = int(nz[0].item())
            if first_pos > 0:
                pred_logit = inter.shift_logits[i, first_pos - 1]
                for tid in pred_logit.topk(self._top_k).indices.tolist():
                    self._votes[tid] = self._votes.get(tid, 0) + 1

    def compute(self) -> dict[str, float | str]:
        if not self._votes:
            return {}
        top = sorted(self._votes.items(), key=lambda kv: kv[1], reverse=True)[: self._top_k]
        parts = []
        for tid, count in top:
            tok_text = self._tok.decode([tid]) if tid != self._sep_id else "<SEP>"
            frac     = count / max(self._micros, 1)
            parts.append(f"'{tok_text}'({frac:.0%})")
        sep_frac = self._votes.get(self._sep_id, 0) / max(sum(self._votes.values()), 1)
        return {
            self._k_top5: "  ".join(parts),
            self._k_sep:  sep_frac,
        }

    def reset(self) -> None:
        self._votes  = {}
        self._micros = 0


class EntropyMetric(_Metric):
    """Mean logit entropy at transcript positions, normalized by log(vocab_size)."""

    input_family         = "logits"
    compute_only_on_emit = False

    def __init__(self, phase: str, emit_period: int) -> None:
        self.phases      = {phase}
        self.emit_period = emit_period
        _s               = "train" if phase == "train" else "eval"
        self._key        = f"entropy/{_s}"
        self.name        = [self._key]
        self._sum: float = 0.0
        self._n:   int   = 0

    def update(self, *, inter: _LogitsInter) -> None:
        if inter.n_unmasked > 0:
            self._sum += float(inter.entropy.sum().item())
            self._n   += inter.n_unmasked

    def compute(self) -> dict[str, float | str]:
        if self._n == 0:
            return {}
        return {self._key: (self._sum / self._n) / _MAX_ENTROPY}

    def reset(self) -> None:
        self._sum = 0.0
        self._n   = 0


class MaxLogitMetric(_Metric):
    """Mean of per-(batch,position) absolute max logit — saturation indicator."""

    input_family         = "logits"
    compute_only_on_emit = False

    def __init__(self, phase: str, emit_period: int) -> None:
        self.phases      = {phase}
        self.emit_period = emit_period
        _p               = "diag" if phase == "train" else "diag_eval"
        self._key        = f"{_p}/mean_max_logit"
        self.name        = [self._key]
        self._sum: float = 0.0
        self._n:   int   = 0

    def update(self, *, inter: _LogitsInter) -> None:
        self._sum += inter.max_logit_mean
        self._n   += 1

    def compute(self) -> dict[str, float | str]:
        if self._n == 0:
            return {}
        return {self._key: self._sum / self._n}

    def reset(self) -> None:
        self._sum = 0.0
        self._n   = 0


class GenEntropyMetric(_Metric):
    """Logit entropy at the autoregressive generation decision point.

    Mirrors diagnostics.record_generation_entropy. Fed via
    MetricCollector.observe_generation(), not the logits-family path.
    """

    input_family         = "generation"
    phases               = {"train"}
    compute_only_on_emit = False

    def __init__(self, emit_period: int) -> None:
        self.emit_period = emit_period
        self.name        = ["diag/gen_entropy", "diag/gen_entropy_fraction"]
        self._sum: float = 0.0
        self._n:   int   = 0

    def update(
        self,
        *,
        logits:        torch.Tensor,   # (B, S, V)
        logit_indices: torch.Tensor,   # (B,)
    ) -> None:
        B   = logits.shape[0]
        idx = logit_indices.unsqueeze(-1).unsqueeze(-1).expand(B, 1, logits.shape[-1])
        sel = logits.gather(dim=1, index=idx).squeeze(1)   # (B, V)
        sel = sel.detach().float().cpu()
        log_probs = F.log_softmax(sel, dim=-1)
        probs     = log_probs.exp()
        entropy   = -(probs * log_probs).sum(dim=-1)       # (B,)
        self._sum += float(entropy.sum().item())
        self._n   += B

    def compute(self) -> dict[str, float | str]:
        if self._n == 0:
            return {}
        ge = self._sum / self._n
        return {
            "diag/gen_entropy":          ge,
            "diag/gen_entropy_fraction": ge / _MAX_ENTROPY,
        }

    def reset(self) -> None:
        self._sum = 0.0
        self._n   = 0


# ── Collector ─────────────────────────────────────────────────────────────────

class MetricCollector:
    """Routes training/eval inputs to metric objects and flushes on schedule.

    Logits-family shared intermediates are computed once per observe() call
    (one GPU→CPU transfer) and passed to all matching metrics.

    Typical usage::

        collector = MetricCollector(tokenizer, sep_token_id, log_every=10)

        # micro-step loop, after forward + backward:
        collector.observe("train", step, logits=logits, labels=labels)
        # after optimizer step:
        collector.observe("train", step, encoder=encoder, adapter=adapter, llama=llama)

        out = collector.flush("train", step)
        if out:
            render(out, "train", step)
            wandb.log(out, step=step)

        # eval pass:
        for logits, labels, _ in eval_loader:
            collector.observe("eval", step, logits=logits, labels=labels)
        eval_out = collector.flush("eval", step)
        render(eval_out, "eval", step)
    """

    def __init__(
        self,
        tokenizer:    "PrunedTokenizer",
        sep_token_id: int,
        log_every:    int = 10,
        top_k:        int = 5,
    ) -> None:
        self._log_every = log_every

        self._metrics: list[_Metric] = [
            # ── grads: train only, emit every optimizer step ───────────────────
            GradNormMetric(),
            # ── logits: train, emit every log_every steps ─────────────────────
            TokenBudgetMetric("train",   log_every),
            LossDecompositionMetric("train", log_every),
            FirstTokenVotesMetric("train", log_every, top_k, tokenizer, sep_token_id),
            EntropyMetric("train",        log_every),
            MaxLogitMetric("train",       log_every),
            # ── generation entropy: train, same cadence as logits metrics ──────
            GenEntropyMetric(log_every),
            # ── logits: eval, emit every eval pass ────────────────────────────
            TokenBudgetMetric("eval",    1),
            LossDecompositionMetric("eval",  1),
            FirstTokenVotesMetric("eval",  1, top_k, tokenizer, sep_token_id),
            EntropyMetric("eval",         1),
            MaxLogitMetric("eval",        1),
        ]

    def observe(
        self,
        phase:   str,
        step:    int,
        *,
        logits:  torch.Tensor | None = None,
        labels:  torch.Tensor | None = None,
        loss:    float | None        = None,
        encoder: nn.Module | None    = None,
        adapter: nn.Module | None    = None,
        llama:   nn.Module | None    = None,
    ) -> None:
        """Route inputs to all eligible metrics for this phase.

        Args:
            phase:   "train" or "eval"
            step:    current optimizer step (used by compute_only_on_emit gate)
            logits:  (B, L, V) teacher-forced logits — triggers logits family
            labels:  (B, L)   required when logits is provided
            loss:    scalar float — triggers loss family (currently no metrics)
            encoder/adapter/llama: nn.Module — triggers grads family
        """
        if logits is not None and labels is not None:
            inter = _compute_inter(logits, labels)
            for m in self._metrics:
                if m.input_family != "logits" or phase not in m.phases:
                    continue
                if m.compute_only_on_emit and step % m.emit_period != 0:
                    continue
                m.update(inter=inter)

        if loss is not None:
            for m in self._metrics:
                if m.input_family == "loss" and phase in m.phases:
                    m.update(loss=loss)

        if encoder is not None:
            for m in self._metrics:
                if m.input_family == "grads" and phase in m.phases:
                    m.update(encoder=encoder, adapter=adapter, llama=llama)

    def observe_generation(
        self,
        phase:         str,
        step:          int,
        *,
        logits:        torch.Tensor,
        logit_indices: torch.Tensor,
    ) -> None:
        """Route generation-time logits to generation-family metrics.

        Kept separate from observe() so the micro-step hot path is not
        burdened with a generation-family dispatch on every step.

        Args:
            logits:        (B, S, V) from llama.forward() during greedy decode
            logit_indices: (B,) per-sample index for the next-token logit
        """
        for m in self._metrics:
            if m.input_family != "generation" or phase not in m.phases:
                continue
            if m.compute_only_on_emit and step % m.emit_period != 0:
                continue
            m.update(logits=logits, logit_indices=logit_indices)

    def flush(self, phase: str, step: int) -> dict[str, float | str]:
        """Emit due metrics, reset all metrics for this phase, return merged dict.

        Train: metrics emit when step % emit_period == 0.
        Eval:  all eval metrics emit unconditionally (caller controls frequency).

        Metrics are always reset after flush regardless of emission, matching
        diagnostics.py behavior (one optimizer-step accumulation window).

        Args:
            phase: "train" or "eval"
            step:  current optimizer step

        Returns:
            Merged {key: value} dict for all emitted metrics this call.
        """
        out: dict[str, float | str] = {}
        for m in self._metrics:
            if phase not in m.phases:
                continue
            should_emit = (phase == "eval") or (step % m.emit_period == 0)
            if should_emit:
                out.update(m.compute())
            m.reset()
        return out


# ── Console renderer ──────────────────────────────────────────────────────────

def render(metrics: dict, mode: str, step: int) -> None:
    """Print the diagnostic console block. Call after flush().

    Reproduces diagnostics._print_summary output exactly.

    Args:
        metrics: dict returned by MetricCollector.flush()
        mode:    "train" or "eval"
        step:    current optimizer step
    """
    label = "TRAIN DIAG" if mode == "train" else "EVAL DIAG"
    _print_summary(step, metrics, label=label, mode=mode)


def _print_summary(step: int, m: dict, label: str, mode: str) -> None:
    sep = "─" * 56
    print(f"\n{sep}")
    print(f"  {label}  step {step}")
    print(sep)

    if mode == "train":
        tkn       = m.get("diag/transcript_tokens_per_micro")
        lf        = m.get("loss/train_first_token")
        lr        = m.get("loss/train_rest")
        lf_hit    = m.get("loss/train_first_hit")
        lf_miss   = m.get("loss/train_first_miss")
        lr_hit    = m.get("loss/train_rest_hit")
        lr_miss   = m.get("loss/train_rest_miss")
        ge        = m.get("grad/encoder")
        ga        = m.get("grad/adapter")
        gl        = m.get("grad/llama")
        gaa       = m.get("grad/audio_ada")
        ratio     = m.get("grad/enc_llm_ratio")
        top5      = m.get("diag/first_token_top5")
        sep_frac  = m.get("collapse/train_sep_fraction")
        ent       = m.get("entropy/train")
        gen_ent   = m.get("diag/gen_entropy_fraction")
        max_logit = m.get("diag/mean_max_logit")
    else:
        tkn       = m.get("diag_eval/transcript_tokens_per_micro")
        lf        = m.get("loss/eval_first_token")
        lr        = m.get("loss/eval_rest")
        lf_hit    = m.get("loss/eval_first_hit")
        lf_miss   = m.get("loss/eval_first_miss")
        lr_hit    = m.get("loss/eval_rest_hit")
        lr_miss   = m.get("loss/eval_rest_miss")
        ge = ga = gl = gaa = ratio = gen_ent = None
        top5      = m.get("diag_eval/first_token_top5")
        sep_frac  = m.get("collapse/eval_sep_fraction")
        ent       = m.get("entropy/eval")
        max_logit = m.get("diag_eval/mean_max_logit")

    if tkn is not None:
        print(f"  token budget     {tkn:.1f} transcript tokens/micro-step")

    if lf is not None and lr is not None:
        arrow = "↑ healthy" if lf > lr else "⚠ collapse signal"
        print(f"  loss first tok   {lf:.3f}   rest {lr:.3f}   {arrow}")
    elif lf is not None:
        print(f"  loss first tok   {lf:.3f}   (rest: n/a)")

    if lf_hit is not None or lf_miss is not None:
        hit_s  = f"{lf_hit:.3f}"  if lf_hit  is not None else "  n/a"
        miss_s = f"{lf_miss:.3f}" if lf_miss is not None else "  n/a"
        print(f"  first hit/miss   hit {hit_s}   miss {miss_s}")
    if lr_hit is not None or lr_miss is not None:
        hit_s  = f"{lr_hit:.3f}"  if lr_hit  is not None else "  n/a"
        miss_s = f"{lr_miss:.3f}" if lr_miss is not None else "  n/a"
        print(f"  rest  hit/miss   hit {hit_s}   miss {miss_s}")

    if ge is not None:
        line = f"  grad norms       enc {ge:.3e}  ada {ga:.3e}  llm {gl:.3e}"
        if gaa is not None:
            line += f"  aud {gaa:.3e}"
        print(line)
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


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # ── Fake tokenizer ────────────────────────────────────────────────────────
    class _FakeTok:
        def decode(self, ids: list[int]) -> str:
            return f"w{ids[0]}"

    SEP_ID    = _VOCAB_SIZE - 1
    LOG_EVERY = 10
    B, L, V   = 3, 60, _VOCAB_SIZE

    collector = MetricCollector(
        tokenizer    = _FakeTok(),
        sep_token_id = SEP_ID,
        log_every    = LOG_EVERY,
        top_k        = 5,
    )

    # ── Synthetic data ────────────────────────────────────────────────────────
    # Labels: positions 0..29 masked (-100), positions 30..58 transcript, 59 = SEP
    labels = torch.full((B, L), -100, dtype=torch.long)
    labels[:, 30:59] = torch.randint(0, V - 1, (B, 29))
    labels[:, 59]    = SEP_ID

    # Construct logits with guaranteed hits at first_pos and a mix of
    # hits/misses at rest positions, so all four hit/miss keys always appear.
    #
    # shift space: unmasked positions are t=29..58 (shift_labels[t]=labels[t+1])
    # first_pos=29; rest_positions=30..58
    # Sample 0,1: hit at first_pos. Sample 2: miss at first_pos.
    # Even shift-t in rest: hit. Odd shift-t in rest: miss.
    logits = torch.full((B, L, V), -10.0)
    for b in range(B):
        first_true = int(labels[b, 30].item())
        if b < B - 1:
            logits[b, 29, first_true] = 100.0                     # hit
        else:
            logits[b, 29, (first_true + 1) % (V - 1)] = 100.0    # miss
        for t_shift in range(30, 59):
            true_id = int(labels[b, t_shift + 1].item())
            if t_shift % 2 == 0:
                logits[b, t_shift, true_id] = 100.0               # hit
            else:
                logits[b, t_shift, (true_id + 1) % (V - 1)] = 100.0  # miss

    loss = torch.tensor(3.5)

    # Fake modules with gradients for grad norm
    class _FakeMod(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.randn(8, 8))
            self.w.grad = torch.randn(8, 8)

    encoder = _FakeMod()
    adapter = _FakeMod()
    llama   = _FakeMod()

    # ── Run 10 optimizer steps, each with 4 micro-steps ───────────────────────
    EXPECTED_TRAIN_GRAD_KEYS = {
        "grad/encoder", "grad/adapter", "grad/llama", "grad/audio_ada",
        "grad/enc_llm_ratio",
    }
    EXPECTED_TRAIN_LOG_KEYS = EXPECTED_TRAIN_GRAD_KEYS | {
        "diag/transcript_tokens_per_micro",
        "diag/transcript_tokens_this_window",
        "loss/train_first_token",
        "loss/train_rest",
        "loss/train_first_hit",
        "loss/train_first_miss",
        "loss/train_rest_hit",
        "loss/train_rest_miss",
        "diag/first_token_top5",
        "collapse/train_sep_fraction",
        "entropy/train",
        "diag/gen_entropy",
        "diag/gen_entropy_fraction",
        "diag/mean_max_logit",
    }
    EXPECTED_EVAL_KEYS = {
        "diag_eval/transcript_tokens_per_micro",
        "diag_eval/transcript_tokens_this_window",
        "loss/eval_first_token",
        "loss/eval_rest",
        "loss/eval_first_hit",
        "loss/eval_first_miss",
        "loss/eval_rest_hit",
        "loss/eval_rest_miss",
        "diag_eval/first_token_top5",
        "collapse/eval_sep_fraction",
        "entropy/eval",
        "diag_eval/mean_max_logit",
    }

    for step in range(1, LOG_EVERY + 1):
        for _ in range(4):
            collector.observe("train", step, logits=logits, labels=labels)

        # generation entropy (simulates one generation step per optimizer step)
        gen_logits  = torch.randn(B, 10, V)
        gen_indices = torch.zeros(B, dtype=torch.long)
        collector.observe_generation("train", step, logits=gen_logits, logit_indices=gen_indices)

        collector.observe("train", step, encoder=encoder, adapter=adapter, llama=llama)

        out = collector.flush("train", step)

        # grads emit every step
        missing_grad = EXPECTED_TRAIN_GRAD_KEYS - set(out)
        assert not missing_grad, f"step {step}: missing grad keys: {missing_grad}"

        if step % LOG_EVERY == 0:
            missing = EXPECTED_TRAIN_LOG_KEYS - set(out)
            extra   = set(out) - EXPECTED_TRAIN_LOG_KEYS
            assert not missing, f"step {step} log: missing keys: {missing}"
            assert not extra,   f"step {step} log: unexpected keys: {extra}"
            # entropy in [0, 1]
            ent = out["entropy/train"]
            assert 0.0 <= ent <= 1.0, f"entropy/train={ent} out of [0,1]"
            gen_ent = out["diag/gen_entropy_fraction"]
            assert 0.0 <= gen_ent <= 1.0, f"diag/gen_entropy_fraction={gen_ent} out of [0,1]"
        else:
            non_grad = set(out) - EXPECTED_TRAIN_GRAD_KEYS
            assert not non_grad, f"step {step}: non-log step emitted non-grad keys: {non_grad}"

    # ── Verify reset clears accumulators ──────────────────────────────────────
    # Immediately after flush at step LOG_EVERY, a second flush at step LOG_EVERY+1
    # (non-log step) should yield only grad keys IF grads are observed, else empty.
    out_after_reset = collector.flush("train", LOG_EVERY + 1)
    # No observe() calls before this flush, so grads are None → grad keys absent
    assert set(out_after_reset) == set(), \
        f"after reset, unexpected keys: {set(out_after_reset)}"

    # ── Eval pass ─────────────────────────────────────────────────────────────
    for _ in range(3):
        collector.observe("eval", LOG_EVERY, logits=logits, labels=labels)

    eval_out = collector.flush("eval", LOG_EVERY)

    missing = EXPECTED_EVAL_KEYS - set(eval_out)
    extra   = set(eval_out) - EXPECTED_EVAL_KEYS
    assert not missing, f"eval: missing keys: {missing}"
    assert not extra,   f"eval: unexpected keys: {extra}"

    ent_eval = eval_out["entropy/eval"]
    assert 0.0 <= ent_eval <= 1.0, f"entropy/eval={ent_eval} out of [0,1]"

    # ── Eval emits on every call, not only at log_every steps ─────────────────
    for _ in range(2):
        collector.observe("eval", 3, logits=logits, labels=labels)
    eval_out2 = collector.flush("eval", 3)   # step 3 is NOT a log step
    assert EXPECTED_EVAL_KEYS.issubset(set(eval_out2)), \
        f"eval at non-log step missing keys: {EXPECTED_EVAL_KEYS - set(eval_out2)}"

    # ── Console render (visual check — no assertion) ───────────────────────────
    # Re-run one full log step to get a populated metrics dict to render.
    for _ in range(4):
        collector.observe("train", LOG_EVERY, logits=logits, labels=labels)
    collector.observe_generation(
        "train", LOG_EVERY, logits=torch.randn(B, 10, V),
        logit_indices=torch.zeros(B, dtype=torch.long),
    )
    collector.observe("train", LOG_EVERY, encoder=encoder, adapter=adapter, llama=llama)
    render_out = collector.flush("train", LOG_EVERY)
    render(render_out, "train", LOG_EVERY)

    for _ in range(3):
        collector.observe("eval", LOG_EVERY, logits=logits, labels=labels)
    eval_render_out = collector.flush("eval", LOG_EVERY)
    render(eval_render_out, "eval", LOG_EVERY)

    print("PASSED")
    sys.exit(0)
