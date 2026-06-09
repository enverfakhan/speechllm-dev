"""Optimizer construction helpers for speech-llm training."""

from __future__ import annotations

import torch


def build_adamw8bit(
    param_groups: list[dict],
    *,
    betas:        tuple[float, float],
    weight_decay: float,
) -> "bitsandbytes.optim.AdamW8bit":
    """Construct a bitsandbytes 8-bit AdamW optimizer.

    Centralises the import and call so every optimizer construction site
    (stage-1 and stage-2, including the auto-stage transition) stays DRY.
    """
    import bitsandbytes as bnb
    return bnb.optim.AdamW8bit(param_groups, betas=betas, weight_decay=weight_decay)


def make_warmup_scheduler(
    optimizer:    torch.optim.Optimizer,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear ramp 0 → peak over warmup_steps, then constant.

    One lambda scales all param groups; per-group peak comes from each group's
    base_lr, so a single factor is correct for all three simultaneously.
    """
    def lr_lambda(step: int) -> float:
        return min(step / max(warmup_steps, 1), 1.0)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


