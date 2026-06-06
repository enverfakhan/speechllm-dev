"""Optimizer construction helpers for speech-llm training.

Centralises bitsandbytes AdamW8bit construction and the stage-2 param-group
/ LR-scheduler setup so train.py does not duplicate them.
"""

from __future__ import annotations

import torch
import torch.nn as nn


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


def build_stage2_param_groups(
    encoder:    nn.Module,
    adapter:    nn.Module,
    llama:      nn.Module,
    lr_encoder: float,
    lr_adapter: float,
    lr_llama:   float,
) -> list[dict]:
    """Three named param groups for stage-2 joint training.

    Each group carries its own peak LR.  A shared warmup scheduler scales all
    three by the same factor so relative LR ratios are preserved during warmup.
    """
    groups = [
        {"name": "encoder", "params": list(encoder.parameters()), "lr": lr_encoder},
        {"name": "adapter", "params": list(adapter.parameters()), "lr": lr_adapter},
        {"name": "llama",   "params": list(llama.parameters()),   "lr": lr_llama},
    ]
    for g in groups:
        assert len(g["params"]) > 0, (
            f"Param group '{g['name']}' is empty — check model construction."
        )
    return groups


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


def print_stage2_startup(
    param_groups: list[dict],
    warmup_steps: int,
    betas:        tuple[float, float],
) -> None:
    """Print stage-2 optimizer config to stdout so the run is self-documenting."""
    print("─" * 60)
    print("Stage 2 — joint fine-tune config:")
    for g in param_groups:
        n = sum(p.numel() for p in g["params"])
        print(f"  {g['name']:8s}  {n / 1e6:8.1f}M params  peak LR {g['lr']:.2e}")
    print(f"  warmup_steps={warmup_steps}  betas={betas}  weight_decay=0.01")
    print("─" * 60)
