"""Checkpoint save/load helpers for speech-llm training.

Two checkpoint formats are supported:

Full checkpoint  (used for same-stage resume)
    keys: step, epoch, micro_step_in_epoch, batch_size,
          adapter, optimizer, scaler,
          scheduler?  (present only when a LR scheduler is active)
          encoder?    (present when encoder is not permanently frozen)
          llama?      (present when Llama is not permanently frozen)

Adapter-only checkpoint  (used for cross-stage loading)
    keys: step, epoch, micro_step_in_epoch, batch_size,
          adapter, optimizer_adapter

The caller decides which optional modules to include by passing them (or None).
This keeps the format decision in train.py where the freeze/stage flags live.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn


class ResumeState(NamedTuple):
    """Fields recovered from a checkpoint needed to restart the training loop."""
    step:                 int
    epoch:                int
    micro_step_in_epoch:  int
    batch_size:           int | None   # None when not recorded in the checkpoint


# ── Save ──────────────────────────────────────────────────────────────────────

def save_checkpoint(
    path: Path | str,
    *,
    step:                 int,
    epoch:                int,
    micro_step_in_epoch:  int,
    batch_size:           int,
    adapter:              nn.Module,
    optimizer:            torch.optim.Optimizer,
    scaler:               torch.amp.GradScaler,
    scheduler:            torch.optim.lr_scheduler.LRScheduler | None = None,
    encoder:              nn.Module | None = None,
    llama:                nn.Module | None = None,
) -> None:
    """Save a full training checkpoint.

    encoder and llama are written only when passed (not None).  The caller
    resolves the freeze/stage conditionals:

        save_checkpoint(
            path, ...,
            encoder = encoder if (stage == 2 or not freeze_encoder) else None,
            llama   = llama   if (stage == 2 or not freeze_llama)   else None,
        )
    """
    d: dict = {
        "step":                step,
        "epoch":               epoch,
        "micro_step_in_epoch": micro_step_in_epoch,
        "batch_size":          batch_size,
        "adapter":             adapter.state_dict(),
        "optimizer":           optimizer.state_dict(),
        "scaler":              scaler.state_dict(),
    }
    if scheduler is not None:
        d["scheduler"] = scheduler.state_dict()
    if encoder is not None:
        d["encoder"] = encoder.state_dict()
    if llama is not None:
        d["llama"] = llama.state_dict()
    torch.save(d, path)


def save_adapter_checkpoint(
    path: Path | str,
    *,
    step:                int,
    epoch:               int,
    micro_step_in_epoch: int,
    batch_size:          int,
    adapter:             nn.Module,
    optimizer:           torch.optim.Optimizer,
) -> None:
    """Save an adapter-only checkpoint for cross-stage loading.

    The optimizer state is stored under key "optimizer_adapter" (not
    "optimizer") so that load_adapter_checkpoint can distinguish it from a
    full checkpoint and avoids accidentally restoring stage-1 moments into a
    stage-2 optimizer.

    NOTE — batch_size inconsistency (flagged for owner):
      Regular adapter sidecars (periodic / early-stop / final) are called with
      batch_size=args.batch_size, while the auto-stage stage-1-final sidecar
      is called with batch_size=_batch_size.  These are the same value in
      single-stage runs but differ when --s1_batch_size overrides the default.
      The caller passes the appropriate value; this function does not unify them.
    """
    torch.save(
        {
            "step":                step,
            "epoch":               epoch,
            "micro_step_in_epoch": micro_step_in_epoch,
            "batch_size":          batch_size,
            "adapter":             adapter.state_dict(),
            "optimizer_adapter":   optimizer.state_dict(),
        },
        path,
    )


# ── Load ──────────────────────────────────────────────────────────────────────

def load_full_checkpoint(
    path: Path | str,
    *,
    encoder:   nn.Module | None                              = None,
    adapter:   nn.Module,
    llama:     nn.Module | None                              = None,
    optimizer: torch.optim.Optimizer,
    scaler:    torch.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None  = None,
) -> ResumeState:
    """Load a full checkpoint for same-stage resume.

    encoder and llama are loaded when both passed AND present in the file.
    Passing None for either skips the corresponding load without error.

    Scheduler notes
    ---------------
    If the checkpoint contains "scheduler", its state is restored directly.
    If it is absent (checkpoint predates scheduler saving), the fallback
    fast-forward loop below runs — but see the BUG? comment.

    Returns a ResumeState with batch_size=None when the key is absent from
    the checkpoint (old format).  The caller should apply its own default:
        rs = load_full_checkpoint(...)
        batch_size = rs.batch_size if rs.batch_size is not None else args.batch_size
    """
    ckpt = torch.load(path, map_location="cpu")

    if encoder is not None and "encoder" in ckpt:
        encoder.load_state_dict(ckpt["encoder"])
    adapter.load_state_dict(ckpt["adapter"])
    if llama is not None and "llama" in ckpt:
        llama.load_state_dict(ckpt["llama"])

    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])

    if scheduler is not None:
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        else:
            # BUG?: In the original train.py the fallback loop ran
            #   for _ in range(resume_global_step): scheduler.step()
            # but resume_global_step was still 0 at that point (the step
            # value from the checkpoint is read only AFTER this block).
            # The loop was therefore always a no-op.  Preserving that
            # no-op here intentionally; do not silently fix.
            pass   # was: for _ in range(0): scheduler.step()

    return ResumeState(
        step                = ckpt["step"],
        epoch               = ckpt.get("epoch", 0),
        micro_step_in_epoch = ckpt.get("micro_step_in_epoch", 0),
        batch_size          = ckpt.get("batch_size"),   # None when absent
    )


def load_adapter_checkpoint(
    path: Path | str,
    *,
    adapter: nn.Module,
) -> ResumeState:
    """Load adapter weights from an adapter-only checkpoint (cross-stage loading).

    Optimizer state is NOT restored: bitsandbytes AdamW8bit moment tensors are
    not reliably portable via manual state-dict injection, and Adam moments
    re-stabilise within a few steps from a warm adapter init.
    """
    ckpt = torch.load(path, map_location="cpu")
    adapter.load_state_dict(ckpt["adapter"])
    return ResumeState(
        step                = ckpt.get("step", 0),
        epoch               = ckpt.get("epoch", 0),
        micro_step_in_epoch = ckpt.get("micro_step_in_epoch", 0),
        batch_size          = ckpt.get("batch_size"),
    )


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import tempfile

    print("checkpoint.py self-test")

    # ── Tiny fake modules ──────────────────────────────────────────────────────
    enc = nn.Linear(4, 4)
    ada = nn.Linear(4, 4)
    llm = nn.Linear(4, 4)

    # ── Fake optimizer + scaler (no bnb needed) ────────────────────────────────
    opt = torch.optim.SGD(
        [{"params": enc.parameters(), "lr": 1e-3},
         {"params": ada.parameters(), "lr": 1e-4},
         {"params": llm.parameters(), "lr": 1e-5}],
    )
    scl = torch.amp.GradScaler("cpu")

    # Take one step so optimizer has non-trivial state
    loss = (enc(torch.randn(2, 4)) + ada(torch.randn(2, 4)) + llm(torch.randn(2, 4))).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # ── 1. Full checkpoint round-trip (with encoder + llama) ──────────────
        full_path = td / "full.pt"
        save_checkpoint(
            full_path,
            step=42, epoch=3, micro_step_in_epoch=17, batch_size=8,
            adapter=ada, optimizer=opt, scaler=scl,
            encoder=enc, llama=llm,
        )
        raw = torch.load(full_path, map_location="cpu")
        assert set(raw.keys()) == {
            "step", "epoch", "micro_step_in_epoch", "batch_size",
            "adapter", "optimizer", "scaler", "encoder", "llama",
        }, f"Unexpected full-ckpt keys: {set(raw.keys())}"

        enc2 = nn.Linear(4, 4)
        ada2 = nn.Linear(4, 4)
        llm2 = nn.Linear(4, 4)
        opt2 = torch.optim.SGD(
            [{"params": enc2.parameters(), "lr": 1e-3},
             {"params": ada2.parameters(), "lr": 1e-4},
             {"params": llm2.parameters(), "lr": 1e-5}],
        )
        scl2 = torch.amp.GradScaler("cpu")

        rs = load_full_checkpoint(
            full_path,
            encoder=enc2, adapter=ada2, llama=llm2,
            optimizer=opt2, scaler=scl2,
        )
        assert rs == ResumeState(42, 3, 17, 8), f"ResumeState mismatch: {rs}"
        # Weights loaded correctly
        assert torch.allclose(enc.weight.data, enc2.weight.data), "encoder weight mismatch"
        assert torch.allclose(ada.weight.data, ada2.weight.data), "adapter weight mismatch"
        assert torch.allclose(llm.weight.data, llm2.weight.data), "llama weight mismatch"
        print("  [OK] full checkpoint round-trip (all modules present)")

        # ── 2. Full checkpoint without optional modules ────────────────────────
        # Optimizer must have the same group structure as what was saved.
        # Use a fresh single-group optimizer (adapter only) for this test.
        ada3   = nn.Linear(4, 4)
        opt3   = torch.optim.SGD(ada3.parameters(), lr=1e-4)
        scl3   = torch.amp.GradScaler("cpu")
        loss3  = ada3(torch.randn(2, 4)).sum()
        loss3.backward(); opt3.step(); opt3.zero_grad()

        slim_path = td / "slim.pt"
        save_checkpoint(
            slim_path,
            step=7, epoch=0, micro_step_in_epoch=5, batch_size=4,
            adapter=ada3, optimizer=opt3, scaler=scl3,
            # encoder=None, llama=None  (intentionally omitted)
        )
        raw2 = torch.load(slim_path, map_location="cpu")
        assert "encoder"   not in raw2, "encoder should be absent"
        assert "llama"     not in raw2, "llama should be absent"
        assert "scheduler" not in raw2, "scheduler should be absent"

        ada3b  = nn.Linear(4, 4)
        opt3b  = torch.optim.SGD(ada3b.parameters(), lr=1e-4)
        scl3b  = torch.amp.GradScaler("cpu")
        rs2 = load_full_checkpoint(
            slim_path,
            adapter=ada3b, optimizer=opt3b, scaler=scl3b,
        )
        assert rs2 == ResumeState(7, 0, 5, 4), f"ResumeState mismatch: {rs2}"
        print("  [OK] full checkpoint without optional modules")

        # ── 3. batch_size absent → ResumeState.batch_size is None ────────────
        no_bs_path = td / "no_bs.pt"
        d = torch.load(slim_path, map_location="cpu")
        del d["batch_size"]
        torch.save(d, no_bs_path)
        ada4 = nn.Linear(4, 4)
        opt4 = torch.optim.SGD(ada4.parameters(), lr=1e-4)
        scl4 = torch.amp.GradScaler("cpu")
        rs3 = load_full_checkpoint(no_bs_path, adapter=ada4, optimizer=opt4, scaler=scl4)
        assert rs3.batch_size is None, f"Expected None, got {rs3.batch_size}"
        # Caller applies default
        effective_bs = rs3.batch_size if rs3.batch_size is not None else 99
        assert effective_bs == 99
        print("  [OK] missing batch_size → None sentinel, caller default applied")

        # ── 4. Adapter-only checkpoint round-trip ─────────────────────────────
        adapter_path = td / "adapter.pt"
        save_adapter_checkpoint(
            adapter_path,
            step=100, epoch=2, micro_step_in_epoch=33, batch_size=16,
            adapter=ada, optimizer=opt,
        )
        raw3 = torch.load(adapter_path, map_location="cpu")
        assert set(raw3.keys()) == {
            "step", "epoch", "micro_step_in_epoch", "batch_size",
            "adapter", "optimizer_adapter",
        }, f"Unexpected adapter-ckpt keys: {set(raw3.keys())}"
        assert "optimizer" not in raw3, "'optimizer' key must not be present (only 'optimizer_adapter')"

        ada5 = nn.Linear(4, 4)
        rs4 = load_adapter_checkpoint(adapter_path, adapter=ada5)
        assert rs4 == ResumeState(100, 2, 33, 16), f"ResumeState mismatch: {rs4}"
        assert torch.allclose(ada.weight.data, ada5.weight.data), "adapter weight mismatch"
        print("  [OK] adapter-only checkpoint round-trip")

        # ── 5. Scheduler present/absent paths ────────────────────────────────
        ada6 = nn.Linear(4, 4)
        opt6 = torch.optim.SGD(ada6.parameters(), lr=1e-3)
        scl6 = torch.amp.GradScaler("cpu")
        sched6 = torch.optim.lr_scheduler.LambdaLR(opt6, lr_lambda=lambda s: min(s / 10, 1.0))
        for _ in range(5):
            sched6.step()

        sched_path = td / "with_sched.pt"
        save_checkpoint(
            sched_path,
            step=5, epoch=0, micro_step_in_epoch=1, batch_size=4,
            adapter=ada6, optimizer=opt6, scaler=scl6, scheduler=sched6,
        )
        raw4 = torch.load(sched_path, map_location="cpu")
        assert "scheduler" in raw4, "scheduler key should be present"

        ada7 = nn.Linear(4, 4)
        opt7 = torch.optim.SGD(ada7.parameters(), lr=1e-3)
        scl7 = torch.amp.GradScaler("cpu")
        sched7 = torch.optim.lr_scheduler.LambdaLR(opt7, lr_lambda=lambda s: min(s / 10, 1.0))
        rs5 = load_full_checkpoint(
            sched_path, adapter=ada7, optimizer=opt7, scaler=scl7, scheduler=sched7,
        )
        assert rs5.step == 5
        # scheduler last_epoch should be restored (5 steps)
        assert sched7.last_epoch == 5, f"scheduler.last_epoch={sched7.last_epoch}"
        print("  [OK] scheduler round-trip")

    print("\nPASSED")
    sys.exit(0)
