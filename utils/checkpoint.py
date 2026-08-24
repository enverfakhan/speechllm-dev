"""Checkpoint save/load helpers for speech-llm training.

THE INVARIANT (why checkpoints are small)
-----------------------------------------
A checkpoint is a DELTA over the pretrained base (Whisper ckpt + Llama ckpt +
deterministic fresh inits).  It must contain every parameter that DIFFERS from
that base and may OMIT anything identical to it.  Corollary:

    pretrained base + checkpoint overlay = complete model

…with no dependence on init_from at load time.  Loading is always: build from
pretrained, then overlay whatever keys the checkpoint carries (load_weights /
apply_full_checkpoint skip absent keys — the pretrained value already sits in
the freshly-built model).  This keeps periodic checkpoints off the multi-GB
frozen-Llama floor (~32 GB) that fills a small container disk.

A module is "dirty" (must be saved) when it was trainable in the current OR any
earlier stage of the run, or was overlaid by model.init_from at build time.  The
training loop tracks this union and gates the heavy saves on it.

Save policy
-----------
    adapter          ALWAYS (bridge, ~40 MB; also captures init_from bridge values)
    audio_adapters   ALWAYS when the model owns them AND full llama is not saved
                     (~130 MB at r=128, stored as {param_name: tensor}); when full
                     llama IS saved its state_dict subsumes them — no duplicate.
    optimizer/scaler/scheduler  ALWAYS (optimizer state already covers only
                     trainable params by construction)
    encoder          iff "encoder" in dirty set (~350 MB)
    llama (full)     iff "llama"   in dirty set (~32 GB)

Two checkpoint formats are supported:

Full checkpoint  (used for same-stage resume; also a valid init_from source)
    keys: step, epoch, micro_step_in_epoch, batch_size, kind,
          step_in_stage, stage_index,
          adapter, optimizer, scaler,
          scheduler?       (present only when a LR scheduler is active)
          encoder?         (present iff encoder is dirty)
          llama?           (present iff llama is dirty)
          audio_adapters?  (present iff model has adapters AND llama not saved)
          modules_dirty?   (the dirty set; absent in legacy checkpoints)
          init_from?       (provenance string or None; absent in legacy checkpoints)

Adapter-only checkpoint  (saved at stage boundaries for archival)
    keys: step, epoch, micro_step_in_epoch, batch_size, kind,
          step_in_stage, stage_index,
          adapter, optimizer_adapter,
          audio_adapters?  (present iff the model owns gated audio adapters)

The caller decides which optional modules to include by passing them (or None).
Untagged legacy checkpoints (no "kind" key) are treated as kind="periodic" and
are resumable; their dirty set is inferred from which module keys they carry.

Future work: a consolidation tool could merge pretrained + a delta checkpoint
into one self-contained file if single-file loading is ever needed.
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
    step_in_stage:        int = 0      # stage-local step; defaults to global step for old checkpoints
    stage_index:          int = 0
    # Modules that diverge from the pretrained base (see save_checkpoint invariant).
    # The resumed run keeps saving these.  For legacy checkpoints (no "modules_dirty"
    # key) it is inferred from which full-module states the file carries.
    modules_dirty:        tuple[str, ...] = ()


# ── Backward-compatible llama loading ───────────────────────────────────────────

def _load_llama_state_tolerant(llama: nn.Module, state: dict) -> None:
    """load_state_dict for Llama, tolerating ONLY missing audio_adapter keys.

    Checkpoints written before gated audio adapters existed have no
    'audio_adapter' parameters.  Those specific keys are allowed to be absent —
    the fresh init is kept and one message is printed.  Any OTHER missing key,
    or any unexpected key, remains a hard error: we deliberately do NOT use a
    blanket strict=False, which would silently swallow real load bugs.
    """
    result  = llama.load_state_dict(state, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)

    adapter_missing     = [k for k in missing if "audio_adapter" in k]
    non_adapter_missing = [k for k in missing if "audio_adapter" not in k]

    if non_adapter_missing or unexpected:
        raise RuntimeError(
            f"Llama checkpoint load failed: "
            f"missing keys {non_adapter_missing}, unexpected keys {unexpected}"
        )
    if adapter_missing:
        print(
            f"[load] {len(adapter_missing)} audio-adapter tensors not found in "
            f"checkpoint — keeping fresh init"
        )


# The two learned audio delimiter vectors the chat convention splices around the
# audio (model/adapter.py).  Bridges built before the chat convention existed have
# neither, so their checkpoints legitimately lack these two keys.
_BRIDGE_MARKER_KEYS = ("audio_bos", "audio_eos")


def _load_bridge_state_tolerant(adapter: nn.Module, state: dict) -> None:
    """load_state_dict for the bridge, tolerating ONLY missing marker keys.

    Same contract as _load_llama_state_tolerant: a checkpoint written before the
    audio markers existed keeps its fresh (normal-init) markers and prints one
    message; any OTHER missing key, or any unexpected key, stays a hard error.
    A blanket strict=False would swallow a real bridge-variant mismatch — which
    is exactly the mlp↔swiglu confusion this project wants to fail loudly.
    """
    result     = adapter.load_state_dict(state, strict=False)
    missing    = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)

    marker_missing = [k for k in missing if k in _BRIDGE_MARKER_KEYS]
    other_missing  = [k for k in missing if k not in _BRIDGE_MARKER_KEYS]

    if other_missing or unexpected:
        raise RuntimeError(
            f"Bridge checkpoint load failed: "
            f"missing keys {other_missing}, unexpected keys {unexpected}"
        )
    if marker_missing:
        print(
            f"[load] audio marker(s) {marker_missing} not found in checkpoint "
            "— keeping fresh init (pre-chat-convention bridge)"
        )


def _collect_audio_adapters(llama: nn.Module) -> dict[str, torch.Tensor]:
    """Return {param_name: cpu_tensor} for every gated audio-adapter parameter.

    These live inside llama's state_dict but are always carried as a small,
    separate delta (~130 MB at r=128) so that a checkpoint stays complete over
    the pretrained base even when the multi-gigabyte full llama state is omitted
    (frozen backbone).  Selected by the same 'audio_adapter' substring used
    everywhere else.  Empty dict when the model owns no audio adapters.
    """
    return {
        name: p.detach().cpu()
        for name, p in llama.named_parameters()
        if "audio_adapter" in name
    }


def _overlay_audio_adapters(llama: nn.Module, tensors: dict[str, torch.Tensor]) -> None:
    """Copy audio-adapter tensors into llama in place, with shape assertions.

    Shared by load_weights (init_from / sidecar overlay) and
    apply_full_checkpoint (slim-checkpoint resume).  Fails loudly on an unknown
    name or a shape mismatch — a silent skip here would leave a partly-trained
    adapter in fresh-init state.
    """
    own = dict(llama.named_parameters())
    for name, tensor in tensors.items():
        if name not in own:
            raise RuntimeError(
                f"audio_adapters key '{name}' not found in llama parameters"
            )
        if own[name].shape != tensor.shape:
            raise ValueError(
                f"audio_adapters '{name}' shape {tuple(tensor.shape)} != "
                f"model {tuple(own[name].shape)}"
            )
        with torch.no_grad():
            own[name].copy_(tensor)


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
    audio_adapters_from:  nn.Module | None = None,
    modules_dirty:        set[str] | list[str] | None = None,
    init_from:            str | None = None,
    step_in_stage:        int = 0,
    stage_index:          int = 0,
    kind:                 str = "periodic",
) -> None:
    """Save a full training checkpoint as a DELTA over the pretrained base.

    Invariant: a checkpoint carries every parameter that differs from the
    pretrained base (Whisper ckpt + Llama ckpt + deterministic fresh inits) and
    may omit anything identical to it, so that
    ``pretrained base + this checkpoint = complete model`` with no dependence on
    init_from at load time.  The caller passes only the "dirty" heavy modules:

        save_checkpoint(
            path, ...,
            encoder = encoder if "encoder" in modules_dirty else None,   # ~350 MB
            llama   = llama   if "llama"   in modules_dirty else None,   # ~32 GB
            audio_adapters_from = llama,   # harvest the small adapter delta
            modules_dirty       = modules_dirty,
            init_from           = str(cfg.model.init_from) if cfg.model.init_from else None,
        )

    audio adapters (~130 MB) are always carried: when full ``llama`` is saved
    its state_dict subsumes them (no duplicate key); otherwise they are harvested
    from ``audio_adapters_from`` and stored under "audio_adapters".

    ``modules_dirty`` / ``init_from`` are recorded for resume: the resumed run
    restores the dirty set (so it keeps saving the same modules) and warns if
    init_from provenance changed.  They are stored only when modules_dirty is
    provided, so legacy callers (and the self-tests) produce byte-identical
    key sets to before.
    """
    d: dict = {
        "step":                step,
        "epoch":               epoch,
        "micro_step_in_epoch": micro_step_in_epoch,
        "batch_size":          batch_size,
        "step_in_stage":       step_in_stage,
        "stage_index":         stage_index,
        "kind":                kind,
        "adapter":             adapter.state_dict(),
        "optimizer":           optimizer.state_dict(),
        "scaler":              scaler.state_dict(),
    }
    if scheduler is not None:
        d["scheduler"] = scheduler.state_dict()
    if encoder is not None:
        d["encoder"] = encoder.state_dict()
    if llama is not None:
        # Full llama state_dict already contains the audio-adapter tensors —
        # storing them separately too would duplicate ~130 MB for nothing.
        d["llama"] = llama.state_dict()
    elif audio_adapters_from is not None:
        audio_adapters = _collect_audio_adapters(audio_adapters_from)
        if audio_adapters:
            d["audio_adapters"] = audio_adapters
    if modules_dirty is not None:
        # Record provenance alongside the dirty set so a resume can (a) keep
        # saving the same modules and (b) detect an init_from swap.
        d["modules_dirty"] = sorted(modules_dirty)
        d["init_from"]     = init_from   # may be None (run had no warm-start)
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
    llama:               nn.Module | None = None,
    step_in_stage:       int = 0,
    stage_index:         int = 0,
    kind:                str = "handoff",
) -> None:
    """Save an adapter-only checkpoint for cross-stage loading.

    The optimizer state is stored under key "optimizer_adapter" (not
    "optimizer") so that load_adapter_checkpoint can distinguish it from a
    full checkpoint and avoids accidentally restoring stage-1 moments into a
    stage-2 optimizer.

    When ``llama`` is passed and it owns gated audio adapters, those tensors are
    also stored under a distinct "audio_adapters" key ({param_name: tensor}).
    The sidecar has no full "llama" state_dict, so this is how a trained
    audio-adapter stage hands its new weights forward; load_weights overlays
    them.  A legacy sidecar without the key loads fine (fresh init is kept).

    NOTE — batch_size inconsistency (flagged for owner):
      Regular adapter sidecars (periodic / early-stop / final) are called with
      batch_size=args.batch_size, while the auto-stage stage-1-final sidecar
      is called with batch_size=_batch_size.  These are the same value in
      single-stage runs but differ when --s1_batch_size overrides the default.
      The caller passes the appropriate value; this function does not unify them.
    """
    d: dict = {
        "step":                step,
        "epoch":               epoch,
        "micro_step_in_epoch": micro_step_in_epoch,
        "batch_size":          batch_size,
        "step_in_stage":       step_in_stage,
        "stage_index":         stage_index,
        "kind":                kind,
        "adapter":             adapter.state_dict(),
        "optimizer_adapter":   optimizer.state_dict(),
    }
    if llama is not None:
        audio_adapters = _collect_audio_adapters(llama)
        if audio_adapters:
            d["audio_adapters"] = audio_adapters
    torch.save(d, path)


# ── Load ──────────────────────────────────────────────────────────────────────

def load_weights(
    path: Path | str,
    *,
    encoder: nn.Module | None = None,
    adapter: nn.Module | None = None,
    llama:   nn.Module | None = None,
) -> list[str]:
    """Load model weights from a checkpoint file; ignore all non-weight keys.

    Only loads state_dicts for the three model modules (encoder, adapter, llama).
    Optimizer, scaler, scheduler, step, epoch, etc. are silently skipped.

    Thin wrapper over read_checkpoint + apply_weights.  A caller that also wants
    the metadata (step / stage_index / kind / …) should use those two directly
    so the file is read once — checkpoints run to tens of gigabytes.

    Args:
        path:    checkpoint file path
        encoder: WhisperEncoder module, or None to skip
        adapter: bridge adapter module, or None to skip
        llama:   Llama module, or None to skip

    Returns:
        list of module names that were actually loaded, in encoder→adapter→llama order
    """
    return apply_weights(
        read_checkpoint(path), encoder=encoder, adapter=adapter, llama=llama,
    )


def apply_weights(
    ckpt: dict,
    *,
    encoder: nn.Module | None = None,
    adapter: nn.Module | None = None,
    llama:   nn.Module | None = None,
) -> list[str]:
    """Overlay the weight keys of an already-loaded checkpoint dict onto modules.

    Identical to load_weights but takes the dict instead of a path, so a caller
    that needs checkpoint metadata (step, epoch, step_in_stage, stage_index,
    kind, modules_dirty) can read the file once and use both halves — see
    tools/run_wer.py, which tags every summary row with those fields.

    Args:
        ckpt:    checkpoint dict from read_checkpoint()
        encoder: WhisperEncoder module, or None to skip
        adapter: bridge adapter module, or None to skip
        llama:   Llama module, or None to skip

    Returns:
        list of module names that were actually loaded, in encoder→adapter→llama order
    """
    loaded: list[str] = []
    for key, module in [("encoder", encoder), ("adapter", adapter), ("llama", llama)]:
        if module is not None and key in ckpt:
            if key == "llama":
                # Tolerate legacy llama state_dicts missing audio_adapter keys.
                _load_llama_state_tolerant(module, ckpt[key])
            elif key == "adapter":
                # …and legacy bridges missing the audio marker vectors.
                _load_bridge_state_tolerant(module, ckpt[key])
            else:
                module.load_state_dict(ckpt[key])
            loaded.append(key)

    # Slim checkpoints / handoff sidecars have no full "llama" state_dict, so any
    # trained audio adapters ride along under a separate "audio_adapters" key.
    # Overlay them onto the live llama when present; a legacy file without the key
    # simply leaves the fresh init in place.
    if llama is not None and "audio_adapters" in ckpt:
        _overlay_audio_adapters(llama, ckpt["audio_adapters"])
        if "llama" not in loaded:
            loaded.append("audio_adapters")

    return loaded


def read_checkpoint(path: Path | str) -> dict:
    """Read a checkpoint file into a raw dict, without applying it to any module.

    Lets a caller inspect checkpoint metadata (e.g. stage_index) before
    deciding which modules/optimizer/scheduler to construct and pass to
    apply_full_checkpoint — avoiding a second torch.load of a large file.
    """
    return torch.load(path, map_location="cpu")


def apply_full_checkpoint(
    ckpt: dict,
    *,
    encoder:   nn.Module | None                              = None,
    adapter:   nn.Module,
    llama:     nn.Module | None                              = None,
    optimizer: torch.optim.Optimizer,
    scaler:    torch.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None  = None,
    current_init_from: str | None                           = None,
) -> ResumeState:
    """Apply an already-loaded full checkpoint dict for same-stage resume.

    Under the delta invariant (see save_checkpoint) the checkpoint carries only
    the modules that diverge from the pretrained base; anything absent was
    identical to pretrained and is left as the freshly-built model has it (the
    caller builds from pretrained first).  So:

      - encoder / llama full states load when present AND the module was passed;
        absent → skipped (the pretrained value already sits in the model).
      - "audio_adapters" (the small partial-llama delta present in slim saves)
        is overlaid onto llama when present.

    Dirty set
    ---------
    ResumeState.modules_dirty is restored from the "modules_dirty" key so the
    resumed run keeps saving the same modules.  Legacy checkpoints predate that
    key: there, the PRESENCE of a full "encoder" / "llama" state IS the dirty
    signal, so we infer the set from the keys the file carries.

    init_from provenance
    --------------------
    If the checkpoint records "init_from" and it differs from
    ``current_init_from`` (string compare, None-consistent), a prominent WARNING
    is printed (never raised): a different init_from can overlay a module the
    original run left pristine (hence absent here), silently diverging the resume.
    Path differences across machines are legitimate, so this is a warning.

    Scheduler notes
    ---------------
    If the checkpoint contains "scheduler", its state is restored directly.
    If it is absent (checkpoint predates scheduler saving), the fallback
    fast-forward loop below runs — but see the BUG? comment.

    Returns a ResumeState with batch_size=None when the key is absent from
    the checkpoint (old format).  The caller should apply its own default:
        ckpt = read_checkpoint(path)
        rs = apply_full_checkpoint(ckpt, ...)
        batch_size = rs.batch_size if rs.batch_size is not None else args.batch_size
    """
    if encoder is not None and "encoder" in ckpt:
        encoder.load_state_dict(ckpt["encoder"])
    _load_bridge_state_tolerant(adapter, ckpt["adapter"])
    if llama is not None and "llama" in ckpt:
        # Tolerate resuming a checkpoint written before audio adapters existed.
        _load_llama_state_tolerant(llama, ckpt["llama"])
    if llama is not None and "audio_adapters" in ckpt:
        # Slim save: full llama was omitted; overlay the carried adapter delta.
        _overlay_audio_adapters(llama, ckpt["audio_adapters"])

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

    # init_from provenance check (warn, never raise).
    if "init_from" in ckpt and ckpt["init_from"] != current_init_from:
        print(
            "WARNING: resume init_from mismatch — checkpoint was written with "
            f"init_from={ckpt['init_from']!r} but this run has "
            f"init_from={current_init_from!r}. A different warm-start can overlay "
            "a module the original run left pristine (hence absent from this "
            "checkpoint), silently diverging the resumed model. Continuing anyway "
            "(path differences across machines are legitimate)."
        )

    # Restore the dirty set; legacy checkpoints infer it from carried module keys.
    if "modules_dirty" in ckpt:
        modules_dirty = tuple(ckpt["modules_dirty"])
    else:
        modules_dirty = tuple(m for m in ("encoder", "llama") if m in ckpt)

    return ResumeState(
        step                = ckpt["step"],
        epoch               = ckpt.get("epoch", 0),
        micro_step_in_epoch = ckpt.get("micro_step_in_epoch", 0),
        batch_size          = ckpt.get("batch_size"),   # None when absent
        step_in_stage       = ckpt.get("step_in_stage", ckpt["step"]),  # old ckpts: use global step
        stage_index         = ckpt.get("stage_index", 0),
        modules_dirty       = modules_dirty,
    )


def load_full_checkpoint(
    path: Path | str,
    *,
    encoder:   nn.Module | None                              = None,
    adapter:   nn.Module,
    llama:     nn.Module | None                              = None,
    optimizer: torch.optim.Optimizer,
    scaler:    torch.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None  = None,
    current_init_from: str | None                           = None,
) -> ResumeState:
    """Load a full checkpoint for same-stage resume.

    Thin wrapper: read_checkpoint(path) then apply_full_checkpoint(...).
    See apply_full_checkpoint for behavior notes.
    """
    return apply_full_checkpoint(
        read_checkpoint(path),
        encoder=encoder, adapter=adapter, llama=llama,
        optimizer=optimizer, scaler=scaler, scheduler=scheduler,
        current_init_from=current_init_from,
    )



# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import contextlib
    import io
    import sys
    import tempfile

    # Allow `python utils/checkpoint.py` to import the model package (repo root).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
            "step_in_stage", "stage_index", "kind",
            "adapter", "optimizer", "scaler", "encoder", "llama",
        }, f"Unexpected full-ckpt keys: {set(raw.keys())}"
        assert raw["kind"] == "periodic", f"default kind should be 'periodic', got {raw['kind']!r}"
        assert raw["step_in_stage"] == 0, "step_in_stage default not stored"
        assert raw["stage_index"]   == 0, "stage_index default not stored"

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
        # Legacy full checkpoint (no "modules_dirty" key): the dirty set is
        # inferred from the module states the file carries — here encoder+llama.
        assert rs == ResumeState(step=42, epoch=3, micro_step_in_epoch=17, batch_size=8,
                                 step_in_stage=0, stage_index=0,
                                 modules_dirty=("encoder", "llama")), f"ResumeState mismatch: {rs}"
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
        assert rs2 == ResumeState(step=7, epoch=0, micro_step_in_epoch=5, batch_size=4,
                                  step_in_stage=0, stage_index=0), f"ResumeState mismatch: {rs2}"
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

        # ── 4. Old-format checkpoint: step_in_stage / stage_index absent → defaults ──
        # Simulates a checkpoint written before these fields were added (pre-Prompt-6).
        old_path = td / "old_fmt.pt"
        d_old = torch.load(slim_path, map_location="cpu")
        del d_old["step_in_stage"]
        del d_old["stage_index"]
        torch.save(d_old, old_path)

        ada_old = nn.Linear(4, 4)
        opt_old = torch.optim.SGD(ada_old.parameters(), lr=1e-4)
        scl_old = torch.amp.GradScaler("cpu")
        rs_old = load_full_checkpoint(old_path, adapter=ada_old, optimizer=opt_old, scaler=scl_old)
        # step_in_stage falls back to the global step value; stage_index falls back to 0
        assert rs_old.step_in_stage == rs_old.step, (
            f"expected step_in_stage={rs_old.step}, got {rs_old.step_in_stage}"
        )
        assert rs_old.stage_index == 0, f"expected stage_index=0, got {rs_old.stage_index}"
        print("  [OK] old-format checkpoint: step_in_stage→step, stage_index→0")

        # ── 5. Scheduler + scaler continuity round-trip ───────────────────────
        ada6 = nn.Linear(4, 4)
        opt6 = torch.optim.SGD(ada6.parameters(), lr=1e-3)
        scl6 = torch.amp.GradScaler("cpu")
        # Set a non-default scale so the round-trip assertion is meaningful
        scl6.load_state_dict({
            "scale": 256.0, "growth_factor": 2.0, "backoff_factor": 0.5,
            "growth_interval": 2000, "_growth_tracker": 0,
        })
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
        scl7 = torch.amp.GradScaler("cpu")   # default scale (65536), not 256
        sched7 = torch.optim.lr_scheduler.LambdaLR(opt7, lr_lambda=lambda s: min(s / 10, 1.0))
        rs5 = load_full_checkpoint(
            sched_path, adapter=ada7, optimizer=opt7, scaler=scl7, scheduler=sched7,
        )
        assert rs5.step == 5
        # scheduler last_epoch must be restored (5 steps)
        assert sched7.last_epoch == 5, f"scheduler.last_epoch={sched7.last_epoch}"
        # LR value must match — catches a resume landing at the wrong warmup position
        assert sched7.get_last_lr() == sched6.get_last_lr(), (
            f"scheduler LR mismatch after resume: {sched7.get_last_lr()} vs {sched6.get_last_lr()}"
        )
        # Scaler scale must survive the round-trip (not revert to the fresh-scaler default)
        assert scl7.state_dict()["scale"] == scl6.state_dict()["scale"], (
            f"scaler scale mismatch after resume: "
            f"{scl7.state_dict()['scale']} vs {scl6.state_dict()['scale']}"
        )
        print("  [OK] scheduler + scaler continuity round-trip")

        # ── 6. load_weights: weights-only overlay ────────────────────────────
        # Save a checkpoint that has encoder + adapter but NOT llama.
        ada_lw = nn.Linear(4, 4)
        enc_lw = nn.Linear(4, 4)
        lw_path = td / "load_weights.pt"
        torch.save(
            {
                "step": 99, "epoch": 1, "micro_step_in_epoch": 7, "batch_size": 4,
                "step_in_stage": 10, "stage_index": 0,
                "adapter":  ada_lw.state_dict(),
                "encoder":  enc_lw.state_dict(),
                "optimizer": {},  # non-weight keys should be ignored
                "scaler":    {},
            },
            lw_path,
        )

        # Target modules with fresh (different) random weights.
        ada_t  = nn.Linear(4, 4)
        enc_t  = nn.Linear(4, 4)
        llm_t  = nn.Linear(4, 4)
        llm_orig = llm_t.weight.data.clone()

        loaded_lw = load_weights(lw_path, encoder=enc_t, adapter=ada_t, llama=llm_t)
        assert loaded_lw == ["encoder", "adapter"], f"Expected ['encoder','adapter'], got {loaded_lw}"
        assert torch.allclose(enc_t.weight.data, enc_lw.weight.data),  "encoder weight mismatch"
        assert torch.allclose(ada_t.weight.data, ada_lw.weight.data),  "adapter weight mismatch"
        assert torch.allclose(llm_t.weight.data, llm_orig), "llama should be untouched"
        print("  [OK] load_weights: encoder+adapter loaded, llama untouched")

        # Only the adapter module provided — encoder/llama keys in ckpt are ignored.
        ada_t2 = nn.Linear(4, 4)
        loaded_lw2 = load_weights(lw_path, adapter=ada_t2)
        assert loaded_lw2 == ["adapter"], f"Expected ['adapter'], got {loaded_lw2}"
        assert torch.allclose(ada_t2.weight.data, ada_lw.weight.data), "adapter weight mismatch"
        print("  [OK] load_weights: only provided modules considered")

        # ── 6b. apply_weights: same overlay from an already-read dict ─────────
        # The one-read path used by tools/run_wer.py, which needs the metadata
        # (step / stage_index / kind) alongside the weights.
        raw_lw   = read_checkpoint(lw_path)
        ada_t3   = nn.Linear(4, 4)
        enc_t3   = nn.Linear(4, 4)
        loaded_lw3 = apply_weights(raw_lw, encoder=enc_t3, adapter=ada_t3)
        assert loaded_lw3 == loaded_lw, f"apply_weights disagrees with load_weights: {loaded_lw3}"
        assert torch.allclose(enc_t3.weight.data, enc_lw.weight.data), "encoder weight mismatch"
        assert torch.allclose(ada_t3.weight.data, ada_lw.weight.data), "adapter weight mismatch"
        assert raw_lw["step"] == 99 and raw_lw["stage_index"] == 0, "metadata from the same read"
        print("  [OK] apply_weights: dict overlay matches load_weights, metadata intact")

        # ── 7. audio-adapter back-compat: legacy llama ckpt → adapter-enabled model ──
        # Save a full checkpoint carrying a llama state_dict written BEFORE audio
        # adapters existed, then warm-start it into an adapter-enabled model via
        # load_weights (the init_from path).  It must succeed, print the tolerance
        # message, load every pretrained tensor, and leave the fresh audio-adapter
        # init untouched.
        from model.llama import Llama, LlamaConfig

        _dims = dict(n_layers=3, d_model=32, n_heads=4, n_kv_heads=2,
                     intermediate_size=64, vocab_size=50)
        legacy_llama = Llama(LlamaConfig(**_dims, audio_adapter_r=0))   # no adapters
        legacy_path  = td / "legacy_llama.pt"
        torch.save({"llama": legacy_llama.state_dict()}, legacy_path)

        new_llama = Llama(LlamaConfig(**_dims, audio_adapter_r=8))      # adapter-enabled
        fresh_aa  = {n: p.detach().clone()
                     for n, p in new_llama.named_parameters() if "audio_adapter" in n}
        assert fresh_aa, "adapter-enabled model must own audio_adapter params"

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            loaded7 = load_weights(legacy_path, llama=new_llama)
        msg = buf.getvalue()
        assert loaded7 == ["llama"], f"expected ['llama'], got {loaded7}"
        assert "audio-adapter tensors not found" in msg, f"missing tolerance message: {msg!r}"

        legacy_named = dict(legacy_llama.named_parameters())
        for n, p in new_llama.named_parameters():
            if "audio_adapter" in n:
                assert torch.equal(p, fresh_aa[n]), f"{n} must retain fresh init"
            else:
                assert torch.allclose(p, legacy_named[n]), f"{n} must load from legacy ckpt"
        print("  [OK] back-compat: legacy llama loads, audio adapters keep fresh init")

        # ── 8. sidecar round-trip: save_adapter_checkpoint carries audio adapters ──
        # A handoff sidecar has no full "llama" state_dict, so trained audio
        # adapters ride under the "audio_adapters" key and load_weights overlays
        # them onto a fresh adapter-enabled model.
        src_llama = Llama(LlamaConfig(**_dims, audio_adapter_r=8))
        with torch.no_grad():
            for n, p in src_llama.named_parameters():
                if "audio_adapter" in n:
                    p.add_(1.0)   # make them distinguishable from any fresh init
        opt_sc = torch.optim.SGD(src_llama.audio_adapter_parameters(), lr=1e-3)
        sidecar_path = td / "handoff-adapter.pt"
        save_adapter_checkpoint(
            sidecar_path,
            step=1, epoch=0, micro_step_in_epoch=0, batch_size=4,
            adapter=nn.Linear(4, 4), optimizer=opt_sc, llama=src_llama,
        )
        raw_sc = torch.load(sidecar_path, map_location="cpu")
        assert "audio_adapters" in raw_sc, "sidecar should carry audio_adapters key"

        dst_llama = Llama(LlamaConfig(**_dims, audio_adapter_r=8))
        loaded8 = load_weights(sidecar_path, llama=dst_llama)
        assert "audio_adapters" in loaded8, f"expected audio_adapters loaded, got {loaded8}"
        src_named = dict(src_llama.named_parameters())
        for n, p in dst_llama.named_parameters():
            if "audio_adapter" in n:
                assert torch.equal(p, src_named[n]), f"{n} must match sidecar tensor"
        print("  [OK] sidecar round-trip: audio adapters saved and overlaid")

        # Small helpers for the delta-checkpoint tests below.
        def _sgd(mod):
            return torch.optim.SGD(mod.parameters(), lr=1e-4)

        # ── 9. slim save: encoder/llama omitted, audio_adapters + provenance kept ──
        slim_llama = Llama(LlamaConfig(**_dims, audio_adapter_r=8))
        slim_ada   = nn.Linear(4, 4)
        save_checkpoint(
            td / "slim_delta.pt",
            step=5, epoch=0, micro_step_in_epoch=0, batch_size=4,
            adapter=slim_ada, optimizer=_sgd(slim_ada), scaler=torch.amp.GradScaler("cpu"),
            encoder=None, llama=None,             # nothing heavy dirty
            audio_adapters_from=slim_llama,
            modules_dirty=set(), init_from=None,
        )
        raw9 = torch.load(td / "slim_delta.pt", map_location="cpu")
        assert "encoder" not in raw9 and "llama" not in raw9, sorted(raw9)
        assert "adapter" in raw9 and "audio_adapters" in raw9, sorted(raw9)
        assert raw9["modules_dirty"] == [], raw9["modules_dirty"]   # stored (empty), not absent
        assert "init_from" in raw9 and raw9["init_from"] is None
        print("  [OK] slim save: adapter + audio_adapters only; modules_dirty stored")

        # ── 10. full-llama save subsumes audio_adapters (no duplicate key) ────────
        full_llama = Llama(LlamaConfig(**_dims, audio_adapter_r=8))
        full_ada   = nn.Linear(4, 4)
        save_checkpoint(
            td / "full_delta.pt",
            step=5, epoch=0, micro_step_in_epoch=0, batch_size=4,
            adapter=full_ada, optimizer=_sgd(full_ada), scaler=torch.amp.GradScaler("cpu"),
            encoder=None, llama=full_llama, audio_adapters_from=full_llama,
            modules_dirty={"llama"}, init_from=None,
        )
        raw10 = torch.load(td / "full_delta.pt", map_location="cpu")
        assert "llama" in raw10, "full llama should be saved"
        assert "audio_adapters" not in raw10, "audio_adapters must be subsumed by full llama"
        assert raw10["modules_dirty"] == ["llama"], raw10["modules_dirty"]
        print("  [OK] full-llama save subsumes audio_adapters (no duplicate key)")

        # ── 11a. apply overlays audio_adapters + restores modules_dirty ──────────
        ov_src = Llama(LlamaConfig(**_dims, audio_adapter_r=8))
        with torch.no_grad():
            for n, p in ov_src.named_parameters():
                if "audio_adapter" in n:
                    p.add_(2.0)
        ov_ada = nn.Linear(4, 4)
        save_checkpoint(
            td / "overlay_delta.pt",
            step=3, epoch=0, micro_step_in_epoch=0, batch_size=4,
            adapter=ov_ada, optimizer=_sgd(ov_ada), scaler=torch.amp.GradScaler("cpu"),
            encoder=None, llama=None, audio_adapters_from=ov_src,
            modules_dirty=set(), init_from=None,   # audio-adapter-only run: llama not dirty
        )
        ov_dst     = Llama(LlamaConfig(**_dims, audio_adapter_r=8))
        ov_dst_ada = nn.Linear(4, 4)
        rs11 = apply_full_checkpoint(
            read_checkpoint(td / "overlay_delta.pt"),
            adapter=ov_dst_ada, llama=ov_dst,
            optimizer=_sgd(ov_dst_ada), scaler=torch.amp.GradScaler("cpu"),
        )
        ov_src_named = dict(ov_src.named_parameters())
        for n, p in ov_dst.named_parameters():
            if "audio_adapter" in n:
                assert torch.equal(p, ov_src_named[n]), f"{n} overlay mismatch"
        assert rs11.modules_dirty == (), rs11.modules_dirty
        print("  [OK] apply overlays audio_adapters; modules_dirty restored")

        # ── 11b. legacy fallback: no modules_dirty key + llama present → ('llama',) ──
        leg_llama = Llama(LlamaConfig(**_dims, audio_adapter_r=0))
        leg_ada   = nn.Linear(4, 4)
        save_checkpoint(
            td / "legacy_full.pt",
            step=2, epoch=0, micro_step_in_epoch=0, batch_size=4,
            adapter=leg_ada, optimizer=_sgd(leg_ada), scaler=torch.amp.GradScaler("cpu"),
            encoder=None, llama=leg_llama,   # NO modules_dirty kwarg → legacy-style save
        )
        raw_leg = torch.load(td / "legacy_full.pt", map_location="cpu")
        assert "modules_dirty" not in raw_leg and "llama" in raw_leg and "encoder" not in raw_leg
        leg_dst_ada = nn.Linear(4, 4)
        rs_leg = apply_full_checkpoint(
            read_checkpoint(td / "legacy_full.pt"),
            adapter=leg_dst_ada, llama=Llama(LlamaConfig(**_dims, audio_adapter_r=0)),
            optimizer=_sgd(leg_dst_ada), scaler=torch.amp.GradScaler("cpu"),
        )
        assert rs_leg.modules_dirty == ("llama",), rs_leg.modules_dirty
        print("  [OK] legacy fallback infers modules_dirty from carried keys")

        # ── 12. init_from mismatch warns (no raise); match is silent ─────────────
        mm_ada = nn.Linear(4, 4)
        save_checkpoint(
            td / "mm.pt",
            step=1, epoch=0, micro_step_in_epoch=0, batch_size=4,
            adapter=mm_ada, optimizer=_sgd(mm_ada), scaler=torch.amp.GradScaler("cpu"),
            encoder=None, llama=None, modules_dirty=set(), init_from="a.pt",
        )
        mm_dst = nn.Linear(4, 4)
        buf_mm = io.StringIO()
        with contextlib.redirect_stdout(buf_mm):
            apply_full_checkpoint(
                read_checkpoint(td / "mm.pt"),
                adapter=mm_dst, optimizer=_sgd(mm_dst), scaler=torch.amp.GradScaler("cpu"),
                current_init_from="b.pt",
            )
        assert "init_from mismatch" in buf_mm.getvalue(), buf_mm.getvalue()
        mm_dst2 = nn.Linear(4, 4)
        buf_ok = io.StringIO()
        with contextlib.redirect_stdout(buf_ok):
            apply_full_checkpoint(
                read_checkpoint(td / "mm.pt"),
                adapter=mm_dst2, optimizer=_sgd(mm_dst2), scaler=torch.amp.GradScaler("cpu"),
                current_init_from="a.pt",
            )
        assert "init_from mismatch" not in buf_ok.getvalue(), buf_ok.getvalue()
        print("  [OK] init_from mismatch warns; matching provenance is silent")

        # ── 13. delta-completeness round-trip: pretrained base + slim delta = model ──
        torch.manual_seed(123)
        base_enc         = nn.Linear(4, 4)
        base_ada         = nn.Linear(4, 4)
        base_llama       = Llama(LlamaConfig(**_dims, audio_adapter_r=8))
        base_enc_state   = {k: v.clone() for k, v in base_enc.state_dict().items()}
        base_llama_state = {k: v.clone() for k, v in base_llama.state_dict().items()}

        # Training copy of the base; perturb ONLY adapter + audio adapters.
        tr_ada   = nn.Linear(4, 4); tr_ada.load_state_dict(base_ada.state_dict())
        tr_llama = Llama(LlamaConfig(**_dims, audio_adapter_r=8))
        tr_llama.load_state_dict(base_llama.state_dict())
        with torch.no_grad():
            for p in tr_ada.parameters():
                p.add_(0.5)
            for n, p in tr_llama.named_parameters():
                if "audio_adapter" in n:
                    p.add_(0.7)
        pert_ada_state = {k: v.clone() for k, v in tr_ada.state_dict().items()}
        pert_aa        = {n: p.detach().clone()
                          for n, p in tr_llama.named_parameters() if "audio_adapter" in n}

        save_checkpoint(
            td / "delta_complete.pt",
            step=10, epoch=0, micro_step_in_epoch=0, batch_size=4,
            adapter=tr_ada, optimizer=_sgd(tr_ada), scaler=torch.amp.GradScaler("cpu"),
            encoder=None, llama=None, audio_adapters_from=tr_llama,
            modules_dirty=set(), init_from=None,
        )
        # Rebuild FRESH models from the same pretrained base, then overlay the delta.
        fr_enc   = nn.Linear(4, 4); fr_enc.load_state_dict(base_enc_state)
        fr_ada   = nn.Linear(4, 4); fr_ada.load_state_dict(base_ada.state_dict())
        fr_llama = Llama(LlamaConfig(**_dims, audio_adapter_r=8))
        fr_llama.load_state_dict(base_llama_state)
        apply_full_checkpoint(
            read_checkpoint(td / "delta_complete.pt"),
            encoder=fr_enc, adapter=fr_ada, llama=fr_llama,
            optimizer=_sgd(fr_ada), scaler=torch.amp.GradScaler("cpu"),
        )
        for k, v in pert_ada_state.items():
            assert torch.equal(fr_ada.state_dict()[k], v), f"adapter {k} not restored to perturbed"
        fr_named = dict(fr_llama.named_parameters())
        for n, v in pert_aa.items():
            assert torch.equal(fr_named[n], v), f"audio adapter {n} not restored to perturbed"
        for n, p in fr_llama.named_parameters():
            if "audio_adapter" not in n:
                assert torch.equal(p, base_llama_state[n]), f"llama {n} must equal pristine base"
        for k, v in base_enc_state.items():
            assert torch.equal(fr_enc.state_dict()[k], v), f"encoder {k} must equal pristine base"
        print("  [OK] delta-completeness: pretrained base + slim delta = full model")

        # ── 14. dirty propagation across stages (mirrors training.py union) ──────
        dirty: set[str] = set()
        dirty |= {"llama"}   & {"encoder", "llama"}   # stage 0: trainable=[llama]
        dirty |= {"adapter"} & {"encoder", "llama"}   # stage 1: trainable=[adapter]
        assert dirty == {"llama"}, "llama dirtiness must persist into stage 1"
        dp_llama = Llama(LlamaConfig(**_dims, audio_adapter_r=0))
        dp_ada   = nn.Linear(4, 4)
        save_checkpoint(
            td / "dirty_prop.pt",
            step=20, epoch=0, micro_step_in_epoch=0, batch_size=4,
            adapter=dp_ada, optimizer=_sgd(dp_ada), scaler=torch.amp.GradScaler("cpu"),
            encoder=None,                                   # never dirtied
            llama=dp_llama if "llama" in dirty else None,   # dirty → full llama saved
            audio_adapters_from=dp_llama,
            modules_dirty=dirty, init_from=None,
        )
        raw_dp = torch.load(td / "dirty_prop.pt", map_location="cpu")
        assert "llama" in raw_dp, "stage-1 save must include full llama (dirty persists)"
        assert "encoder" not in raw_dp, "encoder was never dirtied"
        assert set(raw_dp["modules_dirty"]) == {"llama"}, raw_dp["modules_dirty"]
        print("  [OK] dirty propagation: llama stays saved after it stops being trainable")

    print("\nPASSED")
    sys.exit(0)
