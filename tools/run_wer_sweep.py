"""Multi-EXPERIMENT WER sweep: one checkpoint per experiment, one shared eval set.

Where tools/run_wer.py sweeps several checkpoints of ONE run (same architecture,
same config), this tool sweeps several RUNS whose architectures differ — the
E0…E6 grid of (trainable set × init) experiments.  It exists because those runs
cannot share a single model object: an MLP bridge and a SwiGLU bridge have
different tensor names, and an audio-adapter run has parameters a bridge-only run
does not.

Usage:
    python tools/run_wer_sweep.py \\
        --config      configs/both_formats_scratch.yaml \\
        --experiments checkpoints_aux \\
        --manifest    configs/wer_sweep_experiments.yaml \\
        --full

    # See the plan (per-experiment architecture + model rebuild order) without
    # touching the GPU:
    python tools/run_wer_sweep.py --config ... --experiments ... --dry-run

Layout it expects — one directory per experiment, one .pt inside each:

    checkpoints_aux/
    ├── unfmt-bridge-base/step_0004680.pt
    ├── bothfmt-bridge/step_0004680.pt
    └── …

Artifacts, written per experiment INTO its own directory:

    <exp_dir>/wer.csv                       long-format rows for this experiment
    <exp_dir>/{step:07d}_{split}.jsonl      sampled reference/hypothesis pairs

…plus two combined files at --output (default <experiments>/wer_all.csv):

    wer_all.csv       long format: one row per (experiment, split, format)
    wer_summary.csv   wide format: one row per experiment, one column per
                      split/format — the comparison table

WHAT THE CONFIG PROVIDES, AND WHAT IT DOES NOT
----------------------------------------------
--config supplies only the things that must be IDENTICAL across experiments for
the comparison to mean anything: tokenizer, eval shards, eval batch size, and
the WER settings.  The per-experiment MODEL ARCHITECTURE is NOT taken from a
config — it is probed from the checkpoint itself (bridge type, audio-adapter
type and rank).  That removes the one step where a wrong guess would silently
produce a valid-looking but meaningless number, and it means a config file per
experiment is not needed.

Pick a --config whose model.init_from is null (e.g. both_formats_scratch.yaml):
config validation rejects an init_from path that does not exist on this machine,
and the warm start would be discarded here regardless (each checkpoint is a
complete delta over the pretrained base — see utils/checkpoint.py).

CROSS-EXPERIMENT WEIGHT HYGIENE
-------------------------------
Checkpoints are deltas: a bridge-only run carries no encoder weights, so loading
it after an encoder+bridge run would leave the PREVIOUS experiment's encoder in
place and score the wrong model.  This tool snapshots the pristine
pretrained encoder / bridge / audio-adapter tensors right after building, and
restores all three before every checkpoint overlay.  A checkpoint carrying a
full "llama" state_dict (a full-finetune run) is handled by rebuilding the model
from pretrained before the next experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build import build_models
from data import INSTRUCTION_VARIANTS, PrunedTokenizer, build_sorted_eval_dataloader
# These two are private to utils.checkpoint, but reimplementing them here would
# risk diverging from how training.py loads the very same tensors.  They are
# imported rather than load_weights() because this tool must convert a legacy
# checkpoint dict (see fold_legacy_gate) BEFORE it is applied, and load_weights
# only takes a path.
from utils.checkpoint import _load_llama_state_tolerant, _overlay_audio_adapters
from utils.config import Config, load_config
from utils.evaluate import evaluate_all_splits


# ── Experiment / architecture records ─────────────────────────────────────────

@dataclass(frozen=True)
class Arch:
    """The model-architecture fields that must match a checkpoint to load it.

    ``audio_adapter_zero_writer_layers`` is deliberately absent: it only affects
    INITIALISATION, and every adapter tensor is overlaid from the checkpoint
    here, so it cannot influence an evaluation.
    """
    bridge_type:        str   # "mlp" | "swiglu"
    audio_adapter_r:    int   # 0 = no per-layer audio adapters
    audio_adapter_type: str   # "mlp" | "swiglu"  (meaningless when r == 0)

    def describe(self) -> str:
        if self.audio_adapter_r == 0:
            return f"bridge={self.bridge_type}, no audio adapters"
        return (f"bridge={self.bridge_type}, "
                f"audio_adapters={self.audio_adapter_type} r={self.audio_adapter_r}")


@dataclass(frozen=True)
class Experiment:
    """One experiment directory holding exactly one checkpoint."""
    id:        str            # short grid ID, e.g. "E0"; falls back to the dir name
    name:      str            # directory name, e.g. "bothfmt-audad(E2)"
    directory: Path
    ckpt:      Path
    trainable: str = ""       # descriptive columns, carried through to the CSV
    init:      str = ""
    data:      str = ""
    formats:   tuple[str, ...] | None = None   # None → whatever --formats says


# ── Discovery ─────────────────────────────────────────────────────────────────

def _single_checkpoint(directory: Path) -> Path:
    """Return the one .pt file in *directory*, failing loudly on 0 or >1."""
    pts = sorted(directory.glob("*.pt"))
    if not pts:
        raise FileNotFoundError(f"No .pt checkpoint in experiment directory {directory}")
    if len(pts) > 1:
        raise ValueError(
            f"Experiment directory {directory} holds {len(pts)} checkpoints "
            f"({[p.name for p in pts]}). Keep one per directory, or name the "
            f"intended file with 'checkpoint:' in the manifest."
        )
    return pts[0]


def discover_experiments(root: Path, manifest_path: Path | None) -> list[Experiment]:
    """Build the experiment list from *root*, refined by an optional manifest.

    Without a manifest every immediate subdirectory of *root* containing a .pt
    becomes an experiment, ordered by name.  With one, the manifest's `experiments`
    list defines both the SET and the ORDER (so the summary table comes out in
    grid order, E0…E6) and adds the descriptive columns.

    Manifest entries name directories relative to *root*; an entry whose
    directory is missing is a hard error, so a typo cannot silently drop an
    experiment from the comparison.
    """
    if not root.is_dir():
        raise NotADirectoryError(f"--experiments root is not a directory: {root}")

    if manifest_path is None:
        experiments = []
        for directory in sorted(p for p in root.iterdir() if p.is_dir()):
            if not any(directory.glob("*.pt")):
                print(f"[warn] {directory.name}: no .pt file — skipping")
                continue
            experiments.append(
                Experiment(id=directory.name, name=directory.name,
                           directory=directory, ckpt=_single_checkpoint(directory))
            )
        if not experiments:
            raise FileNotFoundError(f"No experiment directories with a .pt under {root}")
        return experiments

    with manifest_path.open() as f:
        entries = (yaml.safe_load(f) or {}).get("experiments") or []
    if not entries:
        raise ValueError(f"Manifest {manifest_path} has no non-empty 'experiments:' list")

    experiments = []
    for entry in entries:
        directory = root / entry["dir"]
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Manifest entry {entry.get('id', entry['dir'])!r} names directory "
                f"{directory}, which does not exist"
            )
        ckpt = directory / entry["checkpoint"] if entry.get("checkpoint") \
            else _single_checkpoint(directory)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        formats = entry.get("formats")
        experiments.append(Experiment(
            id        = str(entry.get("id", entry["dir"])),
            name      = str(entry["dir"]),
            directory = directory,
            ckpt      = ckpt,
            trainable = str(entry.get("trainable", "")),
            init      = str(entry.get("init", "")),
            data      = str(entry.get("data", "")),
            formats   = tuple(formats) if formats else None,
        ))
    return experiments


# ── Architecture probing ──────────────────────────────────────────────────────

def _audio_adapter_tensors(ckpt: dict) -> dict[str, torch.Tensor]:
    """Return the checkpoint's audio-adapter tensors, from wherever they live.

    Slim saves carry them under "audio_adapters"; a full-llama save has them
    inside the "llama" state_dict instead (no duplicate key — see
    utils/checkpoint.save_checkpoint).
    """
    if "audio_adapters" in ckpt:
        return dict(ckpt["audio_adapters"])
    if "llama" in ckpt:
        return {k: v for k, v in ckpt["llama"].items() if "audio_adapter" in k}
    return {}


def probe_arch(ckpt: dict, source: Path) -> Arch:
    """Infer the model architecture a checkpoint was written by.

    Reads tensor NAMES and SHAPES rather than trusting a config file, so an
    experiment cannot be scored against a mismatched model.

    Args:
        ckpt:   a loaded checkpoint dict
        source: its path, for error messages

    Returns:
        The Arch that reproduces the checkpoint's model.
    """
    bridge_state = ckpt.get("adapter")
    if not bridge_state:
        raise ValueError(f"{source}: no 'adapter' (bridge) state_dict — cannot probe bridge type")

    if any(k.startswith("mlp.") for k in bridge_state):
        bridge_type = "mlp"
    elif "gate_proj.weight" in bridge_state:
        bridge_type = "swiglu"
    else:
        raise ValueError(
            f"{source}: unrecognised bridge state_dict keys {sorted(bridge_state)[:6]} — "
            f"expected 'mlp.*' (mlp bridge) or 'gate_proj.weight' (swiglu bridge)"
        )

    aa = _audio_adapter_tensors(ckpt)
    if not aa:
        return Arch(bridge_type=bridge_type, audio_adapter_r=0, audio_adapter_type="mlp")

    # Both variants are keyed …audio_adapter.{gate,up,down}_proj.weight; only the
    # SwiGLU variant has a gate_proj, and in both variants the (r, d_model)
    # tensor is the one that gives the rank.
    if any(k.endswith("audio_adapter.gate_proj.weight") for k in aa):
        adapter_type, rank_suffix = "swiglu", "audio_adapter.gate_proj.weight"
    else:
        adapter_type, rank_suffix = "mlp", "audio_adapter.down_proj.weight"

    rank_keys = [k for k in aa if k.endswith(rank_suffix)]
    if not rank_keys:
        raise ValueError(
            f"{source}: audio adapters present but no {rank_suffix!r} tensor to read the rank from"
        )
    r = int(aa[sorted(rank_keys)[0]].shape[0])

    return Arch(bridge_type=bridge_type, audio_adapter_r=r, audio_adapter_type=adapter_type)


# ── Legacy scalar-gate conversion ─────────────────────────────────────────────

def legacy_gate_keys(ckpt: dict) -> list[str]:
    """Names of per-adapter scalar-gate tensors, which current code no longer has.

    The audio adapters used to multiply their branch by a learned ``tanh(gate)``
    scalar.  That parameter was removed (the input projection already gates the
    branch per-channel), so a checkpoint written before the removal has an
    ``audio_adapter.gate`` key with nowhere to load.
    """
    return sorted(k for k in _audio_adapter_tensors(ckpt) if k.endswith("audio_adapter.gate"))


def fold_legacy_gate(ckpt: dict, adapter_type: str) -> int:
    """Fold each ``tanh(gate)`` scalar into its adapter's writer, in place.

    The old branch was ``mask * tanh(gate) * adapter(x)`` and the current one is
    ``mask * adapter(x)``; since the writer projection is the branch's last
    linear op, scaling its weight by ``tanh(gate)`` reproduces the old branch
    EXACTLY.  The gate keys are then dropped.

    Args:
        ckpt:         loaded checkpoint dict, mutated in place
        adapter_type: "mlp" (writer is up_proj) or "swiglu" (writer is down_proj)

    Returns:
        Number of adapters converted.
    """
    container = ckpt["audio_adapters"] if "audio_adapters" in ckpt else ckpt["llama"]
    writer    = "up_proj.weight" if adapter_type == "mlp" else "down_proj.weight"

    n = 0
    for gate_key in [k for k in container if k.endswith("audio_adapter.gate")]:
        prefix     = gate_key[: -len("gate")]          # "…audio_adapter."
        writer_key = prefix + writer
        if writer_key not in container:
            raise KeyError(f"scalar gate {gate_key!r} has no matching writer {writer_key!r}")
        scale = torch.tanh(container[gate_key].to(torch.float32)).reshape(())
        container[writer_key] = (container[writer_key].to(torch.float32) * scale).to(
            container[writer_key].dtype
        )
        del container[gate_key]
        n += 1
    return n


# ── Checkpoint application ────────────────────────────────────────────────────

def apply_checkpoint(ckpt: dict, *, encoder, adapter, llama) -> list[str]:
    """Overlay a checkpoint's weights onto live modules; return module names loaded.

    Same contract as utils.checkpoint.load_weights, but takes an already-loaded
    dict so the caller can convert a legacy checkpoint (see fold_legacy_gate)
    before it is applied.
    """
    loaded: list[str] = []
    if "encoder" in ckpt:
        encoder.load_state_dict(ckpt["encoder"])
        loaded.append("encoder")
    if "adapter" in ckpt:
        adapter.load_state_dict(ckpt["adapter"])
        loaded.append("adapter")
    if "llama" in ckpt:
        _load_llama_state_tolerant(llama, ckpt["llama"])
        loaded.append("llama")
    # Independent of the branch above, matching load_weights: a slim save carries
    # its audio adapters under a separate key because it has no full llama state.
    if "audio_adapters" in ckpt:
        _overlay_audio_adapters(llama, ckpt["audio_adapters"])
        if "llama" not in loaded:
            loaded.append("audio_adapters")
    if not loaded:
        raise ValueError("checkpoint carries no model weights (no encoder/adapter/llama keys)")
    return loaded


def snapshot_pristine(encoder, adapter, llama) -> dict:
    """CPU copies of every module a checkpoint delta may overwrite (~0.5 GB).

    Restored before each overlay so one experiment's weights can never survive
    into the next one's score.  The frozen Llama backbone is excluded — it is
    large and no checkpoint here rewrites it; the caller rebuilds the model
    instead when one does.
    """
    return {
        "encoder": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
        "adapter": {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()},
        "audio_adapters": {
            n: p.detach().cpu().clone()
            for n, p in llama.named_parameters() if "audio_adapter" in n
        },
    }


def restore_pristine(pristine: dict, *, encoder, adapter, llama) -> None:
    """Reset encoder / bridge / audio adapters to their pretrained-base values."""
    encoder.load_state_dict(pristine["encoder"])
    adapter.load_state_dict(pristine["adapter"])
    if pristine["audio_adapters"]:
        _overlay_audio_adapters(llama, pristine["audio_adapters"])


# ── Misc helpers ──────────────────────────────────────────────────────────────

def _size_mb(path: Path) -> str:
    """Human-readable file size, for diagnosing truncated checkpoint transfers."""
    try:
        return f"{path.stat().st_size / 1e6:.1f} MB"
    except OSError:
        return "size unknown"


def parse_step(stem: str, fallback: int = 0) -> int:
    """Extract the optimizer step from a filename stem ('step0002880', 'step_0004680')."""
    m = re.search(r"step[_-]?(\d+)", stem, re.IGNORECASE)
    return int(m.group(1)) if m else fallback


def cfg_for_arch(cfg: Config, arch: Arch) -> Config:
    """Return *cfg* with its model fields replaced by the probed architecture.

    init_from is cleared: each checkpoint is a complete delta over the pretrained
    base, so a warm start would only risk leaking weights between experiments.
    """
    return replace(cfg, model=replace(
        cfg.model,
        bridge_type                      = arch.bridge_type,
        audio_adapter_r                  = arch.audio_adapter_r,
        audio_adapter_type               = arch.audio_adapter_type,
        audio_adapter_zero_writer_layers = 0,      # init-only; every tensor is overlaid
        init_from                        = None,
        gradient_checkpointing           = False,  # eval only
    ))


# ── Self-test ─────────────────────────────────────────────────────────────────

def _self_test() -> None:
    """Exercise the pure logic on synthetic checkpoints — CPU only, no real data.

    Covers everything that runs BEFORE the 8B model is built, which is where a
    silent mistake would produce a plausible-looking but wrong WER: architecture
    probing, manifest discovery, the legacy scalar-gate conversion (checked for
    exact numerical equivalence against a live adapter), and pristine restore.
    """
    import tempfile

    import torch.nn as nn

    from model.adapter import build_bridge_adapter
    from model.llama import AudioSwiGLUAdapter, Llama, LlamaConfig

    print("run_wer_sweep.py self-test")
    torch.manual_seed(0)

    D_MODEL, RANK, EPS = 32, 8, 1e-5

    def _fake_ckpt(bridge_type: str, adapter_type: str | None, r: int = RANK) -> dict:
        """A checkpoint dict with the same key layout as a real slim save."""
        bridge = build_bridge_adapter(bridge_type, llama_dim=D_MODEL)
        ckpt: dict = {"step": 4680, "adapter": bridge.state_dict()}
        if adapter_type is not None:
            cfg = LlamaConfig(n_layers=3, d_model=D_MODEL, n_heads=4, n_kv_heads=2,
                              intermediate_size=64, vocab_size=50,
                              audio_adapter_r=r, audio_adapter_type=adapter_type)
            llama = Llama(cfg)
            ckpt["audio_adapters"] = {
                n: p.detach().clone()
                for n, p in llama.named_parameters() if "audio_adapter" in n
            }
        return ckpt

    # ── 1. Architecture probing ───────────────────────────────────────────────
    cases = [
        (("mlp",    None),        Arch("mlp",    0,    "mlp")),
        (("mlp",    "mlp"),       Arch("mlp",    RANK, "mlp")),
        (("mlp",    "swiglu"),    Arch("mlp",    RANK, "swiglu")),
        (("swiglu", "swiglu"),    Arch("swiglu", RANK, "swiglu")),
    ]
    for (bridge_type, adapter_type), want in cases:
        got = probe_arch(_fake_ckpt(bridge_type, adapter_type), Path("synthetic.pt"))
        assert got == want, f"probe({bridge_type}, {adapter_type}) → {got}, want {want}"
    print("  [OK] probe_arch reads bridge type, adapter type and rank from tensors")

    # Adapters carried inside a full llama state_dict (full-finetune save) probe
    # the same as the slim "audio_adapters" form.
    full_cfg   = LlamaConfig(n_layers=3, d_model=D_MODEL, n_heads=4, n_kv_heads=2,
                             intermediate_size=64, vocab_size=50,
                             audio_adapter_r=RANK, audio_adapter_type="swiglu")
    full_ckpt  = {"adapter": build_bridge_adapter("mlp", llama_dim=D_MODEL).state_dict(),
                  "llama":   Llama(full_cfg).state_dict()}
    assert probe_arch(full_ckpt, Path("full.pt")) == Arch("mlp", RANK, "swiglu")
    print("  [OK] probe_arch finds adapters inside a full llama state_dict")

    try:
        probe_arch({"adapter": {"bogus.weight": torch.zeros(2)}}, Path("bad.pt"))
    except ValueError:
        pass
    else:
        raise AssertionError("unrecognised bridge keys must raise")
    print("  [OK] unrecognised bridge state_dict rejected")

    # ── 2. Legacy scalar-gate fold is EXACT ───────────────────────────────────
    # Reconstruct the removed branch (mask * tanh(gate) * adapter(x)) and check
    # the folded weights reproduce it bit-for-bit through the current adapter.
    gate_value = torch.tensor(0.37)
    live = AudioSwiGLUAdapter(D_MODEL, RANK, n_layers=3, eps=EPS)
    with torch.no_grad():
        nn.init.normal_(live.gate_proj.weight, std=0.1)   # undo the zero init
    x    = torch.randn(2, 5, D_MODEL)
    mask = torch.ones(2, 5, 1)
    want = torch.tanh(gate_value) * live(x, mask)

    legacy = {f"layers.0.audio_adapter.{n}": p.detach().clone()
              for n, p in live.named_parameters()}
    legacy["layers.0.audio_adapter.gate"] = gate_value.clone()
    ckpt_legacy = {"adapter": build_bridge_adapter("mlp", llama_dim=D_MODEL).state_dict(),
                   "audio_adapters": legacy}

    assert legacy_gate_keys(ckpt_legacy) == ["layers.0.audio_adapter.gate"]
    assert fold_legacy_gate(ckpt_legacy, "swiglu") == 1
    assert not legacy_gate_keys(ckpt_legacy), "gate keys must be dropped after folding"

    folded = AudioSwiGLUAdapter(D_MODEL, RANK, n_layers=3, eps=EPS)
    folded.load_state_dict({
        k.removeprefix("layers.0.audio_adapter."): v
        for k, v in ckpt_legacy["audio_adapters"].items()
    })
    got = folded(x, mask)
    assert torch.allclose(got, want, atol=1e-6), (
        f"folded branch differs from gated branch by {(got - want).abs().max():.2e}"
    )
    print("  [OK] legacy tanh(gate) folds into the writer exactly")

    # The mlp variant folds into up_proj instead; only that tensor may change.
    mlp_legacy = {
        "layers.0.audio_adapter.down_proj.weight": torch.randn(RANK, D_MODEL),
        "layers.0.audio_adapter.up_proj.weight":   torch.randn(D_MODEL, RANK),
        "layers.0.audio_adapter.gate":             gate_value.clone(),
    }
    before = {k: v.clone() for k, v in mlp_legacy.items()}
    fold_legacy_gate({"audio_adapters": mlp_legacy}, "mlp")
    assert torch.allclose(mlp_legacy["layers.0.audio_adapter.up_proj.weight"],
                          before["layers.0.audio_adapter.up_proj.weight"] * torch.tanh(gate_value))
    assert torch.equal(mlp_legacy["layers.0.audio_adapter.down_proj.weight"],
                       before["layers.0.audio_adapter.down_proj.weight"])
    print("  [OK] mlp variant folds into up_proj, leaves down_proj untouched")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        # ── 3. Discovery: bare directory scan ─────────────────────────────────
        for name in ["b-exp", "a-exp"]:
            (root / name).mkdir()
            torch.save(_fake_ckpt("mlp", None), root / name / "step_0004680.pt")
        (root / "empty-exp").mkdir()                     # no .pt → warned and skipped

        found = discover_experiments(root, None)
        assert [e.id for e in found] == ["a-exp", "b-exp"], [e.id for e in found]
        assert all(e.ckpt.name == "step_0004680.pt" for e in found)
        print("  [OK] bare scan finds experiments, sorts by name, skips empty dirs")

        # ── 4. Discovery: manifest defines set, order and columns ─────────────
        manifest = root / "m.yaml"
        manifest.write_text(
            "experiments:\n"
            "  - {id: E9, dir: b-exp, trainable: bridge, init: scratch, data: both}\n"
            "  - {id: E8, dir: a-exp, formats: [unformatted]}\n"
        )
        found = discover_experiments(root, manifest)
        assert [e.id for e in found] == ["E9", "E8"], "manifest order must be preserved"
        assert found[0].trainable == "bridge" and found[0].init == "scratch"
        assert found[0].formats is None and found[1].formats == ("unformatted",)
        print("  [OK] manifest sets experiment order, IDs, columns and per-exp formats")

        manifest.write_text("experiments:\n  - {id: EX, dir: does-not-exist}\n")
        try:
            discover_experiments(root, manifest)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("a manifest entry naming a missing directory must raise")
        print("  [OK] missing manifest directory is a hard error, not a silent drop")

        # Two checkpoints in one directory is ambiguous — refuse to guess.
        torch.save({"step": 1}, root / "a-exp" / "step_0009999.pt")
        try:
            discover_experiments(root, None)
        except ValueError:
            pass
        else:
            raise AssertionError("two .pt files in one experiment directory must raise")
        print("  [OK] ambiguous experiment directory rejected")

    # ── 5. parse_step handles both filename conventions ───────────────────────
    assert parse_step("step0002880") == 2880
    assert parse_step("step_0004680") == 4680
    assert parse_step("no-digits", fallback=7) == 7
    print("  [OK] parse_step reads both 'stepNNNN' and 'step_NNNN'")

    # ── 6. Pristine snapshot/restore undoes a checkpoint overlay ──────────────
    # This is what stops experiment N-1's encoder from scoring experiment N.
    enc     = nn.Linear(4, 4)
    bridge  = build_bridge_adapter("mlp", llama_dim=D_MODEL)
    llama   = Llama(LlamaConfig(n_layers=3, d_model=D_MODEL, n_heads=4, n_kv_heads=2,
                                intermediate_size=64, vocab_size=50,
                                audio_adapter_r=RANK, audio_adapter_type="swiglu"))
    pristine = snapshot_pristine(enc, bridge, llama)
    with torch.no_grad():                                # simulate an overlay
        for p in enc.parameters():
            p.add_(1.0)
        for p in bridge.parameters():
            p.add_(1.0)
        for n, p in llama.named_parameters():
            if "audio_adapter" in n:
                p.add_(1.0)
    restore_pristine(pristine, encoder=enc, adapter=bridge, llama=llama)
    for k, v in enc.state_dict().items():
        assert torch.equal(v, pristine["encoder"][k]), f"encoder {k} not restored"
    for k, v in bridge.state_dict().items():
        assert torch.equal(v, pristine["adapter"][k]), f"bridge {k} not restored"
    for n, p in llama.named_parameters():
        if "audio_adapter" in n:
            assert torch.equal(p, pristine["audio_adapters"][n]), f"{n} not restored"
    print("  [OK] pristine restore reverts encoder, bridge and audio adapters")

    # ── 7. cfg_for_arch rewrites exactly the architecture fields ──────────────
    cfg = load_config(Path(__file__).resolve().parent.parent / "configs" / "example.yaml")
    out = cfg_for_arch(cfg, Arch("swiglu", 88, "swiglu"))
    assert out.model.bridge_type == "swiglu" and out.model.audio_adapter_r == 88
    assert out.model.audio_adapter_type == "swiglu"
    assert out.model.init_from is None and out.model.gradient_checkpointing is False
    assert out.data == cfg.data and out.metrics == cfg.metrics, "eval setup must be untouched"
    print("  [OK] cfg_for_arch overrides architecture only, leaves the eval setup")

    print("\nPASSED")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WER sweep across a directory of per-experiment checkpoints.",
    )
    parser.add_argument(
        "--self-test", action="store_true", dest="self_test",
        help="Run the CPU self-test (probing, manifest discovery, legacy-gate folding, "
             "pristine restore) with synthetic checkpoints and exit.",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Training config YAML supplying the SHARED eval setup (tokenizer, eval "
             "shards, eval_batch_size, wer.*). Model architecture is probed from each "
             "checkpoint, not read from here. Use one whose model.init_from is null.",
    )
    parser.add_argument(
        "--experiments", type=Path, default=None, metavar="DIR",
        help="Root directory holding one subdirectory per experiment, each with one .pt.",
    )
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="Optional YAML defining experiment IDs, order, and descriptive columns "
             "(see configs/wer_sweep_experiments.yaml). Without it, subdirectories are "
             "discovered and sorted by name.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Combined long-format CSV/JSON path (default: <experiments>/wer_all.csv). "
             "The wide comparison table is written alongside it as *_summary.csv.",
    )
    parser.add_argument(
        "--max-batches", type=int, default=None, dest="max_batches",
        help="Override cfg.metrics.wer.max_batches (batches per split per experiment).",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Evaluate entire splits with no batch cap (overrides --max-batches).",
    )
    parser.add_argument(
        "--sample-transcriptions", type=int, default=None, dest="sample_transcriptions",
        help="Override cfg.metrics.wer.sample_transcriptions (rows per .jsonl).",
    )
    parser.add_argument(
        "--formats", nargs="+", choices=["unformatted", "formatted"],
        default=None, metavar="FORMAT",
        help="Default instruction variant(s) to evaluate (default: both). A manifest "
             "entry's own 'formats:' overrides this per experiment.",
    )
    parser.add_argument(
        "--splits", nargs="+", default=None, metavar="SPLIT",
        choices=["dev-clean", "dev-other", "test-clean", "test-other"],
        help="Restrict evaluation to these splits (default: every split configured "
             "and present on disk).",
    )
    parser.add_argument(
        "--fold-legacy-gate", action=argparse.BooleanOptionalAction, default=True,
        help="Fold the removed per-adapter tanh(gate) scalar into the writer "
             "projection when a checkpoint predates its removal (exact, default on). "
             "--no-fold-legacy-gate makes such a checkpoint a hard error instead.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Probe every checkpoint, print the plan and rebuild order, then exit "
             "without building a model or touching the GPU.",
    )
    parser.add_argument(
        "--progress-interval", type=float, default=60.0, dest="progress_interval",
        metavar="SECONDS",
        help="Print a progress line every N seconds per split (default: 60; 0 disables).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device string (default: cuda if available, else cpu).",
    )
    args = parser.parse_args(argv)
    if not args.self_test:
        for required in ("config", "experiments"):
            if getattr(args, required) is None:
                parser.error(f"--{required} is required unless --self-test is passed")
    return args


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    """Probe every experiment checkpoint, evaluate them all, write the tables."""
    args = _parse_args(argv)
    if args.self_test:
        _self_test()
        return

    cfg          = load_config(args.config)
    experiments  = discover_experiments(args.experiments, args.manifest)
    output_path  = args.output or (args.experiments / "wer_all.csv")
    summary_path = output_path.with_name(output_path.stem + "_summary.csv")

    # ── Probe pass: read every checkpoint once, decide the model build order ──
    # Cheap relative to generation, and it turns "wrong architecture" from a
    # confusing mid-sweep crash into an upfront error.
    print(f"Probing {len(experiments)} experiment checkpoint(s) …")
    arch_of:   dict[str, Arch] = {}
    legacy_of: dict[str, int]  = {}
    unreadable: list[tuple[Experiment, str]] = []
    for exp in experiments:
        # Probe every checkpoint even after one fails, so a batch of truncated
        # transfers is reported in a single pass instead of one re-run each.
        try:
            ckpt = torch.load(exp.ckpt, map_location="cpu")
        except Exception as exc:                     # noqa: BLE001 — reported verbatim below
            unreadable.append((exp, f"{type(exc).__name__}: {exc}"))
            print(f"  {exp.id:<4} {exp.name:<26} UNREADABLE ({_size_mb(exp.ckpt)})")
            continue
        arch = probe_arch(ckpt, exp.ckpt)
        gates = legacy_gate_keys(ckpt)
        arch_of[exp.id]   = arch
        legacy_of[exp.id] = len(gates)
        note = f"  [{len(gates)} legacy scalar gates]" if gates else ""
        print(f"  {exp.id:<4} {exp.name:<26} step {parse_step(exp.ckpt.stem):>6}  "
              f"{arch.describe()}{note}")
        del ckpt

    if unreadable:
        detail = "\n".join(
            f"    {exp.ckpt}  ({_size_mb(exp.ckpt)})\n      {err}"
            for exp, err in unreadable
        )
        raise SystemExit(
            f"[error] {len(unreadable)} of {len(experiments)} checkpoints could not be "
            f"read:\n{detail}\n"
            "  A torch .pt file is a zip archive; 'failed finding central directory' "
            "means it is truncated — almost always an interrupted or still-running "
            "copy from GCS. Re-fetch those files (gsutil cp / gcloud storage cp) and "
            "compare sizes against the source before re-running."
        )

    stale_gates = [e for e in experiments if legacy_of[e.id]]
    if stale_gates and not args.fold_legacy_gate:
        raise SystemExit(
            "[error] These checkpoints carry the removed per-adapter scalar gate and "
            f"cannot load into current code: {[e.name for e in stale_gates]}. "
            "Drop --no-fold-legacy-gate to fold tanh(gate) into the writer projection "
            "(an exact conversion)."
        )

    # Evaluate architecture-alike experiments back to back so the 8B backbone is
    # rebuilt as few times as possible; the reported table keeps manifest order.
    order = sorted(range(len(experiments)), key=lambda i: (
        arch_of[experiments[i].id].bridge_type,
        arch_of[experiments[i].id].audio_adapter_type,
        arch_of[experiments[i].id].audio_adapter_r,
        i,
    ))
    n_builds = len({arch_of[experiments[i].id] for i in order})
    print(f"\n{n_builds} distinct architecture(s) → {n_builds} model build(s). "
          f"Evaluation order: {[experiments[i].id for i in order]}")

    if args.dry_run:
        print("\n--dry-run: stopping before any model build.")
        return

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # ── Shared eval loaders — built once, reused by every experiment ──────────
    with (cfg.data.tokenizer / "pruned_config.json").open() as f:
        sep_token_id = json.load(f)["sep_token_id"]
    tokenizer = PrunedTokenizer(cfg.data.tokenizer)

    eval_shard_map = [
        ("dev-clean",  cfg.data.eval.dev_clean),
        ("dev-other",  cfg.data.eval.dev_other),
        ("test-clean", cfg.data.eval.test_clean),
        ("test-other", cfg.data.eval.test_other),
    ]
    eval_loaders: dict[str, list[tuple]] = {}
    for split_name, shard_path in eval_shard_map:
        if shard_path is None or (args.splits and split_name not in args.splits):
            continue
        if not Path(shard_path).exists():
            print(f"[warn] eval shard for {split_name} not found: {shard_path} — skipping")
            continue
        eval_loaders[split_name] = build_sorted_eval_dataloader(
            shard_path,
            tokenizer_path       = cfg.data.tokenizer,
            instruction_variants = INSTRUCTION_VARIANTS,
            batch_size           = cfg.metrics.eval_batch_size,
        )
        print(f"Eval loader: {split_name} → {shard_path} "
              f"({len(eval_loaders[split_name])} batches)")
    if not eval_loaders:
        raise SystemExit("[error] No eval loaders available — check data.eval.* in the config.")

    max_batches = None if args.full else (args.max_batches or cfg.metrics.wer.max_batches)
    n_samples   = args.sample_transcriptions or cfg.metrics.wer.sample_transcriptions
    progress    = args.progress_interval if args.progress_interval > 0 else None

    # ── Sweep ─────────────────────────────────────────────────────────────────
    encoder = adapter = llama = None
    built_arch: Arch | None = None
    pristine:   dict | None = None
    rebuild_needed = False           # set when a checkpoint overwrote frozen Llama

    rows_by_exp: dict[str, list[dict]] = {}

    for n, idx in enumerate(order, start=1):
        exp  = experiments[idx]
        arch = arch_of[exp.id]
        step = parse_step(exp.ckpt.stem, fallback=idx)

        print(f"\n{'=' * 78}\n[{n}/{len(order)}] {exp.id}  {exp.name}  "
              f"(step {step})\n  {arch.describe()}\n{'=' * 78}")

        if built_arch != arch or rebuild_needed:
            if encoder is not None:
                del encoder, adapter, llama, pristine
                torch.cuda.empty_cache()
            reason = "architecture change" if built_arch != arch else \
                     "previous checkpoint rewrote the Llama backbone"
            print(f"Building model ({reason}) …")
            encoder, adapter, llama, _ = build_models(
                cfg_for_arch(cfg, arch), device, train=False, apply_init_from=False
            )
            pristine       = snapshot_pristine(encoder, adapter, llama)
            built_arch     = arch
            rebuild_needed = False
        else:
            restore_pristine(pristine, encoder=encoder, adapter=adapter, llama=llama)

        ckpt = torch.load(exp.ckpt, map_location="cpu")
        if legacy_of[exp.id]:
            n_folded = fold_legacy_gate(ckpt, arch.audio_adapter_type)
            print(f"[convert] folded tanh(gate) into the writer of {n_folded} audio "
                  f"adapter(s) — this checkpoint predates the scalar gate's removal")
        loaded = apply_checkpoint(ckpt, encoder=encoder, adapter=adapter, llama=llama)
        print(f"[load] modules overlaid: {loaded}")
        if "llama" in loaded:
            # A full backbone overlay outlives the pristine snapshot, which covers
            # only encoder/bridge/audio adapters — force a clean rebuild next.
            rebuild_needed = True
        del ckpt

        encoder.eval(); adapter.eval(); llama.eval()

        formats = list(exp.formats) if exp.formats else args.formats
        if formats:
            print(f"  formats: {formats}")

        exp_rows: list[dict] = []
        for split_name, loader in eval_loaders.items():
            split_wer, split_samples = evaluate_all_splits(
                encoder, adapter, llama,
                {split_name: loader},
                tokenizer, sep_token_id, device,
                max_batches       = max_batches,
                n_samples         = n_samples,
                sample_seed       = step,
                formats           = formats,
                progress_interval = progress,
            )

            sample_path = exp.directory / f"{step:07d}_{split_name}.jsonl"
            with sample_path.open("w") as f:
                for sr in split_samples:
                    f.write(json.dumps({
                        "experiment": exp.id, "name": exp.name,
                        "checkpoint": str(exp.ckpt), "step": step, **sr,
                    }) + "\n")
            print(f"  samples → {sample_path}")

            for key, wer_val in split_wer.items():
                split, fmt = key.rsplit("/", 1)
                exp_rows.append({
                    "experiment": exp.id,
                    "name":       exp.name,
                    "trainable":  exp.trainable,
                    "init":       exp.init,
                    "train_data": exp.data,
                    "checkpoint": str(exp.ckpt),
                    "step":       step,
                    "split":      split,
                    "format":     fmt,
                    "wer":        wer_val,
                })

        # Per-experiment CSV, written immediately so a crash later keeps the work.
        exp_csv = exp.directory / "wer.csv"
        with exp_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(exp_rows[0].keys()))
            writer.writeheader()
            writer.writerows(exp_rows)
        print(f"  summary → {exp_csv}")

        rows_by_exp[exp.id] = exp_rows

    # ── Combined outputs, in manifest order ──────────────────────────────────
    all_rows = [r for exp in experiments for r in rows_by_exp.get(exp.id, [])]
    if not all_rows:
        raise SystemExit("[error] No results produced.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".json":
        with output_path.open("w") as f:
            json.dump(all_rows, f, indent=2)
    else:
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    # Wide comparison table: one row per experiment, one column per split/format.
    metric_cols: list[str] = []
    for r in all_rows:
        col = f"{r['split']}/{r['format']}"
        if col not in metric_cols:
            metric_cols.append(col)
    meta_cols = ["experiment", "name", "trainable", "init", "train_data", "step"]

    summary_rows = []
    for exp in experiments:
        rows = rows_by_exp.get(exp.id)
        if not rows:
            continue
        row = {c: rows[0][c] for c in meta_cols}
        row.update({c: "" for c in metric_cols})
        row.update({f"{r['split']}/{r['format']}": round(r["wer"], 5) for r in rows})
        summary_rows.append(row)

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=meta_cols + metric_cols)
        writer.writeheader()
        writer.writerows(summary_rows)

    # ── Printed comparison table ─────────────────────────────────────────────
    id_w   = max(len(r["experiment"]) for r in summary_rows)
    name_w = min(28, max(len(r["name"]) for r in summary_rows))
    header = (f"{'ID':<{id_w}}  {'Experiment':<{name_w}}  {'Step':>6}  "
              + "  ".join(f"{c:>22}" for c in metric_cols))
    print(f"\n── WER Comparison ──\n{header}\n{'-' * len(header)}")
    for row in summary_rows:
        cells = []
        for c in metric_cols:
            cells.append(f"{row[c]:>21.1%}" if row[c] != "" else f"{'—':>22}")
        print(f"{row['experiment']:<{id_w}}  {row['name'][:name_w]:<{name_w}}  "
              f"{row['step']:>6}  " + "  ".join(cells))

    print(f"\nLong format    → {output_path}")
    print(f"Comparison table → {summary_path}")


if __name__ == "__main__":
    main(argv=sys.argv[1:])
