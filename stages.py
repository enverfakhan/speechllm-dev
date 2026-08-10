"""Stage execution unit for speech-llm multi-stage training.

A Stage wraps one entry from the `stages:` list in the run config.  It
configures model trainability in place, builds the optimizer and LR scheduler
for that stage, provides a per-epoch DataLoader, and answers whether the stage
should end so the loop can advance.

Usage pattern (in training.py):

    stage = Stage(cfg.stages[idx], ctx)
    optimizer, scheduler = stage.setup(encoder, adapter, llama)
    for epoch in itertools.count():
        if stage.should_advance({}, step_in_stage, epoch):   # epoch boundary
            break
        loader = stage.make_loader(epoch, stage_idx=idx)
        for batch in loader:
            ...  # forward, backward, scaler.step, scheduler.step
            if stage.should_advance(metrics, step_in_stage, epoch):
                break
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.utils.data

import data
from utils.config import StageConfig
from utils import optim as optim_utils


# ── Module ordering ────────────────────────────────────────────────────────────
# Canonical order for param groups; we iterate this and skip non-trainable ones.
# "audio_adapters" is not a module — it is the name-selected subset of llama's
# parameters holding the gated per-layer audio adapters (see _is_audio_adapter).
_MODULE_ORDER: tuple[str, ...] = ("encoder", "adapter", "llama", "audio_adapters")


def _is_audio_adapter(param_name: str) -> bool:
    """True when a parameter belongs to a gated audio adapter.

    Selected by the substring 'audio_adapter' — the same predicate llama.py,
    the grad metrics, and checkpoint tolerance all use, so they stay in lockstep.
    """
    return "audio_adapter" in param_name


# ── StageContext ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StageContext:
    """Run-level constants shared by all stages.

    A Stage reads these but does not own them; they are set once at run start
    and passed to every Stage constructed during the run.
    """
    shards:               list[str]
    tokenizer_path:       Path
    sep_token_id:         int
    num_workers:          int
    seed:                 int
    # Both variants: [(unfmt_text, "unformatted.txt"), (fmt_text, "formatted.txt")]
    instruction_pairs:    list[tuple[str, str]]
    run_instruction_mode: str
    betas:                tuple[float, float]
    weight_decay:         float
    shuffle_buffer:       int = 1000


# ── Exit-criterion strategy registry ──────────────────────────────────────────

# Every strategy takes the same five arguments — (metrics, step, threshold,
# min_steps, epoch) — and ignores the ones it does not need, so the caller never
# has to know which signal a given strategy reads.

def _exit_first_token_below(
    metrics: dict, step: int, threshold: float | None, min_steps: int, epoch: int
) -> bool:
    val = metrics.get("loss/eval_first_token")
    return (val is not None
            and threshold is not None
            and val < threshold
            and step >= min_steps)


def _exit_eval_loss_below(
    metrics: dict, step: int, threshold: float | None, min_steps: int, epoch: int
) -> bool:
    val = metrics.get("loss/eval_rest")
    return (val is not None
            and threshold is not None
            and val < threshold
            and step >= min_steps)


def _exit_max_steps(
    metrics: dict, step: int, threshold: float | None, min_steps: int, epoch: int
) -> bool:
    # threshold is the per-stage step budget; None means this criterion never fires.
    if threshold is None:
        return False
    return step >= int(threshold)


def _exit_epochs(
    metrics: dict, step: int, threshold: float | None, min_steps: int, epoch: int
) -> bool:
    """End the stage after `threshold` COMPLETED passes over the shard list.

    `epoch` is the 0-based index of the epoch about to run, so it equals the
    number of epochs already completed: epoch >= threshold means the budget is
    spent.  The training loop checks this at the epoch boundary (before the next
    loader is built), which is what makes the exit land exactly on the boundary
    rather than one step into the next epoch.
    """
    if threshold is None:
        return False
    return epoch >= int(threshold) and step >= min_steps


def _exit_never(
    metrics: dict, step: int, threshold: float | None, min_steps: int, epoch: int
) -> bool:
    return False


_EXIT_STRATEGIES: dict[str, Callable[[dict, int, float | None, int, int], bool]] = {
    "first_token_below": _exit_first_token_below,
    "eval_loss_below":   _exit_eval_loss_below,
    "max_steps":         _exit_max_steps,
    "epochs":            _exit_epochs,
    "never":             _exit_never,
}


# ── Stage ─────────────────────────────────────────────────────────────────────

class Stage:
    """One parameterized training stage.

    Consumes a StageConfig and a StageContext.  Does not contain a training
    loop — that is the consumer's responsibility.
    """

    def __init__(self, config: StageConfig, ctx: StageContext) -> None:
        self._config = config
        self._ctx    = ctx

        # Resolve instruction mode: stage override beats run default.
        mode = config.instruction_mode or ctx.run_instruction_mode

        pairs = ctx.instruction_pairs  # index 0=unformatted, 1=formatted
        if mode == "unformatted":
            self._instruction_variants: list[tuple[str, str]] = [pairs[0]]
        elif mode == "formatted":
            self._instruction_variants = [pairs[1]]
        else:  # "both"
            self._instruction_variants = list(pairs)

        # Paired prompts need both variants available and an even sequence budget.
        # Validated here rather than in utils.config because the resolved mode is
        # only known once the run-level default and the stage override are joined.
        if config.paired_prompts:
            if mode != "both":
                raise ValueError(
                    f"stage {config.name!r}: paired_prompts requires the resolved "
                    f"instruction mode to be 'both' (each audio contributes one "
                    f"sequence per variant), got {mode!r}. Set the stage's "
                    f"instruction_mode to 'both' or drop paired_prompts."
                )
            if config.batch_size % 2 != 0:
                raise ValueError(
                    f"stage {config.name!r}: paired_prompts requires an even "
                    f"batch_size (it counts SEQUENCES, two per audio), got "
                    f"{config.batch_size}."
                )

        self._exit_fn = _EXIT_STRATEGIES[config.exit.strategy]

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Stage name, used for checkpoint naming and logging."""
        return self._config.name

    @property
    def accum_steps(self) -> int:
        """Gradient accumulation steps for this stage."""
        return self._config.accum_steps

    @property
    def batch_size(self) -> int:
        """Batch size for this stage."""
        return self._config.batch_size

    @property
    def trainable(self) -> list[str]:
        """Modules trainable in this stage."""
        return list(self._config.trainable)

    @property
    def paired_prompts(self) -> bool:
        """Whether each audio contributes both prompt variants as two sequences.

        When True the loader yields batch_size // 2 unique audios and 2 sequence
        rows per audio; the training loop expands the audio side to match (see
        data.build_dataloader's paired contract).
        """
        return self._config.paired_prompts

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(
        self,
        encoder:        nn.Module,
        adapter:        nn.Module,
        llama:          nn.Module,
        prev_optimizer: Any = None,
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """Configure trainability, build optimizer and LR scheduler.

        Sets requires_grad IN PLACE on all three modules, then returns a fresh
        AdamW8bit optimizer and a LambdaLR scheduler for this stage.

        Args:
            encoder:        WhisperEncoder module
            adapter:        AudioAdapter module
            llama:          Llama module
            prev_optimizer: reserved for the future "inherit" path; ignored for "fresh"

        Returns:
            (optimizer, scheduler)
        """
        if self._config.optimizer_init == "inherit":
            raise NotImplementedError(
                "optimizer_init='inherit' is not yet implemented: bitsandbytes 8-bit "
                "AdamW moment state is not reliably portable across a param-group "
                "change (OPTIMIZER_REFERENCE).  All current configs use 'fresh'."
            )

        modules: dict[str, nn.Module] = {
            "encoder": encoder,
            "adapter": adapter,
            "llama":   llama,
        }
        trainable_set = set(self._config.trainable)

        # Set requires_grad in place on every parameter.  llama is name-aware:
        # its audio-adapter params are governed by "audio_adapters" in the
        # trainable set, every other llama param by "llama".  This lets a stage
        # train the gated adapters while the whole pretrained backbone stays
        # frozen (the two are disjoint parameter subsets of one module).
        llama_on = "llama"          in trainable_set
        aa_on    = "audio_adapters" in trainable_set
        for mod_name, mod in modules.items():
            if mod_name == "llama":
                for name, p in mod.named_parameters():
                    p.requires_grad_(aa_on if _is_audio_adapter(name) else llama_on)
            else:
                grad_on = mod_name in trainable_set
                for p in mod.parameters():
                    p.requires_grad_(grad_on)

        # Selector for each canonical group's parameters.  "audio_adapters" and
        # "llama" both draw from the llama module but from disjoint name subsets.
        def _group_params(group_name: str) -> list[nn.Parameter]:
            if group_name == "llama":
                return [p for n, p in llama.named_parameters()
                        if p.requires_grad and not _is_audio_adapter(n)]
            if group_name == "audio_adapters":
                return [p for n, p in llama.named_parameters()
                        if p.requires_grad and _is_audio_adapter(n)]
            return [p for p in modules[group_name].parameters() if p.requires_grad]

        # Build named param groups in canonical order, skipping non-trainable ones.
        param_groups: list[dict] = []
        for mod_name in _MODULE_ORDER:
            if mod_name not in trainable_set:
                continue
            params = _group_params(mod_name)
            assert params, (
                f"Param group '{mod_name}' is empty after requires_grad — "
                "check model construction (e.g. 'audio_adapters' with audio_adapter_r=0)."
            )
            group: dict = {
                "name":   mod_name,
                "params": params,
                "lr":     self._config.lrs[mod_name],
            }
            # Zero weight decay for the gated adapters: global wd would pull the
            # scalar gate (init 1.0) closed and decay the adapter RMSNorm off 1.
            if mod_name == "audio_adapters":
                group["weight_decay"] = 0.0
            param_groups.append(group)

        optimizer = optim_utils.build_adamw8bit(
            param_groups,
            betas=self._ctx.betas,
            weight_decay=self._ctx.weight_decay,
        )

        # Print before building the scheduler: LambdaLR with a warmup factor of 0
        # at step 0 immediately zeros out g["lr"] in the param groups, so we must
        # read the peak LRs here while they still hold the configured values.
        warmup = self._config.schedule.warmup_steps
        self._print_startup(param_groups, warmup)

        if warmup > 0:
            scheduler = optim_utils.make_warmup_scheduler(optimizer, warmup)
        else:
            # Constant factor so the loop can always call scheduler.step() uniformly.
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda _: 1.0
            )

        return optimizer, scheduler

    def _print_startup(self, param_groups: list[dict], warmup_steps: int) -> None:
        cfg = self._config
        print("─" * 60)
        print(f"Stage: {self.name}")
        for g in param_groups:
            n_params = sum(p.numel() for p in g["params"])
            print(f"  {g['name']:8s}  {n_params / 1e6:8.3f}M params  peak LR {g['lr']:.2e}")
        print(
            f"  warmup_steps={warmup_steps}  "
            f"batch_size={cfg.batch_size}  accum_steps={cfg.accum_steps}"
        )
        print("─" * 60)

    # ── DataLoader ────────────────────────────────────────────────────────────

    def make_loader(self, epoch: int, stage_idx: int) -> torch.utils.data.DataLoader:
        """Build a DataLoader for this stage at a given epoch.

        Reproduces the canonical per-epoch shuffle:
            epoch_shards = list(ctx.shards)
            random.Random(ctx.seed + stage_idx * 10000 + epoch).shuffle(epoch_shards)

        For stage_idx == 0 this reduces to ctx.seed + epoch, preserving the
        existing single-stage behavior.  The stage_idx term decorrelates each
        stage's first pass so stage N never opens with stage 0's ordering.

        Args:
            epoch:     0-based epoch index; controls the shard shuffle seed
            stage_idx: 0-based stage index; decorrelates per-stage shuffles

        Returns:
            DataLoader yielding 6-tuples:
            (mel, audio_lengths, instruction_ids, instruction_lengths,
             transcript_ids, transcript_lengths)

            With paired_prompts the mel/audio_lengths batch dim is
            batch_size // 2 while the instruction/transcript batch dim is
            batch_size — see data.build_dataloader for the full contract.
        """
        ctx = self._ctx
        epoch_shards = list(ctx.shards)
        random.Random(ctx.seed + stage_idx * 10000 + epoch).shuffle(epoch_shards)

        return data.build_dataloader(
            epoch_shards,
            tokenizer_path=ctx.tokenizer_path,
            sep_token_id=ctx.sep_token_id,
            batch_size=self._config.batch_size,
            num_workers=ctx.num_workers,
            instruction_variants=self._instruction_variants,
            shuffle_buffer=ctx.shuffle_buffer,
            paired=self._config.paired_prompts,
        )

    # ── Exit criterion ────────────────────────────────────────────────────────

    def should_advance(self, metrics: dict, step_in_stage: int, epoch: int) -> bool:
        """Return True when this stage should end.

        The global run.max_steps hard cap is the loop's responsibility; this
        method is purely the stage-exit / advance signal.

        Args:
            metrics:       dict of metric name → scalar value, e.g.
                           {"loss/eval_first_token": 0.65, "loss/eval_rest": 1.2}
            step_in_stage: stage-local step count (steps since this stage began)
            epoch:         0-based stage-local epoch index — the epoch currently
                           running, i.e. the count of epochs already completed

        Returns:
            True when the exit criterion fires
        """
        exit_cfg = self._config.exit
        return self._exit_fn(
            metrics, step_in_stage, exit_cfg.threshold, exit_cfg.min_steps, epoch
        )


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import unittest.mock as mock
    from dataclasses import replace
    from pathlib import Path

    from utils.config import load_config, ExitConfig, StageConfig, Schedule

    _PROJECT_ROOT = Path(__file__).parent
    _EXAMPLE = _PROJECT_ROOT / "configs" / "example.yaml"

    cfg = load_config(_EXAMPLE)
    s0_cfg = cfg.stages[0]  # adapter_only:   first_token_below / 0.7 / 500
    s1_cfg = cfg.stages[1]  # full_finetune:  all three trainable, warmup 1000

    # ── Tiny stub modules (CPU, no GPU needed) ─────────────────────────────────
    encoder = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 4))
    adapter = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 4))
    llama   = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 4))

    # ── Hand-built StageContext ────────────────────────────────────────────────
    fake_shards = [f"shard_{i:03d}.tar" for i in range(6)]
    ctx = StageContext(
        shards               = fake_shards,
        tokenizer_path       = Path("data/pruned_tokenizer/"),
        sep_token_id         = 40147,
        num_workers          = 0,
        seed                 = 42,
        instruction_pairs    = [
            ("Transcribe the following audio without formatting.", "unformatted.txt"),
            ("Transcribe the following audio with proper formatting.", "formatted.txt"),
        ],
        run_instruction_mode = "unformatted",
        betas                = (0.9, 0.999),
        weight_decay         = 0.01,
        shuffle_buffer       = 1000,
    )

    # Mock build_adamw8bit so tests run without bitsandbytes.
    def _fake_adamw8bit(param_groups, *, betas, weight_decay):
        return torch.optim.AdamW(param_groups, betas=betas, weight_decay=weight_decay)

    # ── Test: setup stage[0] ───────────────────────────────────────────────────
    with mock.patch.object(optim_utils, "build_adamw8bit", side_effect=_fake_adamw8bit):
        stage0  = Stage(s0_cfg, ctx)
        opt0, sch0 = stage0.setup(encoder, adapter, llama)

    # Only adapter requires grad.
    assert all(p.requires_grad for p in adapter.parameters()), "adapter should require grad"
    assert not any(p.requires_grad for p in encoder.parameters()), "encoder should NOT require grad"
    assert not any(p.requires_grad for p in llama.parameters()), "llama should NOT require grad"

    # Exactly one param group named "adapter" at lr 1e-4.
    assert len(opt0.param_groups) == 1, f"expected 1 group, got {len(opt0.param_groups)}"
    assert opt0.param_groups[0]["name"] == "adapter", opt0.param_groups[0]["name"]
    assert abs(opt0.param_groups[0]["lr"] - 1e-4) < 1e-12, opt0.param_groups[0]["lr"]

    # Constant scheduler: factor is 1.0 at all steps.
    assert abs(sch0.get_last_lr()[0] - 1e-4) < 1e-12, sch0.get_last_lr()
    sch0.step()
    assert abs(sch0.get_last_lr()[0] - 1e-4) < 1e-12, "scheduler should be constant"

    print("[OK] setup stage[0]")

    # ── Test: setup stage[1] ───────────────────────────────────────────────────
    with mock.patch.object(optim_utils, "build_adamw8bit", side_effect=_fake_adamw8bit):
        stage1      = Stage(s1_cfg, ctx)
        opt1, sch1  = stage1.setup(encoder, adapter, llama)

    # All three require grad.
    assert all(p.requires_grad for p in encoder.parameters()), "encoder should require grad"
    assert all(p.requires_grad for p in adapter.parameters()), "adapter should require grad"
    assert all(p.requires_grad for p in llama.parameters()),   "llama should require grad"

    # Three named groups in canonical order with correct LRs.
    assert len(opt1.param_groups) == 3, f"expected 3 groups, got {len(opt1.param_groups)}"
    expected_groups = [
        ("encoder", 1.0e-6),
        ("adapter", 5.0e-5),
        ("llama",   1.5e-5),
    ]
    for i, (exp_name, exp_lr) in enumerate(expected_groups):
        g = opt1.param_groups[i]
        assert g["name"] == exp_name, f"group[{i}] name: expected {exp_name!r}, got {g['name']!r}"
        # LambdaLR with warmup zeros g["lr"] at init (step 0 → factor 0); peak is in initial_lr.
        peak = g.get("initial_lr", g["lr"])
        assert abs(peak - exp_lr) < 1e-15, f"group[{i}] peak LR: expected {exp_lr}, got {peak}"

    # Warmup scheduler: factor rises 0 → 1 over 1000 steps.
    # get_last_lr() returns base_lr * factor; after 0 optimizer steps factor=0.
    warmup_steps = s1_cfg.schedule.warmup_steps  # 1000
    assert warmup_steps == 1000, warmup_steps
    # At step 0 (before any sch1.step()), LambdaLR initialises with last_epoch=-1→0
    # and fires lr_lambda(0) = min(0/1000, 1.0) = 0.0.
    assert abs(sch1.get_last_lr()[0]) < 1e-15, f"warmup factor at step 0 should be 0, got {sch1.get_last_lr()[0]}"
    for _ in range(warmup_steps):
        sch1.step()
    assert abs(sch1.get_last_lr()[0] - 1.0e-6) < 1e-15, (
        f"warmup factor at step {warmup_steps} should yield peak LR 1e-6, "
        f"got {sch1.get_last_lr()[0]}"
    )

    print("[OK] setup stage[1]")

    # ── Test: audio_adapters group (name-selected subset of llama) ─────────────
    class _FakeLlamaWithAdapters(nn.Module):
        """llama stub: one 'backbone' param (frozen) + one 'audio_adapter' param."""
        def __init__(self) -> None:
            super().__init__()
            self.backbone      = nn.Linear(4, 8)   # stands in for pretrained weights
            self.audio_adapter = nn.Linear(8, 4)   # names contain "audio_adapter"

    aa_cfg = StageConfig(
        name             = "audio_layer_adapters",
        trainable        = ["audio_adapters"],
        lrs              = {"audio_adapters": 1.0e-4},
        schedule         = Schedule(warmup_steps=0),
        batch_size       = 64,
        accum_steps      = 1,
        exit             = ExitConfig(strategy="never", threshold=None, min_steps=0),
    )

    enc_aa = nn.Linear(4, 4)
    ada_aa = nn.Linear(4, 4)
    llm_aa = _FakeLlamaWithAdapters()

    with mock.patch.object(optim_utils, "build_adamw8bit", side_effect=_fake_adamw8bit):
        stage_aa       = Stage(aa_cfg, ctx)
        opt_aa, sch_aa = stage_aa.setup(enc_aa, ada_aa, llm_aa)

    # Exactly one param group, named "audio_adapters", with weight_decay 0.0.
    assert len(opt_aa.param_groups) == 1, f"expected 1 group, got {len(opt_aa.param_groups)}"
    assert opt_aa.param_groups[0]["name"] == "audio_adapters", opt_aa.param_groups[0]["name"]
    assert opt_aa.param_groups[0]["weight_decay"] == 0.0, opt_aa.param_groups[0]["weight_decay"]
    assert abs(opt_aa.param_groups[0]["lr"] - 1e-4) < 1e-12, opt_aa.param_groups[0]["lr"]

    # requires_grad True only for name-matching llama params.
    for name, p in llm_aa.named_parameters():
        if "audio_adapter" in name:
            assert p.requires_grad, f"{name} should require grad"
        else:
            assert not p.requires_grad, f"{name} should NOT require grad"
    assert not any(p.requires_grad for p in enc_aa.parameters()), "encoder must stay frozen"
    assert not any(p.requires_grad for p in ada_aa.parameters()), "adapter must stay frozen"

    print("[OK] setup audio_adapters group")

    # ── Test: empty audio_adapters group is caught by the assert guard ─────────
    # A plain llama stub has no 'audio_adapter' params, so trainable=[audio_adapters]
    # selects nothing — this must fail loudly (mirrors audio_adapter_r=0).
    llm_plain = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 4))
    _empty_raised = False
    try:
        with mock.patch.object(optim_utils, "build_adamw8bit", side_effect=_fake_adamw8bit):
            Stage(aa_cfg, ctx).setup(nn.Linear(4, 4), nn.Linear(4, 4), llm_plain)
    except AssertionError as exc:
        _empty_raised = True
        assert "audio_adapters" in str(exc), str(exc)
    assert _empty_raised, "expected AssertionError for empty audio_adapters group"

    print("[OK] empty audio_adapters group caught by guard")

    # ── Test: optimizer_init "inherit" raises NotImplementedError ──────────────
    inherit_cfg = replace(s0_cfg, optimizer_init="inherit")
    stage_inherit = Stage(inherit_cfg, ctx)
    try:
        with mock.patch.object(optim_utils, "build_adamw8bit", side_effect=_fake_adamw8bit):
            stage_inherit.setup(encoder, adapter, llama)
        raise AssertionError("Expected NotImplementedError for optimizer_init='inherit'")
    except NotImplementedError as exc:
        assert "inherit" in str(exc).lower() or "portable" in str(exc).lower(), str(exc)

    print("[OK] optimizer_init 'inherit' raises NotImplementedError")

    # ── Test: should_advance ───────────────────────────────────────────────────

    # stage[0]: first_token_below, threshold=0.7, min_steps=500
    assert s0_cfg.exit.strategy  == "first_token_below", s0_cfg.exit.strategy
    assert s0_cfg.exit.threshold == 0.7,                 s0_cfg.exit.threshold
    assert s0_cfg.exit.min_steps == 500,                 s0_cfg.exit.min_steps

    s0_stage = Stage(s0_cfg, ctx)

    # Below min_steps: never fires regardless of metric.
    assert not s0_stage.should_advance({"loss/eval_first_token": 0.5}, step_in_stage=0, epoch=0)
    assert not s0_stage.should_advance({"loss/eval_first_token": 0.5}, step_in_stage=499, epoch=0)

    # Metric absent: never fires (past min_steps).
    assert not s0_stage.should_advance({}, step_in_stage=500, epoch=0)
    assert not s0_stage.should_advance({}, step_in_stage=1000, epoch=3)

    # Metric present but above threshold: does not fire.
    assert not s0_stage.should_advance({"loss/eval_first_token": 0.7},  step_in_stage=500, epoch=0)
    assert not s0_stage.should_advance({"loss/eval_first_token": 0.75}, step_in_stage=1000, epoch=0)

    # Metric present, below threshold, past min_steps: fires.
    assert s0_stage.should_advance({"loss/eval_first_token": 0.69}, step_in_stage=500, epoch=0)
    assert s0_stage.should_advance({"loss/eval_first_token": 0.0},  step_in_stage=9999, epoch=0)

    print("[OK] should_advance: first_token_below")

    # max_steps stage: fires when step_in_stage >= threshold.
    max_cfg = replace(
        s0_cfg,
        exit=ExitConfig(strategy="max_steps", threshold=5000.0, min_steps=0),
    )
    ms_stage = Stage(max_cfg, ctx)
    assert not ms_stage.should_advance({}, step_in_stage=4999, epoch=0)
    assert     ms_stage.should_advance({}, step_in_stage=5000, epoch=0)
    assert     ms_stage.should_advance({}, step_in_stage=5001, epoch=0)

    print("[OK] should_advance: max_steps")

    # epochs stage: fires at the first check where epoch >= threshold, whatever
    # the step count — the training loop only asks at an epoch boundary.
    ep_cfg = replace(
        s0_cfg,
        exit=ExitConfig(strategy="epochs", threshold=2, min_steps=0),
    )
    ep_stage = Stage(ep_cfg, ctx)
    for _epoch in (0, 1):
        for _step in (0, 1, 12345):
            assert not ep_stage.should_advance({}, step_in_stage=_step, epoch=_epoch), (
                f"epochs(2) must not fire during epoch {_epoch} (step {_step})"
            )
    assert ep_stage.should_advance({}, step_in_stage=0, epoch=2), (
        "epochs(2) must fire at the epoch-2 boundary even at step 0"
    )
    assert ep_stage.should_advance({}, step_in_stage=99, epoch=7)

    # min_steps still gates it: a short epoch must not end the stage early.
    ep_gated = Stage(
        replace(s0_cfg, exit=ExitConfig(strategy="epochs", threshold=2, min_steps=1000)),
        ctx,
    )
    assert not ep_gated.should_advance({}, step_in_stage=999,  epoch=2)
    assert     ep_gated.should_advance({}, step_in_stage=1000, epoch=2)

    print("[OK] should_advance: epochs")

    # never stage: always False.
    never_cfg = replace(
        s0_cfg,
        exit=ExitConfig(strategy="never", threshold=None, min_steps=0),
    )
    nv_stage = Stage(never_cfg, ctx)
    assert not nv_stage.should_advance({}, step_in_stage=0, epoch=0)
    assert not nv_stage.should_advance(
        {"loss/eval_first_token": 0.0}, step_in_stage=10**9, epoch=10**6
    )

    print("[OK] should_advance: never")

    # ── Test: make_loader ──────────────────────────────────────────────────────

    captured: dict = {}

    def _mock_build_dataloader(
        shards_arg, *, tokenizer_path, sep_token_id, batch_size,
        num_workers, instruction_variants, shuffle_buffer, paired=False,
    ):
        captured["shards"]                = shards_arg
        captured["batch_size"]            = batch_size
        captured["instruction_variants"]  = instruction_variants
        captured["paired"]                = paired
        return []  # dummy loader

    with mock.patch.object(data, "build_dataloader", side_effect=_mock_build_dataloader):
        stage0_loader = Stage(s0_cfg, ctx)
        stage0_loader.make_loader(epoch=3, stage_idx=0)

    # batch_size matches stage config.
    assert captured["batch_size"] == s0_cfg.batch_size, (
        f"expected batch_size={s0_cfg.batch_size}, got {captured['batch_size']}"
    )

    # instruction_variants is [unformatted pair] (stage[0] inherits "unformatted" mode).
    assert captured["instruction_variants"] == [ctx.instruction_pairs[0]], (
        f"wrong instruction_variants: {captured['instruction_variants']}"
    )

    # Shards are shuffled with seed + stage_idx*10000 + epoch (stage_idx=0 reduces to seed+epoch).
    expected_shards = list(ctx.shards)
    random.Random(ctx.seed + 0 * 10000 + 3).shuffle(expected_shards)
    assert captured["shards"] == expected_shards, (
        f"shard order mismatch\n  got:      {captured['shards']}\n"
        f"  expected: {expected_shards}"
    )

    # Default stage is unpaired.
    assert captured["paired"] is False, captured["paired"]

    print("[OK] make_loader")

    # ── Test: paired_prompts ───────────────────────────────────────────────────
    # ctx.run_instruction_mode is "unformatted", so the stage override is what
    # decides the resolved mode here.
    paired_cfg = replace(
        s0_cfg, paired_prompts=True, instruction_mode="both", batch_size=4
    )
    with mock.patch.object(data, "build_dataloader", side_effect=_mock_build_dataloader):
        paired_stage = Stage(paired_cfg, ctx)
        paired_stage.make_loader(epoch=0, stage_idx=0)

    assert paired_stage.paired_prompts is True, "paired_prompts property"
    assert captured["paired"] is True, "make_loader must forward paired=True"
    # Both variants reach the loader — one sequence per variant, per audio.
    assert captured["instruction_variants"] == list(ctx.instruction_pairs), (
        f"paired stage must use both variants: {captured['instruction_variants']}"
    )
    assert captured["batch_size"] == 4, captured["batch_size"]

    # Resolved mode other than "both" is refused.
    _mode_raised = False
    try:
        Stage(replace(paired_cfg, instruction_mode="unformatted"), ctx)
    except ValueError as exc:
        _mode_raised = True
        assert "both" in str(exc), str(exc)
    assert _mode_raised, "paired_prompts with mode 'unformatted' must raise ValueError"

    # A stage that inherits a non-"both" run mode is refused the same way.
    _inherit_raised = False
    try:
        Stage(replace(paired_cfg, instruction_mode=None), ctx)  # ctx: "unformatted"
    except ValueError as exc:
        _inherit_raised = True
        assert "both" in str(exc), str(exc)
    assert _inherit_raised, "paired_prompts inheriting a non-'both' run mode must raise"

    # Odd batch_size cannot be split into audio pairs.
    _odd_raised = False
    try:
        Stage(replace(paired_cfg, batch_size=5), ctx)
    except ValueError as exc:
        _odd_raised = True
        assert "even" in str(exc), str(exc)
    assert _odd_raised, "paired_prompts with an odd batch_size must raise ValueError"

    print("[OK] paired_prompts")

    print("\nPASSED")
    sys.exit(0)
