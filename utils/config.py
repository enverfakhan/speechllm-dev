"""Typed configuration layer for speech-llm training.

Load order: configs/base.yaml → optional run config → thin CLI overrides.
See load_config() for the full merge semantics.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field, replace  # noqa: F401 (replace used in __main__)
from pathlib import Path
from typing import Any

import yaml


_PROJECT_ROOT = Path(__file__).parent.parent
_BASE_YAML    = _PROJECT_ROOT / "configs" / "base.yaml"


# ── Helpers ───────────────────────────────────────────────────────────────────

def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override onto base. Lists are replaced wholesale."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _opt_path(v: Any) -> Path | None:
    return Path(v) if v is not None else None


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EvalShards:
    """Paths to evaluation shards for each LibriSpeech split."""
    dev_clean:  Path | None = None
    dev_other:  Path | None = None
    test_clean: Path | None = None
    test_other: Path | None = None


@dataclass(frozen=True)
class DataConfig:
    """Data loading configuration."""
    shards_file: Path | None = None
    shards:      str  | None = None
    tokenizer:   Path | None = Path("data/pruned_tokenizer/")
    diag_shard:  Path | None = None
    eval:        EvalShards  = field(default_factory=EvalShards)
    num_workers: int         = 4


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture and checkpoint paths."""
    stub:             bool       = False
    stub_dims:        dict       = field(default_factory=lambda: {
        "n_layers": 6, "d_model": 512, "n_heads": 8, "n_kv_heads": 2, "intermediate_size": 1024,
    })
    whisper_ckpt:     Path | None = Path("weights/whisper_small.pt")
    llama_ckpt:       Path | None = Path("weights/Llama3.1-8B/")
    adapter_pca_init: Path | None = None


@dataclass(frozen=True)
class GradClip:
    """Gradient clipping configuration."""
    enabled:  bool  = True
    max_norm: float = 1.0


@dataclass(frozen=True)
class OptimConfig:
    """Optimizer configuration."""
    betas:        tuple    = (0.9, 0.999)
    weight_decay: float    = 0.01
    grad_clip:    GradClip = field(default_factory=GradClip)


@dataclass(frozen=True)
class LoggingConfig:
    """Logging and experiment-tracking configuration."""
    wandb:    bool        = False
    project:  str         = "speech-llm-dev"
    run_name: str | None  = None


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpoint saving configuration."""
    dir:        Path      = Path("checkpoints")
    save_every: int | None = None


@dataclass(frozen=True)
class WerConfig:
    """WER evaluation configuration."""
    enabled:                    bool  = False
    period:                     int   = 500
    max_batches:                int   = 5
    readiness_first_token_below: float = 3.0
    sample_transcriptions:      int   = 20


@dataclass(frozen=True)
class MetricsConfig:
    """Metrics and evaluation configuration."""
    train_periods:        dict      = field(default_factory=lambda: {"grads": 1, "logits": 10, "loss": 1})
    eval_every:           int       = 10
    eval_batches:         int       = 3
    eval_batch_size:      int       = 8
    compute_only_on_emit: dict      = field(default_factory=lambda: {"logits": False})
    wer:                  WerConfig = field(default_factory=WerConfig)


@dataclass(frozen=True)
class RunConfig:
    """Top-level run configuration."""
    max_steps:        int | None = None
    start_stage:      int        = 0
    instruction_mode: str        = "unformatted"


@dataclass(frozen=True)
class Schedule:
    """Learning-rate schedule for a stage."""
    warmup_steps: int = 0


@dataclass(frozen=True)
class ExitConfig:
    """Stage exit condition."""
    strategy:  str         = "max_steps"
    threshold: float | None = None
    min_steps: int          = 0


@dataclass(frozen=True)
class StageConfig:
    """Per-stage training configuration."""
    name:             str             = "stage"
    trainable:        list[str]       = field(default_factory=lambda: ["adapter"])
    lrs:              dict[str, float] = field(default_factory=lambda: {"adapter": 1.0e-4})
    schedule:         Schedule        = field(default_factory=Schedule)
    batch_size:       int             = 64
    accum_steps:      int             = 1
    instruction_mode: str | None      = None
    optimizer_init:   str             = "fresh"
    exit:             ExitConfig      = field(default_factory=ExitConfig)


@dataclass(frozen=True)
class Config:
    """Root configuration object."""
    seed:       int           = 42
    data:       DataConfig    = field(default_factory=DataConfig)
    model:      ModelConfig   = field(default_factory=ModelConfig)
    optim:      OptimConfig   = field(default_factory=OptimConfig)
    logging:    LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    metrics:    MetricsConfig = field(default_factory=MetricsConfig)
    run:        RunConfig     = field(default_factory=RunConfig)
    stages:     list[StageConfig] = field(default_factory=list)
    resume:     Path | None   = None  # CLI-only; not present in YAML


# ── Dict → typed construction ─────────────────────────────────────────────────

def _build_eval_shards(d: dict) -> EvalShards:
    return EvalShards(
        dev_clean  = _opt_path(d.get("dev_clean")),
        dev_other  = _opt_path(d.get("dev_other")),
        test_clean = _opt_path(d.get("test_clean")),
        test_other = _opt_path(d.get("test_other")),
    )


def _build_data(d: dict) -> DataConfig:
    eval_raw = d.get("eval") or {}
    return DataConfig(
        shards_file = _opt_path(d.get("shards_file")),
        shards      = d.get("shards"),
        tokenizer   = _opt_path(d.get("tokenizer")),
        diag_shard  = _opt_path(d.get("diag_shard")),
        eval        = _build_eval_shards(eval_raw) if isinstance(eval_raw, dict) else EvalShards(),
        num_workers = int(d.get("num_workers", 4)),
    )


def _build_model(d: dict) -> ModelConfig:
    return ModelConfig(
        stub             = bool(d.get("stub", False)),
        stub_dims        = dict(d.get("stub_dims") or {}),
        whisper_ckpt     = _opt_path(d.get("whisper_ckpt")),
        llama_ckpt       = _opt_path(d.get("llama_ckpt")),
        adapter_pca_init = _opt_path(d.get("adapter_pca_init")),
    )


def _build_optim(d: dict) -> OptimConfig:
    betas_raw = d.get("betas", [0.9, 0.999])
    clip_raw  = d.get("grad_clip") or {}
    return OptimConfig(
        betas        = tuple(float(b) for b in betas_raw),
        weight_decay = float(d.get("weight_decay", 0.01)),
        grad_clip    = GradClip(
            enabled  = bool(clip_raw.get("enabled", True)),
            max_norm = float(clip_raw.get("max_norm", 1.0)),
        ),
    )


def _build_logging(d: dict) -> LoggingConfig:
    return LoggingConfig(
        wandb    = bool(d.get("wandb", False)),
        project  = str(d.get("project", "speech-llm-dev")),
        run_name = d.get("run_name"),
    )


def _build_checkpoint(d: dict) -> CheckpointConfig:
    return CheckpointConfig(
        dir        = Path(d.get("dir", "checkpoints")),
        save_every = d.get("save_every"),
    )


def _build_wer(d: dict) -> WerConfig:
    return WerConfig(
        enabled                     = bool(d.get("enabled", False)),
        period                      = int(d.get("period", 500)),
        max_batches                 = int(d.get("max_batches", 5)),
        readiness_first_token_below = float(d.get("readiness_first_token_below", 3.0)),
        sample_transcriptions       = int(d.get("sample_transcriptions", 20)),
    )


def _build_metrics(d: dict) -> MetricsConfig:
    wer_raw = d.get("wer") or {}
    return MetricsConfig(
        train_periods        = dict(d.get("train_periods") or {"grads": 1, "logits": 10, "loss": 1}),
        eval_every           = int(d.get("eval_every", 10)),
        eval_batches         = int(d.get("eval_batches", 3)),
        eval_batch_size      = int(d.get("eval_batch_size", 8)),
        compute_only_on_emit = dict(d.get("compute_only_on_emit") or {"logits": False}),
        wer                  = _build_wer(wer_raw) if isinstance(wer_raw, dict) else WerConfig(),
    )


def _build_run(d: dict) -> RunConfig:
    return RunConfig(
        max_steps        = d.get("max_steps"),
        start_stage      = int(d.get("start_stage", 0)),
        instruction_mode = str(d.get("instruction_mode", "unformatted")),
    )


def _build_stage(d: dict, run_instruction_mode: str) -> StageConfig:
    sched_raw = d.get("schedule") or {}
    exit_raw  = d.get("exit") or {}
    lrs_raw   = d.get("lrs") or {}
    inst_mode = d.get("instruction_mode")  # None when YAML key is null or absent
    if inst_mode is None:
        inst_mode = run_instruction_mode
    return StageConfig(
        name             = str(d.get("name", "stage")),
        trainable        = list(d.get("trainable", ["adapter"])),
        lrs              = {k: float(v) for k, v in lrs_raw.items()},
        schedule         = Schedule(warmup_steps=int(sched_raw.get("warmup_steps", 0))),
        batch_size       = int(d.get("batch_size", 64)),
        accum_steps      = int(d.get("accum_steps", 1)),
        instruction_mode = inst_mode,
        optimizer_init   = str(d.get("optimizer_init", "fresh")),
        exit             = ExitConfig(
            strategy  = str(exit_raw.get("strategy", "max_steps")),
            threshold = exit_raw.get("threshold"),
            min_steps = int(exit_raw.get("min_steps", 0)),
        ),
    )


def _assemble(d: dict, stages: list[StageConfig], resume: Path | None) -> Config:
    return Config(
        seed       = int(d.get("seed", 42)),
        data       = _build_data(d.get("data") or {}),
        model      = _build_model(d.get("model") or {}),
        optim      = _build_optim(d.get("optim") or {}),
        logging    = _build_logging(d.get("logging") or {}),
        checkpoint = _build_checkpoint(d.get("checkpoint") or {}),
        metrics    = _build_metrics(d.get("metrics") or {}),
        run        = _build_run(d.get("run") or {}),
        stages     = stages,
        resume     = resume,
    )


# ── Validation ────────────────────────────────────────────────────────────────

_VALID_INSTRUCTION_MODES = {"unformatted", "formatted", "both"}
_VALID_TRAINABLE         = {"encoder", "adapter", "llama"}
_VALID_OPTIMIZER_INIT    = {"fresh", "inherit"}
_VALID_EXIT_STRATEGIES   = {"first_token_below", "eval_loss_below", "max_steps", "never"}
_EXIT_NEEDS_THRESHOLD    = {"first_token_below", "eval_loss_below"}
_VALID_METRIC_FAMILIES   = {"grads", "logits", "loss"}


def _validate(cfg: Config) -> None:
    """Raise ValueError with a key-path message if the config is invalid."""

    if not cfg.stages:
        raise ValueError("stages: must contain at least one stage")

    # data
    if cfg.data.shards_file is None and cfg.data.shards is None:
        raise ValueError("data: at least one of shards_file / shards must be set")
    if cfg.data.tokenizer is None:
        raise ValueError("data.tokenizer: required")

    # model
    if not cfg.model.stub:
        if cfg.model.whisper_ckpt is None:
            raise ValueError("model.whisper_ckpt: required when model.stub is false")
        if cfg.model.llama_ckpt is None:
            raise ValueError("model.llama_ckpt: required when model.stub is false")

    # run
    if cfg.run.instruction_mode not in _VALID_INSTRUCTION_MODES:
        raise ValueError(
            f"run.instruction_mode: must be one of {_VALID_INSTRUCTION_MODES}, "
            f"got {cfg.run.instruction_mode!r}"
        )
    if cfg.run.max_steps is not None and cfg.run.max_steps <= 0:
        raise ValueError(f"run.max_steps: must be > 0, got {cfg.run.max_steps}")
    if not (0 <= cfg.run.start_stage < len(cfg.stages)):
        raise ValueError(
            f"run.start_stage: must be in [0, {len(cfg.stages)}), got {cfg.run.start_stage}"
        )

    # optim
    b0, b1 = cfg.optim.betas
    if not (0.0 < b0 < 1.0 and 0.0 < b1 < 1.0):
        raise ValueError(f"optim.betas: both values must be in (0, 1), got {cfg.optim.betas}")
    if cfg.optim.weight_decay < 0:
        raise ValueError(f"optim.weight_decay: must be >= 0, got {cfg.optim.weight_decay}")
    if cfg.optim.grad_clip.enabled and cfg.optim.grad_clip.max_norm <= 0:
        raise ValueError(
            f"optim.grad_clip.max_norm: must be > 0, got {cfg.optim.grad_clip.max_norm}"
        )

    # metrics
    for k in cfg.metrics.train_periods:
        if k not in _VALID_METRIC_FAMILIES:
            raise ValueError(f"metrics.train_periods: key {k!r} not in {_VALID_METRIC_FAMILIES}")
    for k, v in cfg.metrics.train_periods.items():
        if not isinstance(v, int) or v <= 0:
            raise ValueError(f"metrics.train_periods.{k}: must be int > 0, got {v!r}")
    for attr in ("eval_every", "eval_batches", "eval_batch_size"):
        v = getattr(cfg.metrics, attr)
        if v <= 0:
            raise ValueError(f"metrics.{attr}: must be > 0, got {v}")
    for k in cfg.metrics.compute_only_on_emit:
        if k not in _VALID_METRIC_FAMILIES:
            raise ValueError(
                f"metrics.compute_only_on_emit: key {k!r} not in {_VALID_METRIC_FAMILIES}"
            )
    wer = cfg.metrics.wer
    if wer.period <= 0:
        raise ValueError(f"metrics.wer.period: must be > 0, got {wer.period}")
    if wer.max_batches <= 0:
        raise ValueError(f"metrics.wer.max_batches: must be > 0, got {wer.max_batches}")
    if wer.sample_transcriptions <= 0:
        raise ValueError(
            f"metrics.wer.sample_transcriptions: must be > 0, got {wer.sample_transcriptions}"
        )

    # per-stage
    for i, stage in enumerate(cfg.stages):
        pfx = f"stages[{i}] ({stage.name!r})"

        if not stage.trainable:
            raise ValueError(f"{pfx}.trainable: must be non-empty")
        unknown = set(stage.trainable) - _VALID_TRAINABLE
        if unknown:
            raise ValueError(f"{pfx}.trainable: unknown modules {unknown}")

        lrs_keys      = set(stage.lrs.keys())
        trainable_set = set(stage.trainable)
        if lrs_keys != trainable_set:
            raise ValueError(
                f"{pfx}.lrs: keys {lrs_keys} must equal trainable set {trainable_set} exactly"
            )
        for mod, lr in stage.lrs.items():
            if lr <= 0:
                raise ValueError(f"{pfx}.lrs.{mod}: must be float > 0, got {lr}")

        if stage.batch_size <= 0:
            raise ValueError(f"{pfx}.batch_size: must be > 0, got {stage.batch_size}")
        if stage.accum_steps <= 0:
            raise ValueError(f"{pfx}.accum_steps: must be > 0, got {stage.accum_steps}")
        if stage.schedule.warmup_steps < 0:
            raise ValueError(
                f"{pfx}.schedule.warmup_steps: must be >= 0, got {stage.schedule.warmup_steps}"
            )
        if stage.optimizer_init not in _VALID_OPTIMIZER_INIT:
            raise ValueError(
                f"{pfx}.optimizer_init: must be one of {_VALID_OPTIMIZER_INIT}, "
                f"got {stage.optimizer_init!r}"
            )
        if stage.exit.strategy not in _VALID_EXIT_STRATEGIES:
            raise ValueError(
                f"{pfx}.exit.strategy: must be one of {_VALID_EXIT_STRATEGIES}, "
                f"got {stage.exit.strategy!r}"
            )
        if stage.exit.strategy in _EXIT_NEEDS_THRESHOLD and stage.exit.threshold is None:
            raise ValueError(
                f"{pfx}.exit.threshold: required when strategy is {stage.exit.strategy!r}"
            )
        if stage.exit.min_steps < 0:
            raise ValueError(f"{pfx}.exit.min_steps: must be >= 0, got {stage.exit.min_steps}")
        if (stage.instruction_mode is not None
                and stage.instruction_mode not in _VALID_INSTRUCTION_MODES):
            raise ValueError(
                f"{pfx}.instruction_mode: must be one of {_VALID_INSTRUCTION_MODES} or null, "
                f"got {stage.instruction_mode!r}"
            )


# ── Public entry point ────────────────────────────────────────────────────────

def load_config(
    config_path: Path | None = None,
    argv: list[str] | None = None,
) -> Config:
    """Load and merge config from base.yaml, an optional run config, and CLI overrides.

    Merge order:
      1. configs/base.yaml (base)
      2. config_path deep-merged over base; stages list replaced wholesale
      3. stage_defaults = deep_merge(base.stage_defaults, run.stage_defaults or {})
      4. each stage merged onto stage_defaults; null instruction_mode inherits run value
      5. CLI argv overrides applied last
    """
    effective_argv: list[str] = argv if argv is not None else []

    # Parse thin CLI first so --config can be used to resolve config_path.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config",         type=Path, default=None)
    parser.add_argument("--resume",         type=Path, default=None)
    parser.add_argument("--wandb",          action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--run-name",       dest="run_name",       default=None)
    parser.add_argument("--checkpoint-dir", dest="checkpoint_dir", default=None)
    ns, _ = parser.parse_known_args(effective_argv)

    # Explicit arg wins over --config in argv.
    if config_path is None:
        config_path = ns.config

    # 1. Load base.
    with open(_BASE_YAML) as f:
        base: dict = yaml.safe_load(f)

    # 2. Load run config and merge.
    run_cfg: dict = {}
    merged = copy.deepcopy(base)
    if config_path is not None:
        with open(config_path) as f:
            run_cfg = yaml.safe_load(f) or {}
        # stages replaced wholesale; everything else deep-merged
        run_without_stages = {k: v for k, v in run_cfg.items() if k != "stages"}
        merged = deep_merge(merged, run_without_stages)
        if "stages" in run_cfg:
            merged["stages"] = run_cfg["stages"]

    # 3. Resolve stage_defaults.
    base_sd  = base.get("stage_defaults") or {}
    run_sd   = run_cfg.get("stage_defaults") or {}
    stage_defaults = deep_merge(base_sd, run_sd)

    # 4. Build stages: each entry deep-merged onto stage_defaults.
    run_inst_mode = (merged.get("run") or {}).get("instruction_mode", "unformatted")
    raw_stages    = merged.get("stages") or []
    stages        = [
        _build_stage(deep_merge(stage_defaults, s), run_inst_mode)
        for s in raw_stages
    ]

    # 5. Apply CLI overrides to merged dict.
    if ns.wandb is not None:
        log_d = dict(merged.get("logging") or {})
        log_d["wandb"] = ns.wandb
        merged["logging"] = log_d
    if ns.run_name is not None:
        log_d = dict(merged.get("logging") or {})
        log_d["run_name"] = ns.run_name
        merged["logging"] = log_d
    if ns.checkpoint_dir is not None:
        ckpt_d = dict(merged.get("checkpoint") or {})
        ckpt_d["dir"] = ns.checkpoint_dir
        merged["checkpoint"] = ckpt_d

    cfg = _assemble(merged, stages, resume=ns.resume)
    _validate(cfg)
    return cfg


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from dataclasses import replace  # noqa: F811

    _EXAMPLE = _PROJECT_ROOT / "configs" / "example.yaml"

    def _assert_raises(bad_cfg: Config, fragment: str) -> None:
        try:
            _validate(bad_cfg)
            raise AssertionError(
                f"Expected ValueError containing {fragment!r} but none was raised"
            )
        except ValueError as exc:
            msg = str(exc).lower()
            if fragment.lower() not in msg:
                raise AssertionError(
                    f"ValueError {str(exc)!r} does not contain {fragment!r}"
                ) from exc

    # ── 1. load example.yaml ──────────────────────────────────────────────────
    cfg = load_config(_EXAMPLE)

    # run-level overrides beat base
    assert cfg.checkpoint.dir == Path("checkpoints/example"), cfg.checkpoint.dir
    assert cfg.checkpoint.save_every == 360, cfg.checkpoint.save_every
    assert cfg.run.max_steps == 5000, cfg.run.max_steps
    assert cfg.data.shards_file == Path("data/subset_shards.txt"), cfg.data.shards_file

    # stage[0]: accum_steps from run config; optimizer_init defaults to "fresh"
    s0 = cfg.stages[0]
    assert s0.name == "adapter_only", s0.name
    assert s0.accum_steps == 2, s0.accum_steps
    assert s0.optimizer_init == "fresh", s0.optimizer_init
    assert s0.exit.strategy == "first_token_below", s0.exit.strategy
    assert s0.exit.threshold == 0.7, s0.exit.threshold
    assert s0.exit.min_steps == 500, s0.exit.min_steps
    assert set(s0.lrs.keys()) == {"adapter"}, s0.lrs

    # stage[1]: all three trainable, lrs covers exactly those three
    s1 = cfg.stages[1]
    assert set(s1.trainable) == {"encoder", "adapter", "llama"}, s1.trainable
    assert set(s1.lrs.keys()) == {"encoder", "adapter", "llama"}, s1.lrs
    assert s1.schedule.warmup_steps == 1000, s1.schedule.warmup_steps

    print("[OK] load example.yaml")

    # ── 2. CLI overrides ──────────────────────────────────────────────────────
    cfg2 = load_config(argv=[
        "--config",    str(_EXAMPLE),
        "--no-wandb",
        "--run-name",  "x",
        "--resume",    "ckpt.pt",
    ])
    assert cfg2.logging.wandb    == False,           cfg2.logging.wandb
    assert cfg2.logging.run_name == "x",             cfg2.logging.run_name
    assert cfg2.resume           == Path("ckpt.pt"), cfg2.resume

    # --wandb overrides the YAML false → true
    cfg3 = load_config(argv=["--config", str(_EXAMPLE), "--wandb"])
    assert cfg3.logging.wandb == True, cfg3.logging.wandb

    print("[OK] CLI overrides")

    # ── 3. Negative tests ─────────────────────────────────────────────────────

    # lrs key not in trainable
    bad_s = replace(cfg.stages[0], trainable=["adapter"], lrs={"encoder": 1e-4})
    _assert_raises(replace(cfg, stages=[bad_s]), "lrs")
    print("[OK] lrs key not in trainable")

    # unknown exit.strategy
    bad_exit = replace(cfg.stages[0].exit, strategy="totally_bogus")
    bad_s2   = replace(cfg.stages[0], exit=bad_exit)
    _assert_raises(replace(cfg, stages=[bad_s2]), "exit.strategy")
    print("[OK] unknown exit.strategy")

    # first_token_below with no threshold
    bad_exit3 = replace(cfg.stages[0].exit, strategy="first_token_below", threshold=None)
    bad_s3    = replace(cfg.stages[0], exit=bad_exit3)
    _assert_raises(replace(cfg, stages=[bad_s3]), "threshold")
    print("[OK] first_token_below with no threshold")

    # empty stages
    _assert_raises(replace(cfg, stages=[]), "stages")
    print("[OK] empty stages")

    # start_stage out of range
    bad_run = replace(cfg.run, start_stage=99)
    _assert_raises(replace(cfg, run=bad_run), "start_stage")
    print("[OK] start_stage out of range")

    print("\nPASSED")
    sys.exit(0)
