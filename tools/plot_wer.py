"""Plot a WER-vs-step curve from a tools/run_wer.py summary CSV.

One line per (split, format), with the training programme drawn on top of it:
stage boundaries as solid verticals, the stage name written in the span it
covers, and epoch transitions as thin dotted verticals.

    python tools/plot_wer.py \\
        --input  results/staged-full-stack-dev/wer.csv \\
        --output results/staged-full-stack-dev/wer_curve.png

    python tools/plot_wer.py --self-test

Epoch markers are INTERVAL ESTIMATES.  A checkpoint records the epoch that was
running when it was written, so all the CSV can say is "the epoch changed
somewhere between these two evaluated steps"; the marker is drawn at their
midpoint and is exact only at the evaluated points themselves.  Pass
``--steps-per-epoch N`` to draw the analytic boundaries instead — those come
from ``stage_start + k·N`` (with stage_start = step − step_in_stage) and are
exact.  Epochs are per-stage and restart at 0 at every stage handoff.

Reads the CSV with the standard library and draws with matplotlib — no torch,
no W&B, no repo imports.  Runs anywhere a downloaded CSV lands.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # file output only; never needs a display
import matplotlib.pyplot as plt   # noqa: E402  (must follow the backend choice)


# ── Columns ───────────────────────────────────────────────────────────────────
# Without these there is no curve to draw.
REQUIRED_COLUMNS: tuple[str, ...] = ("step", "split", "format", "wer")
# These only add annotation; a CSV from an older run_wer.py simply gets no
# stage/epoch decoration rather than an error.
OPTIONAL_COLUMNS: tuple[str, ...] = (
    "stage_index", "stage_name", "epoch_in_stage", "step_in_stage", "kind",
)


# ── Style ─────────────────────────────────────────────────────────────────────
# Colour identifies the SPLIT, line style identifies the FORMAT.  Two encodings,
# so the four series stay separable in grayscale and under colour-vision
# deficiency — and so a split keeps its colour whether or not the other splits
# are in the file (colour follows the entity, never its position in the legend).
# Hues are slots 1-4 of the validated reference categorical palette, in order.
SPLIT_COLORS: dict[str, str] = {
    "dev-clean":  "#2a78d6",   # blue
    "dev-other":  "#eb6834",   # orange
    "test-clean": "#1baf7a",   # aqua
    "test-other": "#eda100",   # yellow
}
FALLBACK_COLOR: str = "#52514e"

FORMAT_STYLES: dict[str, tuple[str, str]] = {
    "unformatted": ("-",  "o"),
    "formatted":   ("--", "s"),
}
FALLBACK_STYLE: tuple[str, str] = (":", "^")

TEXT_PRIMARY:   str = "#0b0b0b"
TEXT_SECONDARY: str = "#52514e"
SURFACE:        str = "#fcfcfb"
RULE:           str = "#b8b7b1"


@dataclass(frozen=True)
class Point:
    """One evaluated (checkpoint, split, format) measurement."""
    step:           int
    split:          str
    fmt:            str
    wer:            float
    stage_index:    int | None
    stage_name:     str | None
    epoch_in_stage: int | None
    step_in_stage:  int | None
    kind:           str | None


def _opt_int(row: dict, column: str) -> int | None:
    """Read an optional integer column; None when absent or blank."""
    raw = row.get(column, "")
    return int(raw) if raw not in (None, "") else None


def read_points(path: Path) -> list[Point]:
    """Parse a run_wer.py summary CSV into Points, sorted by step.

    Args:
        path: summary CSV written by tools/run_wer.py

    Returns:
        Every data row, ascending by (step, split, format).

    Raises:
        FileNotFoundError: the CSV does not exist.
        ValueError:        the CSV is empty or lacks a required column.  The
                           message lists the columns that WERE found, because
                           the usual cause is pointing at the wrong file (a
                           degeneracies CSV, say).
    """
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        found  = list(reader.fieldnames or [])
        rows   = list(reader)

    missing = [c for c in REQUIRED_COLUMNS if c not in found]
    if missing:
        raise ValueError(
            f"{path}: missing required column(s) {missing}.  "
            f"Columns found: {found or '(none — file has no header)'}.  "
            "Expected a summary CSV from tools/run_wer.py."
        )
    if not rows:
        raise ValueError(f"{path}: header only, no data rows — nothing to plot.")

    points = [
        Point(
            step           = int(r["step"]),
            split          = r["split"],
            fmt            = r["format"],
            wer            = float(r["wer"]),
            stage_index    = _opt_int(r, "stage_index"),
            stage_name     = r.get("stage_name") or None,
            epoch_in_stage = _opt_int(r, "epoch_in_stage"),
            step_in_stage  = _opt_int(r, "step_in_stage"),
            kind           = r.get("kind") or None,
        )
        for r in rows
    ]
    points.sort(key=lambda p: (p.step, p.split, p.fmt))
    return points


def _series(points: list[Point]) -> dict[tuple[str, str], list[Point]]:
    """Group points into one series per (split, format), each sorted by step."""
    series: dict[tuple[str, str], list[Point]] = {}
    for p in points:
        series.setdefault((p.split, p.fmt), []).append(p)
    for pts in series.values():
        pts.sort(key=lambda p: p.step)
    # Canonical order: splits in SPLIT_COLORS order (unknown splits last),
    # unformatted before formatted — so the legend reads the same every time.
    split_order = list(SPLIT_COLORS)
    fmt_order   = list(FORMAT_STYLES)
    return dict(sorted(
        series.items(),
        key=lambda kv: (
            split_order.index(kv[0][0]) if kv[0][0] in split_order else len(split_order),
            fmt_order.index(kv[0][1])   if kv[0][1] in fmt_order   else len(fmt_order),
            kv[0],
        ),
    ))


def _checkpoints(points: list[Point]) -> list[Point]:
    """One representative Point per evaluated step (metadata is per-checkpoint)."""
    by_step: dict[int, Point] = {}
    for p in points:
        by_step.setdefault(p.step, p)
    return [by_step[s] for s in sorted(by_step)]


def stage_boundaries(checkpoints: list[Point]) -> list[tuple[int, str]]:
    """Find the step of each stage's last evaluated checkpoint.

    A handoff checkpoint IS the end of its stage (training.py writes it once,
    at the stage exit).  When a sweep contains no handoff — a partial CSV, or a
    run still inside its last stage — fall back to the last checkpoint of each
    stage_index, which is the same step whenever the handoff was included.

    Args:
        checkpoints: one Point per evaluated step, ascending

    Returns:
        [(step, stage_name), …] ascending by step; empty when the CSV carries
        no stage metadata at all.
    """
    handoffs = [(c.step, c.stage_name or "") for c in checkpoints if c.kind == "handoff"]
    if handoffs:
        return handoffs

    last_of_stage: dict[int, Point] = {}
    for c in checkpoints:
        if c.stage_index is not None:
            last_of_stage[c.stage_index] = c
    return [
        (last_of_stage[i].step, last_of_stage[i].stage_name or f"stage {i}")
        for i in sorted(last_of_stage)
    ]


def epoch_markers(
    checkpoints: list[Point], steps_per_epoch: int | None = None,
) -> list[tuple[float, str]]:
    """Locate epoch transitions along the step axis.

    Two modes:
      - default (interval estimate): every adjacent pair of checkpoints inside
        the SAME stage whose epoch_in_stage increments contributes one marker at
        the midpoint of the two steps.  The true boundary lies somewhere in that
        interval; the sweep cannot see where.
      - steps_per_epoch given (analytic): boundaries at stage_start + k·N inside
        each stage's observed step range, where stage_start = step − step_in_stage.

    Args:
        checkpoints:     one Point per evaluated step, ascending
        steps_per_epoch: optimizer steps in one epoch, or None for the estimate

    Returns:
        [(x, label), …] with labels like "e0→1"; empty when the metadata needed
        for the chosen mode is absent.
    """
    if steps_per_epoch is not None:
        if steps_per_epoch <= 0:
            raise ValueError(f"--steps-per-epoch must be positive, got {steps_per_epoch}")
        markers: list[tuple[float, str]] = []
        # Group by stage so each stage's epoch counter restarts at 0.
        stages: dict[int, list[Point]] = {}
        for c in checkpoints:
            if c.stage_index is not None and c.step_in_stage is not None:
                stages.setdefault(c.stage_index, []).append(c)
        for stage_pts in stages.values():
            starts = {p.step - p.step_in_stage for p in stage_pts}   # type: ignore[operator]
            stage_start = min(starts)
            last_step   = max(p.step for p in stage_pts)
            k = 1
            while stage_start + k * steps_per_epoch <= last_step:
                markers.append(
                    (float(stage_start + k * steps_per_epoch), f"e{k - 1}→{k}")
                )
                k += 1
        return sorted(markers)

    markers = []
    for prev, cur in zip(checkpoints, checkpoints[1:]):
        if prev.stage_index is None or cur.stage_index is None:
            continue
        if prev.stage_index != cur.stage_index:
            continue   # stage handoff resets the epoch counter; not a transition
        if prev.epoch_in_stage is None or cur.epoch_in_stage is None:
            continue
        if cur.epoch_in_stage > prev.epoch_in_stage:
            markers.append((
                (prev.step + cur.step) / 2.0,
                f"e{prev.epoch_in_stage}→{cur.epoch_in_stage}",
            ))
    return markers


def plot(
    points: list[Point],
    output: Path,
    title: str,
    steps_per_epoch: int | None = None,
) -> None:
    """Render the WER curve with stage and epoch annotations to a PNG.

    Args:
        points:          rows from read_points()
        output:          image path (extension picks the format; .png by default)
        title:           axes title
        steps_per_epoch: analytic epoch boundaries instead of interval estimates
    """
    series      = _series(points)
    checkpoints = _checkpoints(points)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for (split, fmt), pts in series.items():
        color         = SPLIT_COLORS.get(split, FALLBACK_COLOR)
        linestyle, mk = FORMAT_STYLES.get(fmt, FALLBACK_STYLE)
        ax.plot(
            [p.step for p in pts],
            [p.wer * 100.0 for p in pts],
            label     = f"{split} / {fmt}",
            color     = color,
            linestyle = linestyle,
            marker    = mk,
            linewidth = 2.0,
            markersize= 5.5,
            # A 2px surface ring keeps overlapping markers from merging.
            markeredgecolor = SURFACE,
            markeredgewidth = 1.0,
            zorder    = 3,
        )

    ax.set_xlabel("global step", color=TEXT_SECONDARY)
    ax.set_ylabel("WER (%)",     color=TEXT_SECONDARY)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=13, pad=14)
    ax.set_ylim(bottom=0)                       # WER is a rate from zero
    ax.grid(True, axis="y", color=RULE, alpha=0.35, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_SECONDARY)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(RULE)

    # ── Stage boundaries + span labels ────────────────────────────────────────
    boundaries = stage_boundaries(checkpoints)
    if boundaries:
        x_left = min(c.step for c in checkpoints)
        # The first stage starts at the left edge of the data, not at step 0 —
        # a partial sweep may not include the opening checkpoints.
        span_start = x_left
        y_top      = ax.get_ylim()[1]
        for step, name in boundaries:
            ax.axvline(step, color=TEXT_SECONDARY, linewidth=1.2, alpha=0.7, zorder=2)
            if name:
                ax.text(
                    (span_start + step) / 2.0, y_top * 0.97, name,
                    rotation=90, ha="center", va="top",
                    fontsize=9, color=TEXT_SECONDARY, alpha=0.9,
                )
            span_start = step

    # ── Epoch transitions ─────────────────────────────────────────────────────
    for x, label in epoch_markers(checkpoints, steps_per_epoch):
        ax.axvline(x, color=TEXT_SECONDARY, linewidth=0.8, linestyle=":",
                   alpha=0.55, zorder=1)
        ax.text(x, ax.get_ylim()[1] * 0.5, label, rotation=90,
                ha="right", va="center", fontsize=7.5,
                color=TEXT_SECONDARY, alpha=0.8)

    # Legend outside the axes: the curves run the full width, and identity must
    # never be carried by colour alone.
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.01, 1.0),
        frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot WER vs step from a tools/run_wer.py summary CSV.",
    )
    parser.add_argument(
        "--input", type=Path,
        help="Summary CSV written by tools/run_wer.py.",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Image path to write (e.g. results/run/wer_curve.png).",
    )
    parser.add_argument(
        "--title", type=str, default=None,
        help="Axes title (default: derived from the input filename).",
    )
    parser.add_argument(
        "--steps-per-epoch", type=int, default=None, dest="steps_per_epoch",
        metavar="N",
        help="Draw analytic epoch boundaries at stage_start + k*N instead of "
             "midpoint estimates between checkpoints whose epoch changed.",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="Render synthetic CSVs to a temporary directory and exit.",
    )
    args = parser.parse_args(argv)
    if not args.self_test:
        if args.input is None:
            parser.error("--input is required unless --self-test is passed")
        if args.output is None:
            parser.error("--output is required unless --self-test is passed")
    return args


def main(argv: list[str] | None = None) -> None:
    """Read the summary CSV and write the annotated WER curve."""
    args = _parse_args(argv)
    if args.self_test:
        _self_test()
        return

    points = read_points(args.input)
    title  = args.title or f"WER vs step — {args.input.parent.name or args.input.stem}"

    n_series = len({(p.split, p.fmt) for p in points})
    n_steps  = len({p.step for p in points})
    print(f"{len(points)} rows: {n_series} series over {n_steps} checkpoint(s).")

    plot(points, args.output, title, args.steps_per_epoch)
    print(f"Plot → {args.output}")


# ── Self-test ─────────────────────────────────────────────────────────────────

def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    """Write a synthetic summary CSV for the self-test."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _synthetic_rows() -> list[dict]:
    """A four-stage sweep: 3 checkpoints per stage, one epoch increment in stage 1.

    Mirrors configs/staged_full_stack.yaml — 4680 steps per stage, handoffs at
    4680/9360/14040/18720 — and deliberately OMITS one checkpoint (step 6120) so
    the partial-CSV path is exercised.
    """
    stages = ["bridge_only", "audio_adapters_only", "bridge_plus_adapters", "full_stack"]
    rows: list[dict] = []
    for stage_index, stage_name in enumerate(stages):
        stage_start = stage_index * 4680
        for step_in_stage in (1440, 2880, 4680):
            step = stage_start + step_in_stage
            if step == 6120:
                continue                      # missing checkpoint
            # Stage 1 rolls over into its second epoch partway through.
            epoch = 1 if (stage_index == 0 and step_in_stage > 1440) else 0
            kind  = "handoff" if step_in_stage == 4680 else "periodic"
            for split, base in (("dev-clean", 0.30), ("dev-other", 0.45)):
                for fmt, penalty in (("unformatted", 0.0), ("formatted", 0.06)):
                    rows.append({
                        "checkpoint":     f"checkpoints/run/step{step:07d}.pt",
                        "step":           step,
                        "split":          split,
                        "format":         fmt,
                        "wer":            round(base + penalty - 0.02 * (step / 4680), 4),
                        "n_samples":      20,
                        "stage_index":    stage_index,
                        "stage_name":     stage_name,
                        "epoch_in_stage": epoch,
                        "step_in_stage":  step_in_stage,
                        "kind":           kind,
                    })
    return rows


def _self_test() -> None:
    """Render the synthetic sweep, a single-split cut, and a metadata-free CSV."""
    import tempfile

    print("plot_wer.py self-test")
    columns = ["checkpoint", "step", "split", "format", "wer", "n_samples",
               "stage_index", "stage_name", "epoch_in_stage", "step_in_stage", "kind"]
    rows = _synthetic_rows()

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # ── 1. Full four-stage sweep ──────────────────────────────────────────
        full_csv = tmp / "wer.csv"
        _write_csv(full_csv, rows, columns)
        points = read_points(full_csv)
        assert len({p.step for p in points}) == 11, "one checkpoint should be missing"
        assert len(_series(points)) == 4, _series(points).keys()

        checkpoints = _checkpoints(points)
        bounds = stage_boundaries(checkpoints)
        assert [s for s, _ in bounds] == [4680, 9360, 14040, 18720], bounds
        assert [n for _, n in bounds] == [
            "bridge_only", "audio_adapters_only", "bridge_plus_adapters", "full_stack",
        ], bounds
        print("  [OK] 4 stage boundaries found, each with its stage name")

        epochs = epoch_markers(checkpoints)
        assert epochs == [((1440 + 2880) / 2.0, "e0→1")], epochs
        print("  [OK] epoch transition estimated at the midpoint of its interval")

        analytic = epoch_markers(checkpoints, steps_per_epoch=2000)
        # Each stage spans 4680 steps, so a 2000-step epoch gives two interior
        # boundaries per stage, offset by that stage's own start step and with
        # the epoch counter restarting at 0 every stage.
        expected = sorted(
            (float(start + k * 2000), f"e{k - 1}→{k}")
            for start in (0, 4680, 9360, 14040)
            for k in (1, 2)
        )
        assert analytic == expected, analytic
        print("  [OK] --steps-per-epoch draws analytic boundaries per stage")

        out_png = tmp / "out" / "wer_curve.png"
        main(argv=["--input", str(full_csv), "--output", str(out_png)])
        assert out_png.exists() and out_png.stat().st_size > 5000, out_png.stat().st_size
        print(f"  [OK] PNG written ({out_png.stat().st_size:,} bytes)")

        # ── 2. Single split, single format ────────────────────────────────────
        one_csv = tmp / "one.csv"
        _write_csv(
            one_csv,
            [r for r in rows if r["split"] == "dev-other" and r["format"] == "formatted"],
            columns,
        )
        one_png = tmp / "out" / "one.png"
        main(argv=["--input", str(one_csv), "--output", str(one_png),
                   "--title", "single series"])
        assert one_png.exists() and one_png.stat().st_size > 5000
        assert len(_series(read_points(one_csv))) == 1
        print("  [OK] single-split, single-format CSV renders")

        # ── 3. No stage/epoch metadata (older run_wer.py output) ──────────────
        bare_csv = tmp / "bare.csv"
        bare_cols = ["checkpoint", "step", "split", "format", "wer", "n_samples"]
        _write_csv(bare_csv, [{c: r[c] for c in bare_cols} for r in rows], bare_cols)
        bare_points = read_points(bare_csv)
        assert stage_boundaries(_checkpoints(bare_points)) == []
        assert epoch_markers(_checkpoints(bare_points)) == []
        bare_png = tmp / "out" / "bare.png"
        main(argv=["--input", str(bare_csv), "--output", str(bare_png)])
        assert bare_png.exists() and bare_png.stat().st_size > 5000
        print("  [OK] metadata-free CSV plots without stage/epoch annotation")

        # ── 4. Loud failures ──────────────────────────────────────────────────
        empty_csv = tmp / "empty.csv"
        _write_csv(empty_csv, [], columns)
        try:
            read_points(empty_csv)
            raise AssertionError("empty CSV must raise")
        except ValueError as exc:
            assert "no data rows" in str(exc), exc

        wrong_csv = tmp / "wrong.csv"
        _write_csv(wrong_csv, [{"step": 1, "loop": 0}], ["step", "loop"])
        try:
            read_points(wrong_csv)
            raise AssertionError("CSV without required columns must raise")
        except ValueError as exc:
            assert "missing required column" in str(exc), exc
            assert "loop" in str(exc), "the message must list the columns found"
        print("  [OK] empty CSV and wrong-columns CSV both fail loudly")

    print("\nPASSED")


if __name__ == "__main__":
    main(argv=sys.argv[1:])
