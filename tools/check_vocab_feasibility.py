"""Can the model even PRODUCE what a probe asks for, under the pruned vocabulary?

THE GATE, AND WHY IT RUNS FIRST
-------------------------------
Decision 005 prunes the Llama tokenizer to the ids appearing in the LibriSpeech
labels.  Anything outside that set has no logit row: the model cannot emit it,
however well it hears and however well it follows instructions.  So a probe
whose expected output falls outside the pruned vocabulary is impossible BY
CONSTRUCTION, and scoring the model on it would report a property of
tools/build_vocab.py as a failure of instruction-following.

The prompt side is worse than the output side.  An unreachable OUTPUT at least
produces a visibly wrong string.  An unreachable word in an INSTRUCTION raises
nothing at all — PrunedTokenizer.encode drops ids it has no mapping for — so the
model receives the instruction with words deleted and the resulting "failure to
follow" is the harness's fault.  Both are measured here, and prompts must be
100% reachable.

This is tools/check_vocab_coverage.py's question asked of candidate outputs
rather than of a corpus of references, and it reuses that tool's FullTokenizer
(the unpruned tokenizer plus the surviving-id set) rather than reimplementing it.

WHAT IT WRITES
--------------
A gate JSON that tools/run_probes.py reads and refuses to run against:

    {"probes": {"P1": {"feasible": false, "reason": "..."}, ...}}

A probe is FEASIBLE when every candidate group gating it clears its threshold,
INFEASIBLE when one does not, and UNDECIDED when a gating group could not be
built (P1 without --shard).  Undecided is not feasible: run_probes.py will not
run a probe this tool never cleared.

USAGE
-----
    python tools/check_vocab_feasibility.py \\
        --tokenizer data/pruned_tokenizer_chat/ \\
        --shard data/eval_subset_shards/dev-clean-480.tar \\
        --out-json out/probe-feasibility.json \\
        --out-md   out/probe-feasibility.md

    # arbitrary strings, no probe suite
    python tools/check_vocab_feasibility.py --strings "HELLO THERE" "5" --quiet

    python tools/check_vocab_feasibility.py --self-test
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TOOLS_DIR = Path(__file__).resolve().parent
for _p in (str(_REPO_ROOT), str(_TOOLS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import probe_spec  # noqa: E402  (stdlib-only, same tools/ dir)
from probe_spec import CandidateGroup, PROBE_IDS  # noqa: E402


@dataclass
class StringResult:
    """Reachability of one candidate string."""

    text:      str
    n_tokens:  int
    reachable: bool
    # (original id, decoded piece) for each token the pruned vocabulary lacks.
    broken:    list[tuple[int, str]]

    def to_dict(self) -> dict:
        return {
            "text":      self.text,
            "n_tokens":  self.n_tokens,
            "reachable": self.reachable,
            "unreachable_tokens": [
                {"id": i, "piece": piece} for i, piece in self.broken
            ],
        }


def check_string(text: str, tok) -> StringResult:
    """Tokenize with the FULL tokenizer and test every id against the pruned set.

    Args:
        text: candidate string
        tok:  FullTokenizer (or the self-test's stub) exposing encode /
              kept_ids / decode_ids

    Returns:
        StringResult naming every token that breaks it
    """
    ids    = tok.encode(text)
    broken = [(i, tok.decode_ids([i])) for i in ids if i not in tok.kept_ids]
    return StringResult(text, len(ids), not broken, broken)


def check_group(group: CandidateGroup, tok) -> dict:
    """Measure one candidate group and decide whether it clears its threshold.

    Returns:
        a report dict; the per-string results ride along under "results"
    """
    results = [check_string(s, tok) for s in group.strings]
    n_ok    = sum(r.reachable for r in results)
    pct     = 100.0 * n_ok / len(results) if results else 0.0
    return {
        "name":      group.name,
        "kind":      group.kind,
        "probes":    list(group.probes),
        "note":      group.note,
        "n":         len(results),
        "n_reachable": n_ok,
        "reachable_pct": round(pct, 2),
        "min_pct":   group.min_pct,
        "passed":    pct >= group.min_pct,
        "results":   [r.to_dict() for r in results],
    }


def decide_probes(group_reports: list[dict]) -> dict[str, dict]:
    """Turn group verdicts into a per-probe feasibility gate.

    A probe needs EVERY group that gates it to have been measured and passed.
    A group that was never built (P1's uppercase set, with no --shard) leaves its
    probes UNDECIDED — which run_probes.py treats as "do not run", because a
    probe this tool never cleared is exactly the case the gate exists to catch.

    Returns:
        probe id → {"feasible": bool, "status": str, "reason": str,
                    "groups": [...]}
    """
    gate: dict[str, dict] = {}
    measured = {g["name"]: g for g in group_reports}

    for pid in PROBE_IDS:
        gating = [g for g in group_reports if pid in g["probes"]]
        # Which groups SHOULD gate this probe, including ones that were not built?
        expected = {
            g.name for g in probe_spec.build_groups(("PLACEHOLDER",))
            if pid in g.probes
        }
        missing = sorted(expected - set(measured))

        if missing:
            gate[pid] = {
                "feasible": False, "status": "undecided",
                "reason": f"candidate group(s) {missing} were not measured "
                          "(pass --shard to build the uppercase set)",
                "groups": [g["name"] for g in gating],
            }
            continue

        failed = [g for g in gating if not g["passed"]]
        if failed:
            detail = "; ".join(
                f"{g['name']} {g['reachable_pct']:.1f}% reachable "
                f"(needs {g['min_pct']:.0f}%)" for g in failed
            )
            gate[pid] = {
                "feasible": False, "status": "infeasible", "reason": detail,
                "groups": [g["name"] for g in gating],
            }
        else:
            gate[pid] = {
                "feasible": True, "status": "feasible",
                "reason": "every gating candidate group is reachable",
                "groups": [g["name"] for g in gating],
            }
    return gate


def uppercase_samples(shard: Path, n: int, seed: int) -> tuple[str, ...]:
    """Upper-cased real references — P1's expected outputs.

    Real references rather than invented strings: what P1 demands is THIS
    corpus's sentences in capitals, and an invented sample would not exercise the
    same subwords.

    Args:
        shard: eval shard .tar in the house three-member format
        n:     how many references to draw
        seed:  RNG seed; selection is over key-sorted samples so it is reproducible

    Returns:
        tuple of upper-cased verbatim references
    """
    from data import read_eval_shard

    samples = sorted(read_eval_shard(shard), key=lambda s: s.key)
    if not samples:
        raise ValueError(f"{shard} yielded no samples")
    rng  = random.Random(seed)
    take = rng.sample(samples, min(n, len(samples)))
    return tuple(s.ref_unformatted.upper() for s in take)


def render_md(report: dict) -> str:
    """Paste-ready Markdown: the group table, the gate, and the offenders."""
    out = [
        "### Probe feasibility under the pruned vocabulary",
        "",
        f"Vocabulary: `{report['tokenizer']}` ({report['vocab_size']:,} tokens).",
        "",
        "A probe whose expected output cannot be tokenized inside this vocabulary "
        "is impossible by construction — reporting it as a model failure would be "
        "wrong. Prompt strings must be **100%** reachable: "
        "`PrunedTokenizer.encode` drops unmapped ids silently, so an unreachable "
        "word in an instruction reaches the model as a hole in the prompt.",
        "",
        "| candidate group | kind | gates | n | fully reachable | required | verdict |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for g in report["groups"]:
        out.append(
            f"| {g['name']} | {g['kind']} | {', '.join(g['probes'])} | {g['n']:,} | "
            f"{g['reachable_pct']:.1f}% | {g['min_pct']:.0f}% | "
            f"{'PASS' if g['passed'] else '**FAIL**'} |"
        )
    out += ["", "**Gate**", "",
            "| probe | status | reason |", "|---|---|---|"]
    for pid, v in report["probes"].items():
        out.append(f"| {pid} | {v['status']} | {v['reason']} |")

    for g in report["groups"]:
        bad = [r for r in g["results"] if not r["reachable"]]
        if not bad:
            continue
        out += ["", f"**{g['name']}** — {len(bad)} of {g['n']} unreachable. Examples:", ""]
        for r in bad[:5]:
            pieces = ", ".join(f"`{t['piece']}`" for t in r["unreachable_tokens"][:8])
            out.append(f"- `{r['text'][:90]}` — broken by {pieces}")
    return "\n".join(out) + "\n"


def _self_test() -> None:
    """Gate logic against a stub tokenizer — no real vocabulary needed."""

    class StubTok:
        """Character-id tokenizer; upper-case letters are 'pruned'."""

        def __init__(self) -> None:
            self.kept_ids = frozenset(ord(c) for c in "abcdefghijklmnopqrstuvwxyz 0123456789.\n?:,")

        def encode(self, text: str) -> list[int]:
            return [ord(c) for c in text]

        def decode_ids(self, ids: list[int]) -> str:
            return "".join(chr(i) for i in ids)

    tok = StubTok()

    r = check_string("hello there", tok)
    assert r.reachable and not r.broken, r
    r = check_string("HELLO", tok)
    assert not r.reachable and len(r.broken) == 5, r
    assert r.broken[0][1] == "H", r.broken
    print("  [OK] check_string names the tokens that break a string")

    g = CandidateGroup("upper", "output", ("P1",), ("HELLO", "there"), 95.0, "")
    rep = check_group(g, tok)
    assert rep["n"] == 2 and rep["n_reachable"] == 1 and not rep["passed"], rep
    g2 = CandidateGroup("digits", "output", ("P2",), ("7", "\n7"), 100.0, "")
    rep2 = check_group(g2, tok)
    assert rep2["passed"] and rep2["reachable_pct"] == 100.0, rep2
    print("  [OK] check_group thresholds")

    # Full gate over the real group set, with the stub tokenizer.
    groups = probe_spec.build_groups(("HELLO THERE", "GOOD DAY"))
    reports = [check_group(gr, tok) for gr in groups]
    gate = decide_probes(reports)
    assert gate["P1"]["status"] == "infeasible", gate["P1"]
    assert gate["P2"]["status"] == "infeasible", gate["P2"]   # prompts are cased
    print("  [OK] decide_probes marks a probe infeasible when a gating group fails")

    # Uppercase group absent → P1 undecided, and undecided is NOT feasible.
    groups_no_p1 = probe_spec.build_groups(())
    gate2 = decide_probes([check_group(gr, tok) for gr in groups_no_p1])
    assert gate2["P1"]["status"] == "undecided" and not gate2["P1"]["feasible"], gate2["P1"]
    print("  [OK] a probe whose gating group was never measured stays undecided")

    md = render_md({
        "tokenizer": "stub/", "vocab_size": 26,
        "groups": reports, "probes": gate,
    })
    assert "HELLO" in md and "P1" in md
    print("  [OK] render_md")
    print("PASSED")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tokenizer", type=Path, default=Path("data/pruned_tokenizer_chat/"),
                   help="Pruned tokenizer directory to judge (default: the chat one).")
    p.add_argument("--shard", type=Path, default=None,
                   help="Eval shard .tar to draw P1's upper-cased references from. "
                        "Without it P1 stays UNDECIDED and will not run.")
    p.add_argument("--n-uppercase", type=int, default=50, dest="n_uppercase",
                   help="References to upper-case for the P1 group (default 50).")
    p.add_argument("--seed", type=int, default=13,
                   help="Seed for the uppercase sample (default 13).")
    p.add_argument("--strings", nargs="+", default=None, metavar="TEXT",
                   help="Extra arbitrary candidate strings, checked as their own "
                        "group (gates no probe).")
    p.add_argument("--strings-file", type=Path, default=None, dest="strings_file",
                   help="File of extra candidates, one per line.")
    p.add_argument("--out-json", type=Path, default=None, dest="out_json",
                   help="Gate file tools/run_probes.py reads.")
    p.add_argument("--out-md", type=Path, default=None, dest="out_md")
    p.add_argument("--quiet", action="store_true", help="Only print the verdicts.")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)

    if args.self_test:
        _self_test()
        return 0

    from check_vocab_coverage import FullTokenizer

    tok = FullTokenizer(args.tokenizer)
    vocab_size = int(tok.pruned_config["vocab_size"])
    print(f"Vocabulary: {args.tokenizer}  ({vocab_size:,} pruned tokens)\n")

    upper: tuple[str, ...] = ()
    if args.shard is not None:
        upper = uppercase_samples(args.shard, args.n_uppercase, args.seed)
        print(f"P1 candidates: {len(upper)} references from {args.shard.name}, "
              "upper-cased\n")
    else:
        print("[warn] no --shard: the uppercase group cannot be built, so P1 will "
              "be reported UNDECIDED and run_probes.py will refuse to run it.\n")

    groups = probe_spec.build_groups(upper)

    extra: list[str] = list(args.strings or [])
    if args.strings_file:
        extra += [ln.strip() for ln in args.strings_file.read_text().splitlines() if ln.strip()]
    if extra:
        groups.append(CandidateGroup(
            "extra_strings", "output", (), tuple(extra), 100.0,
            "Ad-hoc candidates passed on the command line.",
        ))

    reports = [check_group(g, tok) for g in groups]
    gate    = decide_probes(reports)

    for g in reports:
        flag = "PASS" if g["passed"] else "FAIL"
        print(f"{g['name']:<24} {g['kind']:<7} {g['n_reachable']:>5}/{g['n']:<5} "
              f"{g['reachable_pct']:>6.1f}%  (needs {g['min_pct']:.0f}%)  {flag}")
        if not args.quiet:
            for r in (x for x in g["results"] if not x["reachable"]):
                pieces = ", ".join(repr(t["piece"]) for t in r["unreachable_tokens"][:8])
                print(f"{'':24} ✗ {r['text'][:70]!r}  broken by {pieces}")

    print("\n── Gate ──")
    for pid, v in gate.items():
        print(f"  {pid}  {v['status']:<11} {v['reason']}")

    report = {
        "tokenizer":  str(args.tokenizer),
        "vocab_size": vocab_size,
        "shard":      str(args.shard) if args.shard else None,
        "groups":     reports,
        "probes":     gate,
    }
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out_json}")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(render_md(report))
        print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
