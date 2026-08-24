"""Compare two run_wer transcription JSONLs sample-by-sample, joined on key.

Greedy decoding is deterministic, so an unchanged model on unchanged code must
reproduce the same hypothesis for the same utterance.  This is a far sharper
regression signal than an aggregate WER, which can hide compensating errors.

Usage:
    python compare_hyps.py BANKED.jsonl NEW.jsonl
"""
import json, sys
from collections import defaultdict

def load(path):
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[(r["key"], r["type"])] = r
    return out

banked, new = load(sys.argv[1]), load(sys.argv[2])
shared = sorted(set(banked) & set(new))
if not shared:
    sys.exit(f"FAIL: no overlapping (key, type) pairs — banked={len(banked)} new={len(new)}")

diffs = [k for k in shared if banked[k]["hypothesis"] != new[k]["hypothesis"]]
per_fmt = defaultdict(lambda: [0, 0])
for k in shared:
    per_fmt[k[1]][0] += 1
for k in diffs:
    per_fmt[k[1]][1] += 1

print(f"compared {len(shared)} (key, format) pairs present in both files")
for fmt, (n, d) in sorted(per_fmt.items()):
    print(f"  {fmt:12s} {n:4d} compared   {d:4d} differ ({d/n:.1%})")

for k in diffs[:5]:
    print(f"\n  key={k[0]}  [{k[1]}]")
    print(f"    banked: {banked[k]['hypothesis'][:100]}")
    print(f"    new   : {new[k]['hypothesis'][:100]}")

print()
if not diffs:
    print("PASS — every hypothesis is identical; the flat path is bit-for-bit intact.")
elif len(diffs) / len(shared) < 0.01:
    print(f"PASS (with noise) — {len(diffs)}/{len(shared)} differ (<1%), consistent with "
          "fp16 non-determinism rather than a code change. Eyeball the samples above.")
else:
    print(f"FAIL — {len(diffs)}/{len(shared)} hypotheses changed. The flat path is NOT intact.")
    sys.exit(1)
