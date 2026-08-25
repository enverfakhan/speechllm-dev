"""Zero-shot long-form transcription over chat turns.

The model is trained SINGLE-TURN: one user turn carrying one utterance's audio,
one assistant turn carrying its transcript.  This harness asks what happens when
audio instead arrives as SUCCESSIVE user turns and the transcript continues
across assistant turns — a capability nothing in training put there.  Eval only:
no training code is touched and no checkpoint is written.

Conversations are built from LibriSpeech utterance ids, which encode reading
order (``speaker-chapter-utterance``).  Utterances are grouped by
(speaker, chapter), sorted by utterance index, and cut into runs of consecutive
utterances; a run with a gap is dropped rather than silently stitched, because a
gap means the conversation would jump forward in the book mid-context.

Three context modes are decoded over the SAME conversations, and all three come
from this one run — the point of the exercise is a matched comparison, so
reusing a number from a different sweep would defeat it:

  self     each assistant turn is the model's own generated transcript.  Errors
           can propagate.  The realistic condition.
  oracle   each previous assistant turn is replaced by the gold transcript
           before the next turn generates.  Teacher-forced context: isolates
           whether context HELPS, independent of error accumulation.
  single   each utterance decoded on its own, exactly as tools/run_wer.py does
           it, through the untouched single-turn path.  The control.

Usage:
    # Sanity: stub model, local dev corpus, 2 conversations x 3 turns
    python tools/run_longform.py --config configs/stub_longform.yaml \\
        --shard dev-1183=data_dev/dev_shards/dev-diag-1183-000000.tar \\
        --turns 3 --max-convs 2 --print-layout 1 --out-dir out/longform-stub

    # Real checkpoint, one split
    python tools/run_longform.py --config configs/instruct-chat-3stage.yaml \\
        --checkpoint checkpoints/instruct-chat-3stage/step0015040.pt \\
        --shard dev-clean=data/eval_shards/dev-clean-000000.tar \\
        --turns 6 --max-convs 100 --out-dir out

    # Plumbing self-test (synthetic keys; no data, no model, no GPU)
    python tools/run_longform.py --self-test

Output: one JSONL per split, ``longform-<split>.jsonl``, with run_wer.py's field
names (checkpoint, step, key, split, type, reference, hypothesis, wer) plus
``conv_id``, ``turn_index`` (1-based) and ``context_mode``.  Feed it straight to
the existing analysis tools:

    python tools/analyze_slices.py --jsonl out/longform-dev-clean.jsonl \\
        --group-by turn_index --out-md out/longform-dev-clean-by-turn.md
    python tools/analyze_slices.py --jsonl out/longform-dev-clean.jsonl \\
        --group-by context_mode --out-md out/longform-dev-clean-by-ctx.md

NOTE ON COMPARABILITY.  Batches here are formed by CONVERSATION, not by
ascending audio length as build_sorted_eval_dataloader forms them.  Mel padding
is per batch and the Whisper encoder attends bidirectionally across it, so batch
composition perturbs the audio embeddings slightly — which is exactly why the
``single`` rows are decoded here, in the same batches, rather than lifted from a
run_wer.py sweep.  Compare within this file's output; compare to a WER sweep only
at the level of "same order of magnitude".
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build import build_models
from data import (
    EvalSample,
    INSTRUCTION_VARIANTS,
    PrunedTokenizer,
    load_pruned_config,
    read_eval_shard,
    # Private on purpose, and reached into on purpose: using the WER path's own
    # collation is what guarantees this tool pads mel exactly as it does.  The
    # Whisper encoder attends across the padding, so a second implementation
    # that padded differently would silently change the audio embeddings.
    _eval_collate_batch,
)
from model.sequence import ChatTemplate, ConversationPrefix, ConversationPrefixBatch
from utils.checkpoint import apply_weights, read_checkpoint
from utils.config import load_config
from utils.evaluate import compute_wer
from utils.generate import greedy_generate, greedy_generate_from_prefix

# LibriSpeech utterance id: speaker-chapter-utterance.  The third field is the
# reading order within the chapter, which is what makes a conversation possible.
_UTT_RE = re.compile(r"^(\d+)-(\d+)-(\d+)$")

CONTEXT_MODES: tuple[str, ...] = ("self", "oracle", "single")
FORMATS:       tuple[str, ...] = ("unformatted", "formatted")


# ── Conversation construction ─────────────────────────────────────────────────

@dataclass(frozen=True)
class Conversation:
    """A run of consecutive utterances from one (speaker, chapter)."""

    conv_id:    str
    split:      str
    speaker:    int
    chapter:    int
    utterances: list[EvalSample]


def reference_for(sample: EvalSample, fmt: str) -> str:
    """The gold transcript for one instruction variant."""
    return sample.ref_unformatted if fmt == "unformatted" else sample.ref_formatted


def build_conversations(
    samples:            list[EvalSample],
    split:              str,
    tokenizer:          PrunedTokenizer,
    turns:              int,
    max_convs:          int,
    max_ref_tokens:     int | None,
    max_runs_per_group: int,
) -> tuple[list[Conversation], dict]:
    """Group utterances into consecutive runs and select the first ``max_convs``.

    Selection is fully deterministic — no RNG anywhere.  Groups are visited in
    (speaker, chapter) order and their runs in reading order, and the first
    ``max_convs`` are taken.

    Order of operations matters and is deliberate: the per-utterance
    ``max_ref_tokens`` filter runs FIRST, so per-utterance WER stays comparable
    with the single-turn sweeps (whose eval subsets were built with the same
    ≤41-token filter).  Filtering punches holes in the reading order, and the
    consecutiveness requirement then drops any run that spans such a hole.

    Args:
        samples:            everything read out of the shard
        split:              split name, recorded on every row
        tokenizer:          for the reference-length filter
        turns:              utterances per conversation
        max_convs:          conversations to select for this split
        max_ref_tokens:     drop an utterance if EITHER label variant exceeds
                            this many tokens; None disables the filter
        max_runs_per_group: cap runs taken from one (speaker, chapter);
                            0 = unlimited, which is the literal "first N runs"
                            selection.  Raising the cap off 0 trades that
                            literalness for speaker breadth.

    Returns:
        (conversations, stats) — stats carries every count worth logging
    """
    stats: dict = {
        "n_samples":       len(samples),
        "n_unparsed_keys": 0,
        "n_too_long":      0,
        "n_eligible":      0,
        "n_groups":        0,
        "candidate_runs":  0,
        "accepted_runs":   0,
        "dropped_runs":    0,
        "selected":        0,
    }

    eligible: list[tuple[int, int, int, EvalSample]] = []
    for s in samples:
        m = _UTT_RE.match(s.key)
        if m is None:
            stats["n_unparsed_keys"] += 1
            continue
        if max_ref_tokens is not None:
            n_tok = max(
                len(tokenizer.encode(s.ref_unformatted)),
                len(tokenizer.encode(s.ref_formatted)),
            )
            if n_tok > max_ref_tokens:
                stats["n_too_long"] += 1
                continue
        eligible.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), s))
    stats["n_eligible"] = len(eligible)

    groups: dict[tuple[int, int], list[tuple[int, EvalSample]]] = defaultdict(list)
    for spk, chap, idx, s in eligible:
        groups[(spk, chap)].append((idx, s))
    stats["n_groups"] = len(groups)

    conversations: list[Conversation] = []
    for (spk, chap) in sorted(groups):
        items = sorted(groups[(spk, chap)], key=lambda p: p[0])
        # How many runs this group COULD have yielded had the filter left no
        # holes — the denominator for the drop count.
        stats["candidate_runs"] += len(items) // turns

        taken_here = 0
        # Cut the group into maximal contiguous segments, then chunk each into
        # non-overlapping runs.  A run that would span a gap simply never forms.
        segments: list[list[tuple[int, EvalSample]]] = []
        current: list[tuple[int, EvalSample]] = []
        for idx, s in items:
            if current and idx != current[-1][0] + 1:
                segments.append(current)
                current = []
            current.append((idx, s))
        if current:
            segments.append(current)

        for seg in segments:
            for start in range(0, len(seg) - turns + 1, turns):
                if max_runs_per_group and taken_here >= max_runs_per_group:
                    break
                run = seg[start : start + turns]
                stats["accepted_runs"] += 1
                taken_here += 1
                conversations.append(Conversation(
                    conv_id    = f"{split}:{spk}-{chap}:{run[0][0]:04d}",
                    split      = split,
                    speaker    = spk,
                    chapter    = chap,
                    utterances = [s for _, s in run],
                ))

    stats["dropped_runs"] = max(0, stats["candidate_runs"] - stats["accepted_runs"])
    selected = conversations[:max_convs] if max_convs > 0 else conversations
    stats["selected"] = len(selected)
    return selected, stats


# ── Sequence-length guard ─────────────────────────────────────────────────────

def worst_case_length(
    conv:           Conversation,
    chat:           ChatTemplate,
    n_instruction:  int,
    tokenizer:      PrunedTokenizer,
    fmt:            str,
    slack:          int,
) -> int:
    """Longest sequence this conversation can reach, in tokens.

    Every assistant turn is budgeted at its own generation cap (reference length
    + ``slack``), which is the hard ceiling greedy decoding can hit, so the
    number is a true upper bound and not an estimate.
    """
    total = 0
    for t, s in enumerate(conv.utterances):
        opener = chat.seg_pre_audio if t == 0 else chat.seg_user_continuation
        total += (
            len(opener) + 1 + s.audio_length + 1
            + len(chat.seg_pre_instruction) + n_instruction
            + len(chat.seg_pre_transcript)
        )
        # Every turn contributes an assistant span: the earlier ones as context,
        # the last one as what it is still generating.  Both are bounded by the
        # same cap, so the total is a true upper bound.
        total += len(tokenizer.encode(reference_for(s, fmt))) + slack
    return total


# ── Decoding ──────────────────────────────────────────────────────────────────

def _turn_batch(samples: list[EvalSample], instr_ids: list[int]) -> tuple:
    """Collate one turn's utterances through the WER path's own collation."""
    return _eval_collate_batch([
        (s.mel, instr_ids, instr_ids, s.ref_unformatted, s.ref_formatted, s.key)
        for s in samples
    ])


def _generation_cap(refs: list[str], tokenizer: PrunedTokenizer, slack: int) -> int:
    """Per-batch generation cap: longest reference in tokens + slack.

    Same rule utils/evaluate.py applies, so a runaway sample cannot stall the
    run.  Applied identically to all three context modes, which is what keeps
    the ``single`` control matched to the multi-turn conditions.
    """
    return max(len(tokenizer.encode(r)) for r in refs) + slack


@torch.no_grad()
def decode_multi_turn(
    convs:        list[Conversation],
    fmt:          str,
    context_mode: str,
    *,
    encoder, adapter, llama,
    tokenizer:     PrunedTokenizer,
    chat:          ChatTemplate,
    terminator_id: int,
    instr_ids:     list[int],
    device:        torch.device,
    slack:         int,
) -> list[list[str]]:
    """Decode a chunk of conversations turn by turn, batched across conversations.

    All conversations generate turn 1 together, then turn 2, and so on — the only
    way to batch when each conversation's prefix has its own length and grows by
    its own amount.

    Args:
        context_mode: "self" (feed back the generated transcript) or "oracle"
                      (feed back the gold one)

    Returns:
        hyps[conv_index][turn_index] — decoded text
    """
    if context_mode not in ("self", "oracle"):
        raise ValueError(f"decode_multi_turn does not handle context_mode={context_mode!r}")

    n_turns   = len(convs[0].utterances)
    instr_t   = torch.tensor(instr_ids, dtype=torch.long, device=device)
    prefixes  = [
        ConversationPrefix(chat, llama.embed_tokens, adapter.audio_bos, adapter.audio_eos)
        for _ in convs
    ]
    hyps: list[list[str]] = [[] for _ in convs]

    for t in range(n_turns):
        samples = [c.utterances[t] for c in convs]
        batch   = _turn_batch(samples, instr_ids)
        mel           = batch[0].to(device)
        audio_lengths = batch[1].to(device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            adapter_out = adapter(encoder(mel))

        for i, cp in enumerate(prefixes):
            cp.add_user_turn(adapter_out[i, : int(audio_lengths[i].item()), :], instr_t)

        pfx = ConversationPrefixBatch(
            [cp.embeddings  for cp in prefixes],
            [cp.audio_flags for cp in prefixes],
        )
        refs = [reference_for(s, fmt) for s in samples]
        gen  = greedy_generate_from_prefix(
            llama, pfx, terminator_id, _generation_cap(refs, tokenizer, slack),
        )

        for i, ids in enumerate(gen):
            hyps[i].append(tokenizer.decode(ids))
            if t + 1 < n_turns:
                # `self` propagates whatever the model produced, errors and all —
                # that is the condition being measured.  `oracle` replaces it with
                # the gold transcript so context quality is held fixed.
                ctx_ids = ids if context_mode == "self" else tokenizer.encode(refs[i])
                prefixes[i].add_assistant_text(ctx_ids)

    return hyps


@torch.no_grad()
def decode_single_turn(
    convs: list[Conversation],
    fmt:   str,
    *,
    encoder, adapter, llama,
    tokenizer:     PrunedTokenizer,
    chat:          ChatTemplate,
    terminator_id: int,
    instr_ids:     list[int],
    device:        torch.device,
    slack:         int,
) -> list[list[str]]:
    """The matched control: every utterance decoded alone, single-turn.

    Runs through the untouched greedy_generate / EvalPrefixBatch path, and is
    batched turn-index by turn-index over the same conversation chunk, so each
    utterance sits in exactly the same mel batch it sits in under `self` and
    `oracle` and meets exactly the same generation cap.
    """
    n_turns = len(convs[0].utterances)
    hyps: list[list[str]] = [[] for _ in convs]

    for t in range(n_turns):
        samples = [c.utterances[t] for c in convs]
        batch   = _turn_batch(samples, instr_ids)
        mel           = batch[0].to(device)
        audio_lengths = batch[1].to(device)
        ids_t         = batch[2].to(device)   # both slots hold the same instruction
        lens_t        = batch[3].to(device)

        refs = [reference_for(s, fmt) for s in samples]
        gen  = greedy_generate(
            encoder, adapter, llama,
            mel, audio_lengths, ids_t, lens_t,
            stop_token_id  = terminator_id,
            max_new_tokens = _generation_cap(refs, tokenizer, slack),
            chat           = chat,
        )
        for i, hyp_ids in enumerate(gen):
            hyps[i].append(tokenizer.decode(hyp_ids))

    return hyps


# ── Output ────────────────────────────────────────────────────────────────────

def _sample_wer(reference: str, hypothesis: str) -> float:
    """WER of one pair, through the aggregate's own jiwer path, un-normalized.

    Mirrors tools/run_wer.py so a per-sample number from either file means the
    same thing.  An empty reference makes WER undefined, hence NaN.
    """
    if not reference.strip():
        return float("nan")
    return compute_wer([reference], [hypothesis])


def make_rows(
    conv:         Conversation,
    fmt:          str,
    context_mode: str,
    hyps:         list[str],
    checkpoint:   str,
    step:         int,
    dataset:      str | None,
) -> list[dict]:
    """One JSONL row per turn, in run_wer.py's field order plus the three new fields."""
    rows: list[dict] = []
    for t, (s, hyp) in enumerate(zip(conv.utterances, hyps)):
        ref = reference_for(s, fmt)
        rows.append({
            "checkpoint":   checkpoint,
            "step":         step,
            **({"dataset": dataset} if dataset else {}),
            "key":          s.key,
            "split":        conv.split,
            "type":         fmt,
            "reference":    ref,
            "hypothesis":   hyp,
            "wer":          _sample_wer(ref, hyp),
            "conv_id":      conv.conv_id,
            "turn_index":   t + 1,
            "context_mode": context_mode,
        })
    return rows


def print_summary(rows: list[dict], split: str) -> None:
    """Corpus-level WER by (context mode, instruction mode, turn index).

    The headline deliverable: whether WER climbs with turn index, and whether
    `oracle` climbing less than `self` says error propagation is the cause.
    """
    by: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for r in rows:
        by[(r["context_mode"], r["type"], r["turn_index"])].append(r)

    turns = sorted({k[2] for k in by})
    print(f"\n── {split}: WER by turn index (corpus-level, un-normalized) ──")
    header = f"{'context':<8} {'format':<12} " + " ".join(f"{'t' + str(t):>7}" for t in turns) + f" {'all':>8}"
    print(header)
    print("-" * len(header))
    for ctx in CONTEXT_MODES:
        for fmt in FORMATS:
            cells, all_rows = [], []
            for t in turns:
                rs = by.get((ctx, fmt, t), [])
                all_rows.extend(rs)
                cells.append(
                    f"{compute_wer([r['reference'] for r in rs], [r['hypothesis'] for r in rs]):>6.1%}"
                    if rs else f"{'—':>7}"
                )
            if not all_rows:
                continue
            overall = compute_wer(
                [r["reference"] for r in all_rows], [r["hypothesis"] for r in all_rows]
            )
            print(f"{ctx:<8} {fmt:<12} " + " ".join(cells) + f" {overall:>7.1%}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_step(stem: str, fallback: int = 0) -> int:
    """Extract the optimizer step from a filename stem (e.g. 'step0015040')."""
    m = re.search(r"step[_-]?(\d+)", stem, re.IGNORECASE)
    return int(m.group(1)) if m else fallback


def _parse_shards(specs: list[str]) -> list[tuple[str, Path]]:
    """Parse --shard NAME=PATH arguments, exactly as run_wer.py parses --eval-tar."""
    pairs: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for spec in specs:
        name, sep, raw = spec.partition("=")
        if not sep or not name or not raw:
            raise ValueError(
                f"--shard expects NAME=PATH, got {spec!r} "
                "(e.g. dev-clean=data/eval_shards/dev-clean-000000.tar)"
            )
        if name in seen:
            raise ValueError(f"--shard name {name!r} given twice")
        path = Path(raw)
        if not path.exists():
            raise ValueError(f"--shard {name}: {path} does not exist")
        seen.add(name)
        pairs.append((name, path))
    return pairs


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Zero-shot multi-turn (long-form) transcription eval.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=Path, required=True,
                   help="Training config YAML — supplies model dims, tokenizer and "
                        "input_convention.  Must be a chat-convention config.")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Checkpoint to evaluate.  Omitting it evaluates the "
                        "pretrained/stub base and is only useful for plumbing tests.")
    p.add_argument("--shard", nargs="+", dest="shards", default=None, metavar="NAME=PATH",
                   help="Eval shard(s) to build conversations from.  Point these at the "
                        "FULL split shards (data/eval_shards/dev-clean-000000.tar): the "
                        "480-sample subsets are randomly drawn and contain almost no "
                        "consecutive runs.  Default: cfg.data.eval dev-clean/dev-other.")
    p.add_argument("--turns", type=int, default=6, help="utterances per conversation")
    p.add_argument("--max-convs", type=int, default=100, dest="max_convs",
                   help="conversations per split (0 = all)")
    p.add_argument("--max-runs-per-group", type=int, default=0, dest="max_runs_per_group",
                   help="cap runs taken from one (speaker, chapter); 0 = unlimited, "
                        "which is the literal 'first N runs' selection")
    p.add_argument("--max-ref-tokens", type=int, default=41, dest="max_ref_tokens",
                   help="drop utterances whose either label exceeds this many tokens "
                        "(default 41, matching the single-turn eval subsets); 0 disables")
    p.add_argument("--conv-batch-size", type=int, default=8, dest="conv_batch_size",
                   help="conversations decoded concurrently; the context tensor is "
                        "this many full conversations wide, so it drives peak VRAM")
    p.add_argument("--formats", nargs="+", choices=list(FORMATS), default=list(FORMATS),
                   help="instruction variants to run (default: both, paired)")
    p.add_argument("--context-modes", nargs="+", dest="context_modes",
                   choices=list(CONTEXT_MODES), default=list(CONTEXT_MODES),
                   help="context variants to run (default: all three)")
    p.add_argument("--max-new-slack", type=int, default=10, dest="slack",
                   help="generation cap = longest reference in the batch + this")
    p.add_argument("--out-dir", type=Path, default=Path("out"), dest="out_dir",
                   help="directory for longform-<split>.jsonl")
    p.add_argument("--dataset", type=str, default=None,
                   help="dataset tag written onto every row")
    p.add_argument("--print-layout", type=int, default=0, dest="print_layout",
                   help="print the detokenized layout of the first N conversations "
                        "(assembled prefix, turn by turn) and exit before decoding "
                        "if --layout-only is also given")
    p.add_argument("--layout-only", action="store_true", dest="layout_only",
                   help="print layouts and stop; no generation")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg  = load_config(args.config)

    if cfg.model.input_convention != "chat":
        print(f"[error] input_convention is {cfg.model.input_convention!r}. "
              "Multi-turn evaluation only means anything under the chat convention — "
              "the flat layout has no turn structure to extend.")
        return 1

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # ── Tokenizer, scaffold, instruction ids ─────────────────────────────────
    tokenizer     = PrunedTokenizer(cfg.data.tokenizer)
    terminator_id = load_pruned_config(cfg.data.tokenizer).terminator_id
    chat          = ChatTemplate.from_tokenizer(tokenizer)
    if terminator_id != chat.eot_token_id:
        print(f"[error] terminator {terminator_id} != chat eot {chat.eot_token_id}")
        return 1
    # seg_user_continuation swaps seg_pre_audio's leading [bos] for an [eot];
    # if that first id were not the bos the swap would corrupt the header.
    if chat.seg_pre_audio[0] != tokenizer.bos_token_id:
        print(f"[error] chat scaffold does not open with [bos] "
              f"({chat.seg_pre_audio[0]} != {tokenizer.bos_token_id})")
        return 1
    print(f"Chat convention: audio offset {chat.audio_offset}, "
          f"terminator <|eot_id|> = {terminator_id}")

    instruction_ids = {
        "unformatted": tokenizer.encode(INSTRUCTION_VARIANTS[0]),
        "formatted":   tokenizer.encode(INSTRUCTION_VARIANTS[1]),
    }

    # ── Shards ───────────────────────────────────────────────────────────────
    if args.shards:
        shard_map = _parse_shards(args.shards)
    else:
        shard_map = [(n, p) for n, p in (("dev-clean", cfg.data.eval.dev_clean),
                                         ("dev-other", cfg.data.eval.dev_other))
                     if p is not None]
        print("[warn] no --shard given, falling back to cfg.data.eval.  Those are "
              "usually the 480-sample random subsets, which contain almost no "
              "consecutive runs — expect very few conversations.")
    if not shard_map:
        print("[error] no eval shards to read.")
        return 1

    max_ref_tokens = args.max_ref_tokens if args.max_ref_tokens > 0 else None

    # ── Conversation construction (before any model is built) ────────────────
    conversations: dict[str, list[Conversation]] = {}
    for split, path in shard_map:
        print(f"\nReading {split} ← {path}")
        samples = read_eval_shard(path)
        convs, stats = build_conversations(
            samples, split, tokenizer,
            turns              = args.turns,
            max_convs          = args.max_convs,
            max_ref_tokens     = max_ref_tokens,
            max_runs_per_group = args.max_runs_per_group,
        )
        print(f"  {stats['n_samples']:>6} utterances in shard")
        if stats["n_unparsed_keys"]:
            print(f"  {stats['n_unparsed_keys']:>6} keys not of the form "
                  "speaker-chapter-utterance — dropped")
        print(f"  {stats['n_too_long']:>6} dropped by the ≤{max_ref_tokens}-token "
              f"reference filter")
        print(f"  {stats['n_eligible']:>6} eligible, in {stats['n_groups']} "
              "(speaker, chapter) groups")
        print(f"  {stats['candidate_runs']:>6} candidate runs of {args.turns}; "
              f"{stats['accepted_runs']} fully consecutive, "
              f"{stats['dropped_runs']} dropped for gaps")
        print(f"  {stats['selected']:>6} conversations selected "
              f"({len({c.speaker for c in convs})} speakers, "
              f"{len({(c.speaker, c.chapter) for c in convs})} chapters)")
        if not convs:
            print(f"[error] {split} yielded no conversations.  Point --shard at the "
                  "full split shard, or lower --turns.")
            return 1
        conversations[split] = convs

    # ── Model ────────────────────────────────────────────────────────────────
    encoder, adapter, llama, _ = build_models(
        cfg, device, train=False, apply_init_from=False,
    )
    if args.checkpoint is not None:
        if args.checkpoint.name.endswith("-adapter.pt"):
            print(f"[error] {args.checkpoint.name} is a weights-only adapter sidecar, "
                  "not a full checkpoint delta — every other module would stay at its "
                  "pretrained value and the WER would describe a model that never "
                  "existed.  Pass the 'step*.pt' / '*-stage-handoff.pt' file.")
            return 1
        ckpt   = read_checkpoint(args.checkpoint)
        loaded = apply_weights(ckpt, encoder=encoder, adapter=adapter, llama=llama)
        del ckpt
        step = _parse_step(args.checkpoint.stem)
        print(f"Checkpoint {args.checkpoint.name} (step {step}); modules {sorted(loaded)}")
        checkpoint_str = str(args.checkpoint)
    else:
        step, checkpoint_str = 0, "<none>"
        print("[warn] no --checkpoint: evaluating the untrained base model.  "
              "Plumbing test only — the numbers mean nothing.")
    encoder.eval(); adapter.eval(); llama.eval()

    # ── Sequence-length guard ────────────────────────────────────────────────
    # A conversation that overran the context would not error — RoPE tables are
    # precomputed far past it and attention would simply run — so check it here
    # and say the number out loud.
    max_seq_len = llama.config.max_seq_len
    worst = max(
        (worst_case_length(c, chat, len(instruction_ids[f]), tokenizer, f, args.slack),
         c.conv_id, f)
        for convs in conversations.values() for c in convs for f in args.formats
    )
    print(f"\nWorst-case sequence length at {args.turns} turns: {worst[0]} tokens "
          f"({worst[1]}, {worst[2]}); model context {max_seq_len}")
    if worst[0] > max_seq_len:
        print(f"[error] {worst[0]} tokens exceeds the model context of {max_seq_len}. "
              f"Lower --turns.")
        return 1

    # ── Layout dump ──────────────────────────────────────────────────────────
    if args.print_layout:
        _print_layouts(
            conversations, args, chat, tokenizer, instruction_ids,
            encoder, adapter, llama, device,
        )
        if args.layout_only:
            return 0

    # ── Decode ───────────────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, convs in conversations.items():
        rows: list[dict] = []
        chunks = [convs[i : i + args.conv_batch_size]
                  for i in range(0, len(convs), args.conv_batch_size)]
        for fmt in args.formats:
            for ctx_mode in args.context_modes:
                print(f"\n[{split}] {ctx_mode} / {fmt}: "
                      f"{len(convs)} conversations x {args.turns} turns "
                      f"in {len(chunks)} chunk(s)")
                for ci, chunk in enumerate(chunks):
                    kwargs = dict(
                        encoder=encoder, adapter=adapter, llama=llama,
                        tokenizer=tokenizer, chat=chat, terminator_id=terminator_id,
                        instr_ids=instruction_ids[fmt], device=device, slack=args.slack,
                    )
                    if ctx_mode == "single":
                        hyps = decode_single_turn(chunk, fmt, **kwargs)
                    else:
                        hyps = decode_multi_turn(chunk, fmt, ctx_mode, **kwargs)
                    for conv, conv_hyps in zip(chunk, hyps):
                        rows.extend(make_rows(
                            conv, fmt, ctx_mode, conv_hyps,
                            checkpoint_str, step, args.dataset,
                        ))
                    print(f"  chunk {ci + 1}/{len(chunks)} done "
                          f"({len(rows)} rows so far)")

        out_path = args.out_dir / f"longform-{split}.jsonl"
        with out_path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"\n{split}: {len(rows)} rows → {out_path}")
        print_summary(rows, split)

    return 0


def _print_layouts(
    conversations, args, chat, tokenizer, instruction_ids,
    encoder, adapter, llama, device,
) -> None:
    """Assemble and print the first N conversations' prefixes for eyeball inspection.

    Uses the model only for its embedding table and the bridge's markers — the
    audio is real but only its LENGTH shows up in the dump, so this is cheap and
    runs before any decoding.  The instruction mode shown is the first requested.
    """
    fmt     = args.formats[0]
    instr_t = torch.tensor(instruction_ids[fmt], dtype=torch.long, device=device)
    # PrunedTokenizer.decode strips the scaffold specials by design (they are
    # never transcript text), so name them explicitly instead of losing them.
    special_names = {
        tokenizer.bos_token_id:    "<|begin_of_text|>",
        tokenizer.start_header_id: "<|start_header_id|>",
        tokenizer.end_header_id:   "<|end_header_id|>",
        tokenizer.eot_token_id:    "<|eot_id|>",
    }

    def decode(ids: list[int]) -> str:
        out, run = [], []
        for i in ids:
            if i in special_names:
                if run:
                    out.append(tokenizer.decode(run)); run = []
                out.append(special_names[i])
            else:
                run.append(i)
        if run:
            out.append(tokenizer.decode(run))
        return "".join(out)

    shown = 0
    with torch.no_grad():
        for split, convs in conversations.items():
            for conv in convs:
                if shown >= args.print_layout:
                    return
                shown += 1
                batch = _turn_batch(conv.utterances, instruction_ids[fmt])
                mel   = batch[0].to(device)
                lens  = batch[1].to(device)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    adapter_out = adapter(encoder(mel))

                cp = ConversationPrefix(
                    chat, llama.embed_tokens, adapter.audio_bos, adapter.audio_eos,
                )
                print(f"\n══ layout: {conv.conv_id}  ({fmt}, {len(conv.utterances)} turns) ══")
                for t, s in enumerate(conv.utterances):
                    cp.add_user_turn(adapter_out[t, : int(lens[t].item()), :], instr_t)
                    if t + 1 < len(conv.utterances):
                        cp.add_assistant_text(
                            tokenizer.encode(reference_for(s, fmt))
                        )
                print(cp.render(decode))
                print(f"  total {len(cp)} positions, "
                      f"{int(cp.audio_flags.sum())} of them audio")


# ── Self-test ─────────────────────────────────────────────────────────────────

def _self_test() -> None:
    """Conversation construction, on synthetic keys.  No data, no model, no GPU.

    Covers the part that is cheap to get wrong and expensive to notice: a run
    that silently spans a gap in the reading order would look like a valid
    conversation and quietly measure a discontinuity instead of long-form
    context.
    """
    import numpy as np

    print("run_longform.py self-test")

    class _Tok:
        """Token count == word count; long enough refs trip the length filter."""
        def encode(self, text: str) -> list[int]:
            return list(range(len(text.split())))

    def sample(key: str, words: int = 3, audio: int = 10) -> EvalSample:
        return EvalSample(
            key             = key,
            mel             = np.zeros((80, audio * 8), dtype=np.float32),
            audio_length    = audio,
            ref_unformatted = " ".join(["w"] * words),
            ref_formatted   = " ".join(["W"] * words),
        )

    tok = _Tok()

    # ── 1. A gap drops the run that spans it ──────────────────────────────────
    # Chapter 1 runs 0..5 clean.  Chapter 2 is missing utterance 2, so its
    # 0..5 window can never form; 3..8 can.
    samples = [sample(f"100-1-{i:04d}") for i in range(6)]
    samples += [sample(f"100-2-{i:04d}") for i in range(9) if i != 2]
    convs, stats = build_conversations(samples, "s", tok, 3, 0, None, 0)
    got = [[u.key for u in c.utterances] for c in convs]
    assert got == [
        ["100-1-0000", "100-1-0001", "100-1-0002"],
        ["100-1-0003", "100-1-0004", "100-1-0005"],
        ["100-2-0003", "100-2-0004", "100-2-0005"],
        ["100-2-0006", "100-2-0007", "100-2-0008"],
    ], got
    # Chapter 2's utterances 0 and 1 are stranded on the short side of the gap:
    # 8 eligible utterances would allow 2 runs there, only 2 of which form.
    assert stats["candidate_runs"] == 2 + 2, stats
    assert stats["accepted_runs"] == 4 and stats["dropped_runs"] == 0, stats
    for c in convs:
        idx = [int(_UTT_RE.match(u.key).group(3)) for u in c.utterances]
        assert idx == list(range(idx[0], idx[0] + 3)), idx
    print("  [OK] runs are fully consecutive; a gap splits rather than stitches")

    # A gap that leaves too few utterances on either side drops a candidate run.
    holed = [sample(f"200-1-{i:04d}") for i in range(6) if i != 3]
    convs, stats = build_conversations(holed, "s", tok, 3, 0, None, 0)
    assert [len(c.utterances) for c in convs] == [3], convs
    assert stats["candidate_runs"] == 1 and stats["dropped_runs"] == 0, stats
    print("  [OK] drop accounting reports candidates vs accepted")

    # ── 2. The length filter runs BEFORE grouping, and punches gaps ───────────
    mixed = [sample(f"300-1-{i:04d}", words=(50 if i == 2 else 3)) for i in range(6)]
    convs, stats = build_conversations(mixed, "s", tok, 3, 0, max_ref_tokens=41,
                                       max_runs_per_group=0)
    assert stats["n_too_long"] == 1, stats
    assert stats["n_eligible"] == 5, stats
    # 5 eligible would allow one run; the filtered-out utterance 2 means only
    # 3,4,5 is contiguous.
    assert [[u.key for u in c.utterances] for c in convs] == [
        ["300-1-0003", "300-1-0004", "300-1-0005"]
    ], convs
    print("  [OK] ≤N-token filter applied per utterance, before run formation")

    # ── 3. Selection is deterministic and order-independent ──────────────────
    import random
    shuffled = list(samples)
    random.Random(7).shuffle(shuffled)
    a, _ = build_conversations(samples,  "s", tok, 3, 3, None, 0)
    b, _ = build_conversations(shuffled, "s", tok, 3, 3, None, 0)
    assert [c.conv_id for c in a] == [c.conv_id for c in b], (a, b)
    assert len(a) == 3, "max_convs must cap the selection"
    print("  [OK] selection is deterministic regardless of shard member order")

    # ── 4. max_runs_per_group trades literal ordering for speaker breadth ─────
    wide = [sample(f"{spk}-1-{i:04d}") for spk in (400, 401) for i in range(6)]
    capped, _ = build_conversations(wide, "s", tok, 3, 2, None, max_runs_per_group=1)
    assert {c.speaker for c in capped} == {400, 401}, capped
    uncapped, _ = build_conversations(wide, "s", tok, 3, 2, None, max_runs_per_group=0)
    assert {c.speaker for c in uncapped} == {400}, uncapped
    print("  [OK] max_runs_per_group spreads conversations across speakers")

    # ── 5. An unparseable key is counted and dropped, not crashed on ─────────
    _, stats = build_conversations(
        samples + [sample("not-an-utterance-id-x")], "s", tok, 3, 0, None, 0,
    )
    assert stats["n_unparsed_keys"] == 1, stats
    print("  [OK] unparseable utterance ids are counted and dropped")

    # ── 6. Row schema matches run_wer.py plus the three new fields ────────────
    conv = build_conversations(samples, "s", tok, 3, 1, None, 0)[0][0]
    rows = make_rows(conv, "unformatted", "self", ["w w w", "w w", ""],
                     "ckpt.pt", 4680, "libri")
    assert len(rows) == 3
    assert set(rows[0]) == {
        "checkpoint", "step", "dataset", "key", "split", "type",
        "reference", "hypothesis", "wer", "conv_id", "turn_index", "context_mode",
    }, sorted(rows[0])
    assert [r["turn_index"] for r in rows] == [1, 2, 3]
    assert rows[0]["wer"] == 0.0 and rows[2]["wer"] == 1.0
    print("  [OK] JSONL row schema (run_wer.py fields + conv_id/turn_index/context_mode)")

    print("\nPASSED")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
        sys.exit(0)
    sys.exit(main())
