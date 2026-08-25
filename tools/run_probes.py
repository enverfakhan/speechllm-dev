"""Instruction-generalisation probe suite — generation half.

Speech-IFEval in miniature, against this project's own checkpoints.  Every probe
is an OUTPUT CONSTRAINT that does not depend on what the audio says, so
instruction-following is measured SEPARATELY from perception: a model can
transcribe perfectly and still fail all of them.

    P1  uppercase        transcribe in capitals          (see the gate below)
    P2  word_count       transcript, then a count line
    P3  last_word        answer ABOUT the audio, one word
    P4  ignore_audio     answer a question, ignoring the audio entirely
                         + a TEXT-ONLY control of the same questions
    P5  prompt_variation the trained task, in unseen wordings

All five are ZERO-SHOT.  Training only ever saw the two strings in
data.INSTRUCTION_VARIANTS, in one fixed single-turn layout; none of these
instructions, output formats or question texts appears anywhere in it.

THE FEASIBILITY GATE IS NOT OPTIONAL
------------------------------------
The vocabulary is pruned to the LibriSpeech label ids (Decision 005), so a probe
whose expected output falls outside it is impossible BY CONSTRUCTION and scoring
the model on it would report a property of tools/build_vocab.py.  This tool
refuses to run a probe that tools/check_vocab_feasibility.py has not cleared.
Pass --feasibility out/probe-feasibility.json; --force downgrades the refusal to
a warning and stamps every affected row with "gate_forced": true, so a forced
number can never be mistaken for a clean one later.

WHAT IS HELD CONSTANT
---------------------
One utterance sample, selected once, drives every probe, and the batches are
formed the same way each time (ascending audio length, as the WER loader does).
That matters more than it looks: the Whisper encoder attends bidirectionally
across mel padding, so batch composition perturbs the audio embeddings.  Probes
compared against each other, and P5's paraphrases compared against the canonical
string, must sit in the same batches — which is also why P5 decodes the
canonical instruction HERE rather than lifting a number from a banked WER sweep.

USAGE
-----
    # local stub sanity — prints every prompt and raw generation
    python tools/run_probes.py --config configs/stub_probes.yaml \\
        --shard data_dev/dev_shards/dev-diag-2196-000000.tar \\
        --probes P3 P4 --n 5 --print-prompts 2 --out-dir out/probes-stub

    # the real run
    python tools/run_probes.py --config configs/instruct-chat-3stage.yaml \\
        --checkpoint checkpoints/instruct-chat-3stage/step0015040.pt \\
        --shard data/eval_shards/dev-clean-000000.tar \\
        --feasibility out/probe-feasibility.json \\
        --out-dir results/probes/

    python tools/run_probes.py --self-test

Writes one JSONL per probe into --out-dir, named by probe_spec.probe_filename().
tools/score_probes.py consumes them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TOOLS_DIR = Path(__file__).resolve().parent
for _p in (str(_REPO_ROOT), str(_TOOLS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import probe_spec  # noqa: E402  (stdlib-only, same tools/ dir)
from probe_spec import (  # noqa: E402
    P4_QUESTIONS, P5_VARIANTS, PROBES, PROBE_IDS, Question, Variant,
    p4_instruction, probe_filename,
)

from build import build_models
from data import (
    INSTRUCTION_VARIANTS, EvalSample, PrunedTokenizer,
    _eval_collate_batch, load_pruned_config, read_eval_shard,
)
from model.sequence import ChatTemplate, ConversationPrefix, ConversationPrefixBatch
from utils.checkpoint import apply_weights, read_checkpoint
from utils.config import load_config
from utils.evaluate import compute_wer
from utils.generate import greedy_generate, greedy_generate_from_prefix


# Extra tokens allowed past the longest reference in a batch.  Same rule as
# utils/evaluate.py, so one runaway sample cannot stall the run — and so a
# hypothesis that overruns its reference is still VISIBLE rather than cut off at
# the reference length, which is the whole signal for P3.
DEFAULT_SLACK: int = 10

# P2 must fit the transcript AND a second line holding a number.
P2_EXTRA_TOKENS: int = 8


# ── Utterance selection ───────────────────────────────────────────────────────

def select_utterances(
    samples:        list[EvalSample],
    n:              int,
    seed:           int,
    tokenizer:      PrunedTokenizer,
    max_ref_tokens: int | None,
) -> tuple[list[EvalSample], dict]:
    """Draw the probe sample deterministically, then order it for batching.

    Selection is a seeded draw over KEY-SORTED samples, not over tarfile member
    order, so it does not move if a shard is ever rewritten.  The ≤max_ref_tokens
    filter runs FIRST and matches the one the WER sweeps and run_longform.py
    apply, so a WER measured here is comparable with the banked ones.

    The returned list is sorted ascending by audio length — the batching the WER
    loader uses.  Every probe then sees identical batches, which is required for
    them to be comparable to each other (the Whisper encoder attends across mel
    padding, so batch composition changes the audio embeddings).

    Args:
        samples:        everything read out of the shard
        n:              how many utterances to draw
        seed:           RNG seed for the draw
        tokenizer:      for the reference-length filter
        max_ref_tokens: drop utterances whose verbatim reference exceeds this
                        many tokens; None disables the filter

    Returns:
        (selected samples sorted by audio_length, stats dict)
    """
    import random

    by_key = sorted(samples, key=lambda s: s.key)
    stats  = {"n_shard": len(by_key), "max_ref_tokens": max_ref_tokens}

    if max_ref_tokens is not None:
        eligible = [
            s for s in by_key
            if len(tokenizer.encode(s.ref_unformatted)) <= max_ref_tokens
        ]
    else:
        eligible = list(by_key)
    stats["n_dropped_long"] = len(by_key) - len(eligible)
    stats["n_eligible"]     = len(eligible)

    if not eligible:
        raise ValueError("no utterances survive the reference-length filter")
    if n > len(eligible):
        print(f"[warn] asked for {n} utterances but only {len(eligible)} are "
              f"eligible — using all of them")
        n = len(eligible)

    chosen = random.Random(seed).sample(eligible, n)
    stats["n_selected"] = len(chosen)
    stats["n_speakers"] = len({s.key.split("-")[0] for s in chosen})
    return sorted(chosen, key=lambda s: s.audio_length), stats


def _chunks(items: list, size: int) -> list[list]:
    """Contiguous chunks of at most `size`, final partial chunk included."""
    return [items[i : i + size] for i in range(0, len(items), size)]


# ── Generation ────────────────────────────────────────────────────────────────

def _generation_cap(refs: list[str], tokenizer: PrunedTokenizer, slack: int) -> int:
    """Longest reference in the batch, in tokens, plus slack (evaluate.py's rule)."""
    return max(len(tokenizer.encode(r)) for r in refs) + slack


@torch.no_grad()
def decode_with_audio(
    samples:   list[EvalSample],
    instruction_ids: list[int],
    cap_refs:  list[str] | None,
    *,
    encoder, adapter, llama,
    tokenizer:     PrunedTokenizer,
    chat:          ChatTemplate | None,
    terminator_id: int,
    device:        torch.device,
    batch_size:    int,
    slack:         int = DEFAULT_SLACK,
    extra:         int = 0,
    fixed_cap:     int | None = None,
    label:         str = "",
) -> list[str]:
    """Greedy-decode ONE instruction over a list of utterances, batched.

    Runs through the untouched greedy_generate / EvalPrefixBatch path — the same
    code every banked WER number came from — so a probe hypothesis and a sweep
    hypothesis are produced by identical machinery.

    Args:
        samples:         utterances, already in batching order
        instruction_ids: pruned ids of the probe's instruction
        cap_refs:        per-sample reference strings used to size the generation
                         cap (same order as samples); ignored when fixed_cap is set
        slack:           tokens allowed past the longest reference in a batch
        extra:           further tokens for probes whose output is longer than a
                         transcript (P2's count line)
        fixed_cap:       absolute cap, for probes whose expected output has
                         nothing to do with the reference length (P4)
        label:           progress-line prefix

    Returns:
        decoded hypothesis per sample, in the input order
    """
    hyps: list[str] = []
    chunks = _chunks(samples, batch_size)
    for ci, chunk in enumerate(chunks):
        batch = _eval_collate_batch([
            (s.mel, instruction_ids, instruction_ids,
             s.ref_unformatted, s.ref_formatted, s.key)
            for s in chunk
        ])
        mel           = batch[0].to(device)
        audio_lengths = batch[1].to(device)
        instr_t       = batch[2].to(device)   # both slots hold the same instruction
        instr_lens    = batch[3].to(device)

        if fixed_cap is not None:
            cap = fixed_cap
        else:
            start = ci * batch_size
            refs  = cap_refs[start : start + len(chunk)]
            cap   = _generation_cap(refs, tokenizer, slack) + extra

        gen = greedy_generate(
            encoder, adapter, llama,
            mel, audio_lengths, instr_t, instr_lens,
            stop_token_id  = terminator_id,
            max_new_tokens = cap,
            chat           = chat,
        )
        hyps.extend(tokenizer.decode(g) for g in gen)
        if label:
            print(f"  {label}: batch {ci + 1}/{len(chunks)} "
                  f"({len(hyps)}/{len(samples)} decoded, cap {cap})")
    return hyps


def _text_only_prefix_ids(chat: ChatTemplate, instruction_ids: list[int]) -> list[int]:
    """The chat prompt for one instruction with the audio block removed.

    The trained user turn is::

        seg_pre_audio ▸BOS [audio] ▸EOS seg_pre_instruction {instruction}
        seg_pre_transcript

    The text-only control is that, minus the markers and the audio content, and
    nothing else — same headers, same "\\n" separator, same assistant header —
    so the only difference between the two P4 conditions is the presence of
    audio.  Removing more (the "\\n", say) would make the gap between them a
    prompt difference as well as an audio difference, and the gap is the entire
    measurement.
    """
    return (list(chat.seg_pre_audio) + list(chat.seg_pre_instruction)
            + list(instruction_ids) + list(chat.seg_pre_transcript))


@torch.no_grad()
def decode_text_only(
    instruction_id_lists: list[list[int]],
    *,
    llama,
    tokenizer:     PrunedTokenizer,
    chat:          ChatTemplate,
    terminator_id: int,
    device:        torch.device,
    max_new_tokens: int,
) -> list[str]:
    """Decode instructions with NO audio anywhere in the context.

    Batched through ConversationPrefixBatch, which is the class that already
    knows how to greedy-decode prebuilt prefixes of unequal length under the
    same padding / insert-at-gen_pos discipline as the single-turn path.  Its
    audio mask is all zeros here, which is exactly right: with no audio there is
    nothing for the in-layer audio adapters to fire on.

    The backbone is frozen, so this condition is the frozen Instruct model
    answering through the pruned vocabulary.  The gap between it and the same
    question asked over audio is what the audio conditioning costs.
    """
    embed    = llama.embed_tokens
    prefixes: list[torch.Tensor] = []
    flags:    list[torch.Tensor] = []
    for ids in instruction_id_lists:
        seq = torch.tensor(_text_only_prefix_ids(chat, ids), dtype=torch.long, device=device)
        emb = embed(seq)
        prefixes.append(emb)
        flags.append(torch.zeros(emb.shape[0], device=device))

    pfx = ConversationPrefixBatch(prefixes, flags)
    gen = greedy_generate_from_prefix(
        llama, pfx, terminator_id, max_new_tokens, audio_lengths=None,
    )
    return [tokenizer.decode(g) for g in gen]


# ── Row construction ──────────────────────────────────────────────────────────

def _sample_wer(reference: str, hypothesis: str) -> float:
    """WER of one pair through the aggregate's own jiwer path, un-normalized.

    Mirrors tools/run_wer.py and tools/run_longform.py, so a per-sample number
    means the same thing in all three files.  For P3 and P4 the "reference" is a
    single expected word, so this is a crude answer-distance, NOT a
    transcription WER — the scorer uses exact match there and ignores it.
    """
    if not reference.strip():
        return float("nan")
    return compute_wer([reference], [hypothesis])


def base_row(
    sample:      EvalSample | None,
    probe_id:    str,
    instruction: str,
    reference:   str,
    hypothesis:  str,
    *,
    split:       str,
    checkpoint:  str,
    step:        int,
    type_tag:    str,
    gate_forced: bool,
) -> dict:
    """One JSONL row in run_wer.py's field order, plus the probe fields.

    Keeping run_wer.py's names means tools/count_degeneracies.py and
    tools/analyze_slices.py read these files with no adapter — useful for P5,
    whose rows really are transcription rows under a different instruction.
    """
    return {
        "checkpoint":  checkpoint,
        "step":        step,
        "probe":       probe_id,
        "key":         sample.key if sample is not None else f"{probe_id}-text-only",
        "split":       split,
        "type":        type_tag,
        "instruction": instruction,
        "reference":   reference,
        "hypothesis":  hypothesis,
        "wer":         _sample_wer(reference, hypothesis),
        **({"gate_forced": True} if gate_forced else {}),
    }


# ── The probes ────────────────────────────────────────────────────────────────

def run_p1(samples, ctx) -> list[dict]:
    """P1 — transcribe in ALL CAPITALS.

    Scored case-insensitively for WER (perception held constant) and on the
    fraction of alphabetic characters that are upper case (the constraint).
    Reference stays the ordinary verbatim label; the scorer upper-cases.
    """
    probe = PROBES["P1"]
    instr_ids = ctx["tokenizer"].encode(probe.instruction)
    refs = [s.ref_unformatted for s in samples]
    hyps = decode_with_audio(samples, instr_ids, refs, label="P1", **ctx["gen"])
    return [
        base_row(s, "P1", probe.instruction, s.ref_unformatted, h,
                 type_tag="uppercase", **ctx["meta"])
        for s, h in zip(samples, hyps)
    ]


def run_p2(samples, ctx) -> list[dict]:
    """P2 — transcript, then the word count of that transcript on a new line.

    The pass criterion is SELF-consistency (the number matches the model's own
    transcript), because that is what the instruction as written demands.
    Agreement with the reference's word count is reported separately: it mixes
    in transcription errors and is a different question.
    """
    probe = PROBES["P2"]
    instr_ids = ctx["tokenizer"].encode(probe.instruction)
    refs = [s.ref_unformatted for s in samples]
    hyps = decode_with_audio(samples, instr_ids, refs, extra=P2_EXTRA_TOKENS,
                             label="P2", **ctx["gen"])
    return [
        base_row(s, "P2", probe.instruction, s.ref_unformatted, h,
                 type_tag="word_count", **ctx["meta"])
        for s, h in zip(samples, hyps)
    ]


def run_p3(samples, ctx) -> list[dict]:
    """P3 — answer with only the last word spoken.

    The generation cap is the FULL transcript length + slack, not one word, so
    that a model which dumps the whole transcript instead of answering is
    visible rather than truncated into looking like a near-miss.  That dump is
    the ASR-collapse signature and the scorer reports its rate directly.
    """
    probe = PROBES["P3"]
    instr_ids = ctx["tokenizer"].encode(probe.instruction)
    refs = [s.ref_unformatted for s in samples]
    hyps = decode_with_audio(samples, instr_ids, refs, label="P3", **ctx["gen"])

    rows = []
    for s, h in zip(samples, hyps):
        last = s.ref_unformatted.split()[-1] if s.ref_unformatted.split() else ""
        row  = base_row(s, "P3", probe.instruction, last, h,
                        type_tag="last_word", **ctx["meta"])
        # The whole transcript rides along so the scorer can tell "answered the
        # wrong word" from "ignored the question and transcribed everything".
        row["full_reference"] = s.ref_unformatted
        rows.append(row)
    return rows


def run_p4(samples, ctx, n_per_question: int, max_new: int) -> list[dict]:
    """P4 — ignore the audio and answer a question, plus the text-only control.

    The 10 questions get DISJOINT blocks of utterances, so no clip is asked two
    different questions and no question is asked over one clip's quirks.  Blocks
    are cut from the length-sorted sample, so each question's block is
    length-homogeneous — which is also what keeps its batches clean.

    The control asks the same 10 questions with no audio in the context at all.
    The backbone is frozen, so with-audio versus text-only IS the leakage
    measurement: any drop is what the audio conditioning costs, not what the
    LLM does not know.
    """
    probe_instrs = {q.qid: p4_instruction(q) for q in P4_QUESTIONS}
    tokenizer    = ctx["tokenizer"]
    rows: list[dict] = []

    for qi, q in enumerate(P4_QUESTIONS):
        block = samples[qi * n_per_question : (qi + 1) * n_per_question]
        if not block:
            print(f"[warn] no utterances left for question {q.qid} — "
                  "raise --n or lower --n-per-question")
            continue
        instr_ids = tokenizer.encode(probe_instrs[q.qid])
        hyps = decode_with_audio(block, instr_ids, None, fixed_cap=max_new,
                                 label=f"P4/{q.qid}", **ctx["gen"])
        for s, h in zip(block, hyps):
            row = base_row(s, "P4", probe_instrs[q.qid], q.answer, h,
                           type_tag="ignore_audio", **ctx["meta"])
            row.update(qid=q.qid, question=q.text, expected=q.answer,
                       context="audio")
            rows.append(row)

    # ── Text-only control ────────────────────────────────────────────────────
    if ctx["chat"] is None:
        print("[warn] flat convention: the text-only control needs the chat "
              "scaffold to build a prompt with no audio — skipping it.")
        return rows

    instr_lists = [tokenizer.encode(probe_instrs[q.qid]) for q in P4_QUESTIONS]
    hyps = decode_text_only(
        instr_lists,
        llama=ctx["llama"], tokenizer=tokenizer, chat=ctx["chat"],
        terminator_id=ctx["terminator_id"], device=ctx["device"],
        max_new_tokens=max_new,
    )
    print(f"  P4/text-only: {len(hyps)} questions decoded with no audio")
    for q, h in zip(P4_QUESTIONS, hyps):
        row = base_row(None, "P4", probe_instrs[q.qid], q.answer, h,
                       type_tag="ignore_audio", **ctx["meta"])
        row.update(qid=q.qid, question=q.text, expected=q.answer,
                   context="text_only", key=f"text-only-{q.qid}")
        rows.append(row)
    return rows


def run_p5(samples, ctx) -> list[dict]:
    """P5 — the trained task in unseen wordings, canonical string included.

    Every variant runs over the SAME utterances in the SAME batches, and the
    canonical instruction is decoded here rather than read off a banked sweep,
    so the spread across wordings is prompt sensitivity and nothing else.
    """
    tokenizer = ctx["tokenizer"]
    rows: list[dict] = []
    for v in P5_VARIANTS:
        instr_ids = tokenizer.encode(v.text)
        refs = [s.ref_unformatted if v.mode == "unformatted" else s.ref_formatted
                for s in samples]
        hyps = decode_with_audio(samples, instr_ids, refs,
                                 label=f"P5/{v.mode}/{v.vid}", **ctx["gen"])
        for s, ref, h in zip(samples, refs, hyps):
            row = base_row(s, "P5", v.text, ref, h,
                           type_tag=v.mode, **ctx["meta"])
            row.update(variant_id=v.vid, mode=v.mode, is_canonical=v.is_canonical)
            rows.append(row)
    return rows


# ── Prompt dump ───────────────────────────────────────────────────────────────

def _special_decoder(tokenizer: PrunedTokenizer):
    """A decoder that NAMES the scaffold specials instead of dropping them.

    PrunedTokenizer.decode strips them by design (they are never transcript
    text), which is exactly wrong when the scaffold itself is what needs
    eyeballing.
    """
    names = {
        tokenizer.bos_token_id:    "<|begin_of_text|>",
        tokenizer.start_header_id: "<|start_header_id|>",
        tokenizer.end_header_id:   "<|end_header_id|>",
        tokenizer.eot_token_id:    "<|eot_id|>",
    }

    def decode(ids: list[int]) -> str:
        out, run = [], []
        for i in ids:
            if i in names:
                if run:
                    out.append(tokenizer.decode(run)); run = []
                out.append(names[i])
            else:
                run.append(i)
        if run:
            out.append(tokenizer.decode(run))
        return "".join(out)

    return decode


@torch.no_grad()
def print_prompts(samples, probe_ids, ctx, n: int) -> None:
    """Print the assembled prompt for the first n utterances of each probe.

    The point of the stub sanity run: see the EXACT string the model is handed,
    scaffold specials and all, before renting a GPU.  Uses the model only for
    its embedding table and the bridge markers.
    """
    chat = ctx["chat"]
    if chat is None:
        print("[warn] --print-prompts only renders the chat scaffold; the flat "
              "convention has no layout to dump.")
        return
    tokenizer = ctx["tokenizer"]
    decode    = _special_decoder(tokenizer)
    encoder, adapter, llama = ctx["encoder"], ctx["adapter"], ctx["llama"]
    device    = ctx["device"]

    def one(instruction: str, sample: EvalSample | None, title: str) -> None:
        print(f"\n══ prompt: {title} ══")
        instr_ids = tokenizer.encode(instruction)
        if sample is None:
            ids = _text_only_prefix_ids(chat, instr_ids)
            print(f"  (no audio)  {len(ids)} positions")
            print("  " + decode(ids).replace("\n", "\\n"))
            return
        batch = _eval_collate_batch([
            (sample.mel, instr_ids, instr_ids,
             sample.ref_unformatted, sample.ref_formatted, sample.key)
        ])
        mel  = batch[0].to(device)
        lens = batch[1].to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            adapter_out = adapter(encoder(mel))
        cp = ConversationPrefix(chat, llama.embed_tokens,
                                adapter.audio_bos, adapter.audio_eos)
        cp.add_user_turn(
            adapter_out[0, : int(lens[0].item()), :].float(),
            torch.tensor(instr_ids, dtype=torch.long, device=device),
        )
        print(f"  key {sample.key}   reference: {sample.ref_unformatted!r}")
        print(cp.render(decode))
        print(f"  total {len(cp)} positions, {int(cp.audio_flags.sum())} audio")

    for pid in probe_ids:
        if pid == "P4":
            for q in P4_QUESTIONS[:n]:
                one(p4_instruction(q), samples[0], f"P4 {q.qid} (with audio)")
                one(p4_instruction(q), None,       f"P4 {q.qid} (TEXT-ONLY control)")
        elif pid == "P5":
            for v in P5_VARIANTS[:n]:
                one(v.text, samples[0], f"P5 {v.mode}/{v.vid}")
        else:
            for s in samples[:n]:
                one(PROBES[pid].instruction, s, f"{pid} {PROBES[pid].name}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_step(stem: str, fallback: int = 0) -> int:
    """Pull the step number out of a checkpoint filename, as run_wer.py does."""
    import re
    m = re.search(r"(\d{4,})", stem)
    return int(m.group(1)) if m else fallback


def read_gate(path: Path | None) -> dict[str, dict]:
    """Load the feasibility gate, or an empty gate when none was given."""
    if path is None:
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f).get("probes", {})


def apply_gate(
    requested: list[str], gate: dict[str, dict], force: bool,
) -> tuple[list[str], list[dict]]:
    """Drop probes the gate has not cleared, and say why.

    Returns:
        (probes to run, dropped-probe records for the report)
    """
    run, dropped = [], []
    for pid in requested:
        verdict = gate.get(pid)
        if verdict is None:
            record = {"probe": pid, "status": "ungated",
                      "reason": "no feasibility gate supplied"}
        elif verdict.get("feasible"):
            run.append(pid)
            continue
        else:
            record = {"probe": pid, "status": verdict.get("status", "infeasible"),
                      "reason": verdict.get("reason", "")}

        if force:
            record["forced"] = True
            print(f"[warn] --force: running {pid} anyway ({record['status']}: "
                  f"{record['reason']}). Every row it writes is stamped "
                  "gate_forced=true.")
            run.append(pid)
        else:
            print(f"[gate] {pid} DROPPED — {record['status']}: {record['reason']}")
        dropped.append(record)
    return run, dropped


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", type=Path, default=None,
                   help="Training config YAML (model dims, tokenizer, convention). "
                        "Required for everything except --self-test.")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Full checkpoint ('step*.pt' / '*-stage-handoff.pt'). "
                        "Omitted = untrained base model; plumbing only.")
    p.add_argument("--shard", type=Path, default=None,
                   help="Eval shard .tar to draw utterances from "
                        "(default: cfg.data.eval.dev_clean).")
    p.add_argument("--split", type=str, default="dev-clean",
                   help="Split name stamped on every row (default dev-clean).")
    p.add_argument("--probes", nargs="+", default=list(PROBE_IDS),
                   choices=list(PROBE_IDS), metavar="ID",
                   help=f"Probes to run (default: all of {', '.join(PROBE_IDS)}).")
    p.add_argument("--feasibility", type=Path, default=None,
                   help="Gate JSON from tools/check_vocab_feasibility.py. Without "
                        "it every probe is 'ungated' and will not run unless --force.")
    p.add_argument("--force", action="store_true",
                   help="Run probes the gate refused. Rows are stamped gate_forced=true.")
    p.add_argument("--n", type=int, default=200,
                   help="Utterances in the probe sample (default 200).")
    p.add_argument("--n-per-question", type=int, default=None, dest="n_per_question",
                   help="P4 clips per question (default: --n // 10).")
    p.add_argument("--seed", type=int, default=17,
                   help="Seed for the utterance draw (default 17).")
    p.add_argument("--max-ref-tokens", type=int, default=41, dest="max_ref_tokens",
                   help="Drop utterances with a longer verbatim reference; 0 "
                        "disables. Default 41 — the WER sweeps' filter, so a WER "
                        "measured here is comparable with the banked ones.")
    p.add_argument("--batch-size", type=int, default=None, dest="batch_size",
                   help="Generation batch size (default: cfg.metrics.eval_batch_size).")
    p.add_argument("--slack", type=int, default=DEFAULT_SLACK,
                   help=f"Tokens allowed past the longest reference (default {DEFAULT_SLACK}).")
    p.add_argument("--max-new-p4", type=int, default=32, dest="max_new_p4",
                   help="Absolute generation cap for P4, whose expected answer is "
                        "one word but whose failure mode is rambling (default 32).")
    p.add_argument("--out-dir", type=Path, default=Path("results/probes"), dest="out_dir")
    p.add_argument("--print-prompts", type=int, default=0, dest="print_prompts",
                   metavar="N",
                   help="Dump the assembled prompt for the first N items of each "
                        "probe and exit unless --decode-anyway.")
    p.add_argument("--decode-anyway", action="store_true", dest="decode_anyway",
                   help="Decode as well as printing prompts.")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.self_test:
        _self_test()
        return 0
    if args.config is None:
        print("[error] --config is required (it carries the model dims, the "
              "tokenizer and the input convention).")
        return 1

    cfg = load_config(args.config)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # ── The trained instructions must be the ones probe_spec thinks they are ──
    # probe_spec keeps its own copy so the stdlib-only tools can read it; if the
    # two ever drift, P5's "canonical" variant would not be the trained string
    # and its whole spread would be measured against the wrong baseline.
    if list(probe_spec.TRAINED_INSTRUCTIONS.values()) != list(INSTRUCTION_VARIANTS):
        print("[error] probe_spec.TRAINED_INSTRUCTIONS has drifted from "
              "data.INSTRUCTION_VARIANTS. P5's canonical variant would not be "
              "the string the model was trained on.")
        return 1

    # ── Tokenizer / convention ───────────────────────────────────────────────
    tokenizer     = PrunedTokenizer(cfg.data.tokenizer)
    terminator_id = load_pruned_config(cfg.data.tokenizer).terminator_id
    chat: ChatTemplate | None = None
    if cfg.model.input_convention == "chat":
        chat = ChatTemplate.from_tokenizer(tokenizer)
        print(f"Chat convention: audio offset {chat.audio_offset}, "
              f"terminator <|eot_id|> = {terminator_id}")
    else:
        print(f"[warn] input_convention is {cfg.model.input_convention!r}. The "
              "probes run, but P4's text-only control needs the chat scaffold "
              "and will be skipped.")

    # ── Gate ─────────────────────────────────────────────────────────────────
    gate = read_gate(args.feasibility)
    if not gate:
        print("[warn] no --feasibility gate supplied. A probe whose expected "
              "output is unreachable under the pruned vocabulary is impossible "
              "by construction, and scoring it would report a property of the "
              "vocabulary as a model failure. Run "
              "tools/check_vocab_feasibility.py first.")
    probes_to_run, dropped = apply_gate(args.probes, gate, args.force)
    if not probes_to_run:
        print("[error] no probes left to run.")
        return 1
    print(f"Probes: {', '.join(probes_to_run)}")

    # ── Utterances ───────────────────────────────────────────────────────────
    shard = args.shard or cfg.data.eval.dev_clean
    if shard is None or not Path(shard).exists():
        print(f"[error] eval shard not found: {shard}")
        return 1
    print(f"\nReading {args.split} ← {shard}")
    samples, sel_stats = select_utterances(
        read_eval_shard(shard), args.n, args.seed, tokenizer,
        args.max_ref_tokens if args.max_ref_tokens > 0 else None,
    )
    print(f"  {sel_stats['n_shard']:>6} utterances in shard")
    print(f"  {sel_stats['n_dropped_long']:>6} dropped by the "
          f"≤{sel_stats['max_ref_tokens']}-token reference filter")
    print(f"  {sel_stats['n_selected']:>6} selected "
          f"({sel_stats['n_speakers']} speakers), sorted by audio length")

    n_per_question = args.n_per_question or max(1, len(samples) // len(P4_QUESTIONS))

    # ── Model ────────────────────────────────────────────────────────────────
    encoder, adapter, llama, _ = build_models(cfg, device, train=False,
                                              apply_init_from=False)
    step, checkpoint_str = 0, "<none>"
    if args.checkpoint is not None:
        if args.checkpoint.name.endswith("-adapter.pt"):
            print(f"[error] {args.checkpoint.name} is a weights-only adapter "
                  "sidecar, not a full checkpoint delta — every other module "
                  "would stay at its pretrained value and the numbers would "
                  "describe a model that never existed. Pass the 'step*.pt' / "
                  "'*-stage-handoff.pt' file.")
            return 1
        ckpt   = read_checkpoint(args.checkpoint)
        loaded = apply_weights(ckpt, encoder=encoder, adapter=adapter, llama=llama)
        del ckpt
        step, checkpoint_str = _parse_step(args.checkpoint.stem), str(args.checkpoint)
        print(f"Checkpoint {args.checkpoint.name} (step {step}); "
              f"modules {sorted(loaded)}")
    else:
        print("[warn] no --checkpoint: probing the untrained base model. "
              "Plumbing test only — the numbers mean nothing.")
    encoder.eval(); adapter.eval(); llama.eval()

    batch_size = args.batch_size or cfg.metrics.eval_batch_size

    ctx = {
        "tokenizer": tokenizer, "chat": chat, "terminator_id": terminator_id,
        "device": device, "encoder": encoder, "adapter": adapter, "llama": llama,
        "gen": {
            "encoder": encoder, "adapter": adapter, "llama": llama,
            "tokenizer": tokenizer, "chat": chat, "terminator_id": terminator_id,
            "device": device, "batch_size": batch_size, "slack": args.slack,
        },
        "meta": {
            "split": args.split, "checkpoint": checkpoint_str, "step": step,
            "gate_forced": bool(args.force and dropped),
        },
    }

    # ── Prompt dump ──────────────────────────────────────────────────────────
    if args.print_prompts:
        print_prompts(samples, probes_to_run, ctx, args.print_prompts)
        if not args.decode_anyway:
            print("\n(prompt dump only — pass --decode-anyway to generate too)")
            return 0

    # ── Run ──────────────────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written: list[tuple[str, Path, int]] = []
    for pid in probes_to_run:
        print(f"\n── {pid} {PROBES[pid].name} ──\n  {PROBES[pid].measures}")
        if pid == "P1":
            rows = run_p1(samples, ctx)
        elif pid == "P2":
            rows = run_p2(samples, ctx)
        elif pid == "P3":
            rows = run_p3(samples, ctx)
        elif pid == "P4":
            rows = run_p4(samples, ctx, n_per_question, args.max_new_p4)
        else:
            rows = run_p5(samples, ctx)

        out_path = args.out_dir / probe_filename(pid)
        with out_path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        written.append((pid, out_path, len(rows)))
        print(f"  {len(rows)} rows → {out_path}")

    # ── Run manifest: what ran, what was dropped, and why ────────────────────
    manifest = {
        "checkpoint":   checkpoint_str,
        "step":         step,
        "config":       str(args.config),
        "tokenizer":    str(cfg.data.tokenizer),
        "input_convention": cfg.model.input_convention,
        "shard":        str(shard),
        "split":        args.split,
        "seed":         args.seed,
        "selection":    sel_stats,
        "batch_size":   batch_size,
        "n_per_question": n_per_question,
        "feasibility":  str(args.feasibility) if args.feasibility else None,
        "probes_run":   [p for p, _, _ in written],
        "probes_dropped": dropped,
        "forced":       bool(args.force),
        "generations":  sum(n for _, _, n in written),
        "outputs":      {p: str(path) for p, path, _ in written},
    }
    man_path = args.out_dir / "probe-run-manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n{manifest['generations']} generations across "
          f"{len(written)} probe(s) → {args.out_dir}")
    print(f"manifest → {man_path}")
    if dropped:
        print("\nDropped by the feasibility gate (report this):")
        for d in dropped:
            print(f"  {d['probe']}  {d['status']}: {d['reason']}")
    return 0


# ── Self-test ─────────────────────────────────────────────────────────────────

def _self_test() -> None:
    """Selection, gating, batching and row shape — no model, no GPU, no data."""
    import numpy as np

    def sample(key: str, words: int, audio: int) -> EvalSample:
        return EvalSample(
            key=key, mel=np.zeros((80, audio * 8), np.float32),
            audio_length=audio,
            ref_unformatted=" ".join(["word"] * words),
            ref_formatted=" ".join(["Word"] * words) + ".",
        )

    class StubTok:
        """Whitespace tokenizer; enough for the length filter and the caps."""

        def encode(self, text: str) -> list[int]:
            return [1] * len(text.split())

        def decode(self, ids: list[int]) -> str:
            return " ".join("w" for _ in ids)

    tok = StubTok()
    pool = [sample(f"spk{i % 4}-ch-{i:04d}", words=1 + i % 9, audio=100 - i)
            for i in range(40)]

    sel, stats = select_utterances(pool, 10, seed=1, tokenizer=tok, max_ref_tokens=None)
    assert len(sel) == 10 and stats["n_selected"] == 10, stats
    assert [s.audio_length for s in sel] == sorted(s.audio_length for s in sel), \
        "selection must come back sorted ascending by audio length"
    sel2, _ = select_utterances(pool, 10, seed=1, tokenizer=tok, max_ref_tokens=None)
    assert [s.key for s in sel] == [s.key for s in sel2], "selection must be deterministic"
    sel3, _ = select_utterances(pool, 10, seed=2, tokenizer=tok, max_ref_tokens=None)
    assert [s.key for s in sel3] != [s.key for s in sel], "a new seed must redraw"
    print("  [OK] select_utterances: deterministic, seed-sensitive, length-ordered")

    _, stats_f = select_utterances(pool, 5, seed=1, tokenizer=tok, max_ref_tokens=3)
    assert stats_f["n_dropped_long"] > 0 and stats_f["n_eligible"] < len(pool), stats_f
    print("  [OK] reference-length filter drops the long utterances")

    assert _chunks(list(range(7)), 3) == [[0, 1, 2], [3, 4, 5], [6]]
    print("  [OK] _chunks keeps the final partial batch")

    assert _generation_cap(["a b c", "a b"], tok, 10) == 13
    print("  [OK] _generation_cap = longest reference + slack")

    # Gate behaviour: infeasible drops, --force runs and is recorded.
    gate = {
        "P1": {"feasible": False, "status": "infeasible", "reason": "uppercase"},
        "P3": {"feasible": True,  "status": "feasible",   "reason": ""},
    }
    run, dropped = apply_gate(["P1", "P3", "P4"], gate, force=False)
    assert run == ["P3"], run
    assert {d["probe"] for d in dropped} == {"P1", "P4"}, dropped
    assert [d["status"] for d in dropped if d["probe"] == "P4"] == ["ungated"], dropped
    run_f, dropped_f = apply_gate(["P1", "P3"], gate, force=True)
    assert run_f == ["P1", "P3"] and dropped_f[0]["forced"] is True, (run_f, dropped_f)
    print("  [OK] apply_gate: drops the ungated and the infeasible, --force records itself")

    # A row carries run_wer.py's field names, so the existing analysis tools read it.
    row = base_row(pool[0], "P3", "instr", "word", "word", split="dev-clean",
                   checkpoint="ck.pt", step=15040, type_tag="last_word",
                   gate_forced=False)
    for field in ("checkpoint", "step", "key", "split", "type", "reference",
                  "hypothesis", "wer"):
        assert field in row, field
    assert "gate_forced" not in row, "clean rows must not carry the forced stamp"
    forced = base_row(pool[0], "P3", "i", "w", "w", split="s", checkpoint="c",
                      step=0, type_tag="t", gate_forced=True)
    assert forced["gate_forced"] is True
    print("  [OK] base_row: run_wer.py-compatible fields, forced rows stamped")

    # The text-only prompt is the audio prompt minus exactly the audio block.
    chat = ChatTemplate(seg_pre_audio=(1, 2, 3), seg_pre_instruction=(4,),
                        seg_pre_transcript=(5, 6), eot_token_id=6)
    ids = _text_only_prefix_ids(chat, [7, 8])
    assert ids == [1, 2, 3, 4, 7, 8, 5, 6], ids
    print("  [OK] _text_only_prefix_ids drops the markers and audio, nothing else")

    # probe_spec must not have drifted from the trained instructions.
    assert list(probe_spec.TRAINED_INSTRUCTIONS.values()) == list(INSTRUCTION_VARIANTS), \
        "probe_spec.TRAINED_INSTRUCTIONS drifted from data.INSTRUCTION_VARIANTS"
    print("  [OK] probe_spec's copy of the trained instructions matches data.py")
    print("PASSED")


if __name__ == "__main__":
    raise SystemExit(main())
