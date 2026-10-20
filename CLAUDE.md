# speech-llm — Claude Code Project Brief

This document is the persistent brief for Claude Code. Re-read it at the start of
every session. It contains the architecture, all confirmed decisions, repo
structure, component interfaces, invariants, and the active task. Do not make
architectural decisions that contradict anything in this file without explicit
instruction from the project lead.

If this file and the actual code disagree, the code is ground truth — flag the
drift, don't silently follow the stale text.

---

## Project Overview

An end-to-end instructable SpeechLLM trained on LibriSpeech 960h. The model takes
raw audio and produces a transcript, conditioned on a natural-language instruction
prompt (unformatted vs formatted output). Training is **multi-stage** (declarative
stage list in YAML): typically adapter-only first, then wider unfreezing.

**Core goals (in priority order):**
1. Educational clarity — every component readable and self-contained
2. Cost efficiency — decisions minimise training cost (RunPod/GCP)
3. Open-source reproducibility — code, weights, write-ups will be released
4. WER quality — secondary; sufficient to show meaningful loss decrease and a
   cost/WER trade-off narrative
**Project lead:** sole developer. Design decisions happen in the planning chat;
implementation is delegated here via explicit prompts. Flag untested paths and
prefer root-cause fixes over patches.

---

## Architecture

```
Raw audio (16 kHz, ≤ 30 s)
        ↓
[ Whisper small encoder ]   88M params   d_model=768, 12 layers, Pre-LN
        ↓  (B, T_mel//2, 768)            conv stem stride-2 halves T_mel
[ Temporal mean pool ×4 ]   no params    → (B, ceil(T_enc/4), 768)   80 ms/token
        ↓
[ Bridge adapter ]          ~10M params  mlp: Linear(768→2048) → GELU → Linear(2048→4096)
                            ~14M params  swiglu: SwiGLU(768→2048→4096)   [bridge_type]
        ↓  (B, ≤375, 4096)
[ Llama 3.1 8B ]            8B params    32 layers, d_model=4096, GQA 32/8, SwiGLU
        ↓
[ LM head (tied) ]          loss on transcript tokens + trailing terminator
```

### Input sequence format (model/sequence.py)

Two conventions, selected by `cfg.model.input_convention` (Decision 011):

```
flat   [audio embeddings] [SEP] [instruction] [SEP] [transcript] [SEP]

chat   [bos][start_header]"user"[end_header]"\n\n"
           AUDIO_BOS [audio embeddings] AUDIO_EOS "\n" [instruction]
       [eot]
       [start_header]"assistant"[end_header]"\n\n"
           [transcript]
       [eot]                                    <- EOS target
```

- Audio embeddings are adapter outputs injected directly into Llama's residual
  stream (Llama.forward always takes `inputs_embeds`, never token IDs).
- The **trailing terminator is the EOS target**: it is NOT masked to -100; the
  model is trained to predict it so greedy decoding terminates naturally. Under
  `chat` the user-turn `[eot]` is input-only and stays masked — only the final
  one is a label.
- Loss mask: -100 everywhere except real transcript positions and that trailing
  terminator.
- Per-sample padding is stripped before concatenation; sequences are then
  batch-padded to the longest sample. No pad token ever enters the loss.
- `assemble_inputs()` dispatches between the two; the flat path is byte-for-byte
  what it always was, so every pre-chat config and self-test is unaffected.

### Two instruction variants (instructable training)

```
"Transcribe the audio exactly as spoken, in lowercase with no punctuation."
    → unformatted.txt label
"Transcribe the audio as written text, with capitalization, punctuation, and
 numbers as digits."
    → formatted.txt label
```

Variant selection is controlled by `instruction_mode` ("unformatted", "formatted",
"both") at run level, overridable per stage.

---

## Confirmed Decisions

### 001 — LLM backbone: Llama 3.1 8B
From-scratch implementation (RMSNorm, RoPE θ=500000, GQA 32/8, SwiGLU, tied
embeddings). License compatible with open-source release. Vocab pruned to
40,034 (flat) / 40,039 (chat) — see 005. Weight loading supports HF safetensors
and Meta consolidated.*.pth. The chat line (011) swaps the same architecture to
the **Instruct** checkpoint; only the weight values differ.

### 002 — Audio encoder: Whisper small, encoder only
88M params, d_model=768, 12 layers, learned positional embeddings, Pre-LN.
Decoder discarded. Initialised from the OpenAI pretrained checkpoint
(`load_openai_weights`, shape-asserted). No openai-whisper / transformers
dependency in model code.

### 003 — Bridge adapter: temporal mean-pool ×4 + MLP or SwiGLU
(B, T, 768) → pool → projection → (B, ceil(T/4), 4096). Two variants selected by
`cfg.model.bridge_type` (`build_bridge_adapter`; pooling is shared via `_pool_time`):
- `"mlp"` (default, `AudioAdapter`) — Linear(768→2048) → GELU → Linear(2048→4096).
  Every checkpoint predating the SwiGLU bridge contains this one.
- `"swiglu"` (`AudioSwiGLUBridge`) — `down_proj(silu(gate_proj(x)) ⊙ up_proj(x))`,
  768→2048→4096, ~14M params.
**The bridge's output projection must be NEAR-zero, never exactly zero** (both
variants: std = 0.02/√6, zero bias). The bridge output is the entire content of
the audio positions and Llama is bias-free, so exactly-zero embeddings stay zero
through all 32 layers; each RMSNorm on that dead stream adds a 1/√eps ≈ 316×
backward gain, the gradient reaching the bridge hits ~1e7, fp16 overflows, and
GradScaler skips every step — the model never leaves the degenerate state.
(In-layer audio adapters CAN zero their writer: they add to an already-live
residual stream.)

Optional PCA init via `model.adapter_pca_init` (tools/compute_adapter_pca_init.py)
→ `mlp[0]` for the MLP variant, `up_proj` (the un-squashed value path) for SwiGLU.

Both variants also carry the two learned audio delimiter vectors `audio_bos` /
`audio_eos` used by the chat convention (`_init_audio_markers`, Decision 011).
They ride in the bridge's `adapter` param group and state_dict for free; the flat
convention never reads them.

### 004 — Multi-stage training, resume & checkpoint semantics
- Training is a `for stage_idx in range(start_stage, len(stages))` loop over the
  YAML `stages:` list. Each stage sets `requires_grad` in place, builds a FRESH
  AdamW8bit (optimizer_init="inherit" is NotImplementedError — bnb 8-bit moments
  aren't portable across param-group changes) and its own LR schedule.
- Two checkpoint roles, enforced by a `kind` field:
  - `kind="periodic"` — full state (optimizer, scaler, scheduler, RunState);
    consumed by `--resume`. Untagged legacy checkpoints default to periodic.
  - `kind="handoff"` — weights-only (stage handoffs + adapter sidecars);
    consumed by `model.init_from` (warm start). Feeding one to `--resume` is
    refused with a redirect.
- Resume start stage is derived from the checkpoint's `stage_index`
  (`cfg.run.start_stage` does not exist). Data-order freshness on resume comes
  from setting a DIFFERENT `cfg.seed` in the resuming config — there is no
  batch fast-forwarding (was dead code; removed).

### 005 — Vocabulary: pruned Llama tokenizer (40,034 flat / 40,039 chat)
`vocab_map.json` maps old IDs → contiguous new IDs. Embedding rows initialised
from pretrained checkpoint rows via vocab_map. Built by tools/build_vocab.py.

- Flat line: 40,034 tokens, SEP = old 127,999 → new 40,033.
- Chat line (011): 40,039 tokens; the four Llama 3.1 chat specials are FORCED IN
  by canonical id — `<|begin_of_text|>` 128000 → 40,035, `<|start_header_id|>`
  128006 → 40,036, `<|end_header_id|>` 128007 → 40,037, `<|eot_id|>` 128009 →
  40,038. `pruned_config.json` carries all four plus `vocab_size`;
  `eot_token_id` IS the terminator and `sep_token_id` is written as an alias of
  it. `data.load_pruned_config()` is the single reader — no caller parses that
  JSON by hand any more.

Why by canonical id and not by name: a Meta-format directory loads through
transformers' TikToken fallback, which exposes NO specials (`all_special_ids ==
[]`, `convert_tokens_to_ids` returns None, `len(tokenizer) == 128000`). The old
scan-downward fallback therefore landed on 127,999 — the ordinary CJK token
`锦` — which served as the EOS target for the entire flat line.

**The vocabulary must be built from the SAME label text the shards hold**, via
`--labels_file data/labels.jsonl` (records marked `validation: failed` skipped,
plus the two instruction strings). `PrunedTokenizer.encode` silently DROPS ids
missing from the map, so a mismatched vocabulary does not fail — it quietly
deletes tokens from every affected label. The tool's older `--librispeech_dir`
mode re-derives labels with the BasicTextNormalizer / PunctuationModel / spaCy
pipeline and is correct only for shards built that same way.

History: the corpus was relabelled by tools/label_formatted.py (LLM labels, see
FORMATTING_SPEC.md), which changed both fields — `unformatted` keeps apostrophes
the old BasicTextNormalizer stripped, and `formatted` adds `;`, `!`, `Mr`/`Mrs`
and capitalised number words. Against the old 40,148-token vocab that was 325,231
OOV occurrences (22% of unformatted, 42% of formatted utterances), so the vocab
was rebuilt fresh rather than extended. **Rebuilding reassigns every id**, which
is safe only because no checkpoint in the active experiment line stores a trained
`llama`: utils/checkpoint.py saves llama only when it is in `modules_dirty`, and
build.py rebuilds the embedding from the pretrained weights through vocab_map on
every load. A full-finetune checkpoint (`trainable: [encoder, adapter, llama]`)
would need the vocab extended in place instead, preserving rows 0..40,147.

Duplicated copies of these numbers that must be updated together with a rebuild:
`metrics.py:_VOCAB_SIZE` (now only a self-test/fallback default — MetricCollector
takes `vocab_size` from pruned_config at run time) and `tools/probe_accum.py`
(`_VOCAB_SIZE`, VRAM stubs — a standalone probe that builds no tokenizer).
tools/compute_baselines.py, build.py, training.py, tools/run_wer.py and
tools/find_max_batch.py all read pruned_config.json. **`baselines.json` is also
invalidated** — the uniform baseline is log(V); re-run tools/compute_baselines.py.

`--llama_dir` must NOT point at a previously built pruned_tokenizer directory:
build_vocab now refuses to copy a stale `vocab_map.json` / `pruned_config.json`
over its own output. It used to, silently pairing a fresh pruned_config with the
previous build's map and mis-indexing every embedding row.

### 006 — Shard format: variable-length mel
`{key}.mel.npy` float16 (80, T) at natural length; `{key}.unformatted.txt`;
`{key}.formatted.txt`. DataLoader collation pads each batch to
T_max = ceil(max/8)·8. Shards sized by audio duration (default 30 min).

### 007 — audio_lengths formula
`audio_lengths[i] = (T_mel_i // 2 + 3) // 4` — encoder stride-2 (floor), then
pool-4 (ceiling). This is the adapter-token count used for sequence assembly,
padding masks, and throughput accounting (1 adapter token = 80 ms audio).

### 008 — Mixed precision: fp16 autocast + GradScaler, fp32 master
Training forward runs under `torch.amp.autocast("cuda", dtype=torch.float16)`
with GradScaler. **bf16 is ruled out for training**: GradScaler is incompatible
with bf16, and without scaling the adapter gradients underflow to zero through
the frozen Llama layers (confirmed empirically). Params/grads/optimizer master
state are fp32. fp16 master weights are a deferred, documented underflow
experiment — do not implement without instruction.

### 009 — Memory strategy
8-bit AdamW (bitsandbytes) is mandatory for full-model single-GPU training
(fp32 AdamW state = 96 GB > 80 GB). Gradient checkpointing
(`use_reentrant=False`, transformer layers only) is the main activation lever —
with it, measured max batch on A100 80GB: stage 1 = 72 (use 64), stage 2 = 12
(use 8). bnb defers state allocation to early optimizer.step() calls — warm up
before memory probing. Flash Attention via `F.scaled_dot_product_attention`.
`accum_steps=1` for small-corpus local runs (accumulation caused gradient-spike
artefacts; use it only when effective batch size genuinely must exceed VRAM).

### 010 — Diagnostic eval design
The training-time diag shard is a 1920-sample randomized-order subset spanning
all four LibriSpeech eval splits, consumed by a PERSISTENT iterator that sweeps
it fully every 3 eval cycles. The iterator persistence is load-bearing — do not
rebuild it per eval. Per-split detail is the WER tool's job, not the diag signal.
WER eval uses `build_sorted_eval_dataloader` (tarfile-based, finite, sorted
ascending by adapter-token length, references filtered to ≤41 tokens). It yields
9-tuples: the shard member key rides along as the last element so a post-hoc tool
can attribute an individual hypothesis back to its utterance — WER alone cannot
say WHICH samples a checkpoint got wrong. The sort is stable and tarfile member
order is fixed, so batching (and every per-sample position) is reproducible
across runs; that is what makes a per-sample A/B between two checkpoints or two
code revisions meaningful.


### 011 — Instruct backbone + chat input convention (new experiment line)

`cfg.model.input_convention: "flat" | "chat"` (default `flat`; the base-model
line and every stub self-test stay on `flat`, unchanged). The chat line
(`configs/instruct-chat-3stage.yaml`) swaps the frozen LLM to Llama 3.1 8B
**Instruct** and assembles the sequence with the Instruct chat template.

- `<|eot_id|>` is the ONLY terminator, retiring the accidental `锦` SEP (see
  005). Its embedding row comes from the Instruct checkpoint through vocab_map,
  so it carries a real termination prior; an arbitrary CJK token carries none.
- **AUDIO_BOS / AUDIO_EOS**: two learned `nn.Parameter(d_model)` vectors on the
  BRIDGE (`model/adapter.py::_init_audio_markers`), spliced as embeddings. They
  are deliberately not vocabulary tokens — no logit row, never generated. Living
  on the bridge puts them in the `adapter` param group and the always-saved
  adapter state_dict for free (no `_MODULE_ORDER` entry, no delta-invariant
  change). Plain normal(0, 0.02) init: the near-zero rule of 003 applies to the
  bridge OUTPUT (the entire content of the audio positions), not to two vectors
  sitting among live scaffold embeddings.
- **Audio first in the user turn**, instruction last: every token before
  AUDIO_BOS is fixed scaffold, so audio starts at the CONSTANT offset
  `ChatTemplate.audio_offset = len(seg_pre_audio) + 1` in every sample.
- **The audio mask is therefore no longer `[0, audio_lengths)`.**
  `prepare_input_chat` returns an explicit `(B, S, 1)` mask over the audio
  CONTENT positions only (markers and scaffold excluded) and `Llama.forward`
  takes it as `audio_mask=`, which WINS over `audio_lengths`. The flat path still
  passes `audio_lengths` and builds the `[0, L)` mask exactly as before. A wrong
  offset here would silently train the in-layer adapters on scaffold tokens.
- `ChatTemplate` (model/sequence.py) precomputes the scaffold id segments once
  per run and is the single source of truth for assembly; it is id-space
  agnostic, which is what lets tools/build_vocab.py build the same scaffold in
  ORIGINAL id space to check it. The scaffold texts are
  `model/sequence.CHAT_SCAFFOLD_TEXTS`, which build_vocab imports rather than
  re-declaring.
- Scaffold verification at vocabulary-build time, in two parts:
  `_verify_scaffold_boundaries` ALWAYS runs and needs no chat template — it
  proves no BPE merge crosses either text↔text splice (`"\n"`→instruction,
  `"\n\n"`→transcript; everywhere else an atomic special or a marker VECTOR
  separates the runs), checked against 200 real corpus label strings.
  `_verify_chat_scaffold` additionally cross-checks the turn LAYOUT against
  `apply_chat_template` (local windows only — Llama 3.1's template always emits
  a system/date block this convention omits), and is SKIPPED with a notice for a
  Meta-format checkpoint directory, which ships no `chat_template`. That skip is
  expected, not a failure.
- Curriculum is THREE stages (`bridge_only → audio_adapters_only → full_stack`);
  the joint bridge+adapters stage is dropped per Finding D-iii. No warm start:
  the D1 checkpoints were trained against the base backbone, the old vocabulary
  AND the flat layout, so early loss is not step-comparable to D1 — compare
  endpoints (WER at the stage handoffs) instead.
- Checkpoint back-compat: bridge state_dicts written before the markers existed
  load with a warning and keep the fresh markers
  (`utils/checkpoint._load_bridge_state_tolerant`); any other missing or
  unexpected key stays a hard error, so an mlp↔swiglu bridge mismatch still
  fails loudly.

This line is **Series E**, run **E1** (EXPERIMENT_LOG.md). Findings E-i (chat
loses to flat on WER), E-ii (two opposite terminator failures), E-iii (training
past the 15040 handoff makes it worse — treat 15040 as the endpoint) and E-iv
(single-turn training does not transfer to multi-turn at all).

---

## Environment

- Local: WSL2 Ubuntu 22.04, Python 3.11.15, venv `source ~/.venvs/aidev/bin/activate`,
  RTX 3060 12GB — stub-mode dev and self-tests only.
- Experiments: RunPod Community Cloud A100 80GB (~$0.89/hr), Network Volume at
  /workspace. Data/weights/checkpoints sync with GCS `gs://speechllm-data/`.
- Scale (later): GCP project `speechllm-train`, multi-GPU pending.
- Experiment tracking: W&B. Run configs use project `speechllm-auto-stage`;
  `configs/base.yaml` defaults to `speech-llm-dev` for local/stub runs.

---

## Repo Structure

```
speech-llm/
├── CLAUDE.md
├── training.py            # config-driven multi-stage training loop (entry point)
├── build.py               # build_models(): construct + init encoder/adapter/llama
├── data.py                # WebDataset loaders, PrunedTokenizer, INSTRUCTION_VARIANTS
├── metrics.py             # MetricCollector + render(): all diagnostic signals
├── stages.py              # Stage / StageContext: trainability, optimizer, loader, exit
│
├── model/
│   ├── whisper_encoder.py # Whisper small encoder from scratch (+ log_mel_spectrogram)
│   ├── adapter.py         # bridges (pool + MLP/SwiGLU, PCA hook, audio markers)
│   ├── llama.py           # Llama 3.1 8B from scratch (+ weight loading)
│   └── sequence.py        # ChatTemplate, prepare_input[_chat](), assemble_inputs(),
│                          #   EvalPrefixBatch (batched generation, both conventions)
│
├── utils/
│   ├── config.py          # frozen-dataclass config; base.yaml → run yaml → CLI
│   ├── checkpoint.py      # save/load, kind tagging, ResumeState,
│   │                      #   read_checkpoint + apply_weights (one read)
│   ├── evaluate.py        # evaluate_all_splits (WER engine, jiwer, no normalisation)
│   ├── generate.py        # greedy_generate (batched, EvalPrefixBatch)
│   └── optim.py           # build_adamw8bit, make_warmup_scheduler
│
├── tools/                 # standalone CLIs (independently readable; duplication OK)
│   ├── download_data.py         # LibriSpeech splits from OpenSLR, MD5-verified
│   ├── download_weights.py      # Whisper ckpt fetch + Llama manual instructions
│   ├── build_vocab.py           # prune tokenizer → data/pruned_tokenizer/
│   ├── precompute_labels.py     # full formatting pipeline once → labels.jsonl
│   ├── preprocess.py            # FLAC → mel → WebDataset shards (+ manifest)
│   ├── create_subset.py         # reproducible shard-list files
│   ├── make_dev_dataset.py      # single-speaker dev corpus + diag holdout
│   ├── make_eval_subset.py      # reproducible eval subset .tar (tarfile, finite)
│   ├── make_stub_shard.py       # random-mel stub shards → data_stub/ (gitignored)
│   │                            #   from tests/fixtures/stub_labels.jsonl; what
│   │                            #   every configs/stub*.yaml reads. Run it once
│   │                            #   on a fresh checkout before any stub run
│   ├── compute_adapter_pca_init.py  # PCA basis for adapter first linear
│   ├── compute_baselines.py     # uniform/unigram/bigram/first-token baselines
│   ├── find_max_batch.py        # binary-search max batch per stage (VRAM)
│   ├── probe_accum.py           # VRAM probes (fragmentation vs bnb lazy state)
│   ├── run_wer.py               # multi-checkpoint post-hoc WER sweep → CSV/JSON
│   │                            #   + a FULL per-(step, split) transcription JSONL
│   │                            #   (every sample, per-sample WER, utterance key);
│   │                            #   --splits restricts the eval set; --eval-tar
│   │                            #   NAME=PATH evaluates arbitrary shards (the OOD
│   │                            #   sets) with --dataset tagging every row;
│   │                            #   rejects '-adapter.pt' sidecars loudly
│   ├── plot_wer.py              # WER-vs-step curve from run_wer.py's CSV, with
│   │                            #   stage spans + epoch markers (matplotlib +
│   │                            #   stdlib only; no torch)
│   ├── count_degeneracies.py    # empty/truncation/hallucination/loop/style-flip
│   │                            #   counts over run_wer.py's JSONL (stdlib only).
│   │                            #   OWNS the degeneracy rules — analyze_slices.py
│   │                            #   imports its classify(), so the two never drift
│   ├── analyze_slices.py        # WER / degeneracy / instruction-adherence slices
│   │                            #   over the same JSONL: by mode, reference-length
│   │                            #   bucket, numeral presence, rare-word presence;
│   │                            #   + a behavioural example gallery (numeral
│   │                            #   format-switch hits AND misses, loops, flips).
│   │                            #   stdlib only. WER is UN-normalized and
│   │                            #   reconciles with run_wer.py's wer.csv to 2dp
│   ├── build_vocab_freq.py      # word -> training count from labels.jsonl, for
│   │                            #   analyze_slices.py --train-vocab (stdlib only)
│   ├── compare_hyps.py          # per-utterance hypothesis A/B between two
│   │                            #   transcription JSONLs, joined on key — the
│   │                            #   regression gate for a refactor (greedy decode
│   │                            #   is deterministic, so strings must match)
│   ├── label_formatted.py       # LLM labelling run (Batches API): prepare/submit/
│   │                            #   poll/validate/retry/finalize → labels.jsonl
│   ├── relabel_shards.py        # swap labels into EXISTING shards, drop failed
│   │                            #   samples; no mel recompute (see pipeline note)
│   │
│   ├── run_longform.py          # zero-shot MULTI-TURN transcription eval (see
│   │                            #   its own section below): conversations of
│   │                            #   consecutive utterances, self / oracle /
│   │                            #   single context modes, one JSONL in
│   │                            #   run_wer.py's schema + conv_id / turn_index
│   │                            #   / context_mode for analyze_slices --group-by
│   │
│   │  ── instruction-generalisation probe suite (see its own section below) ──
│   ├── probe_spec.py            # the ONE definition of every probe: prompts,
│   │                            #   P4 questions, P5 paraphrases, feasibility
│   │                            #   candidate groups.  stdlib only; the gate,
│   │                            #   the generator and the scorer all import it
│   │                            #   so a prompt string exists exactly once
│   ├── check_vocab_feasibility.py  # THE GATE, runs before any decoding: can
│   │                            #   the pruned vocabulary even express a
│   │                            #   probe's expected output — and does its
│   │                            #   PROMPT survive encoding?  Writes the gate
│   │                            #   JSON run_probes.py refuses to run without
│   ├── run_probes.py            # generation half: one JSONL per probe, in
│   │                            #   run_wer.py's schema; includes P4's
│   │                            #   TEXT-ONLY control (a chat prompt with no
│   │                            #   audio block at all)
│   ├── score_probes.py          # scoring half (stdlib only): Following Rate +
│   │                            #   pass rate per probe → one Markdown report.
│   │                            #   Imports WER / adherence from
│   │                            #   analyze_slices.py so P5's canonical row is
│   │                            #   comparable with the banked slice reports
│   │
│   │  ── out-of-distribution eval protocol (see its own section below) ──
│   ├── ood_shard.py             # the ONE shard-layout contract the three
│   │                            #   prepare_ood_* builders share (+ the .flac
│   │                            #   member the paired control reads)
│   ├── prepare_ood_tedlium.py   # TED-LIUM 3 test, STM segments → shards
│   ├── prepare_ood_commonvoice.py  # Common Voice 17 en test subset → shards
│   ├── prepare_ood_earnings22.py   # Earnings-22 subset → shards (segmentation
│   │                            #   reused from the ESB baseline release)
│   ├── check_vocab_coverage.py  # reference reachability under the pruned vocab
│   │                            #   — runs BEFORE any decoding
│   ├── run_whisper_control.py   # paired openai/whisper-small decode, greedy,
│   │                            #   same segments, same jiwer
│   ├── ood_report.py            # joins ours + control + coverage → the one table
│   └── run_ood_eval.sh          # the whole protocol for ONE checkpoint:
│                                #   preflight → smoke → decode → report → slices
│
├── configs/               # base.yaml + sparse run overrides
│                          #   (staged_full_stack.yaml, instruct-chat-3stage.yaml,
│                          #    audio_layer_adapters*.yaml, both_formats_*.yaml,
│                          #    stub_probes.yaml — chat-convention stub for
│                          #    tools/run_probes.py,
│                          #    stub_chat.yaml — 20-step chat-convention
│                          #    training stub)
├── tests/fixtures/        # stub_labels.jsonl — 64 real label records (8 runs of
│                          #   8 consecutive utterances) for make_stub_shard.py
├── data/                  # pruned_tokenizer/, shards/, dev_shards/, eval subsets,
│                          #   vocab_freq.json (rarity table for analyze_slices),
│                          #   ood/ (raw OOD downloads), ood_shards/ (OOD eval tars)
└── weights/               # whisper_small.pt, Llama3.1-8B/  (gitignored)
```

The old `train.py` / flat `scripts/` layout is gone. Some tool docstrings still
say `scripts/…` in usage examples — paths are `tools/…` now.

---

## Config System (utils/config.py)

Load order: `configs/base.yaml` → optional run YAML (deep-merged; lists replaced
wholesale) → thin CLI overrides (`--config`, `--resume`, `--wandb`/`--no-wandb`,
`--run-name`, `--checkpoint-dir`). Everything lands in frozen dataclasses;
`_validate` runs at load. **Unknown flags are an error** (`parse_args`): there is
no `--stub` or `--max-steps` flag — stub mode and step caps live in YAML
(`model.stub: true`, `run.max_steps`; see `configs/stub*.yaml`).

Key fields:

```yaml
seed: 42                       # also keys shard shuffle; CHANGE on resume for fresh data order
data:
  shards_file | shards         # training shard list file OR glob/brace pattern
  tokenizer: data/pruned_tokenizer/
  diag_shard:                  # training-time diagnostic shard (Decision 010)
  eval: {dev_clean, dev_other, test_clean, test_other}   # WER shards
model:
  stub: false                  # true → tiny random Llama (local RTX 3060)
  stub_dims: {n_layers: 6, d_model: 512, n_heads: 8, n_kv_heads: 2, intermediate_size: 1024}
  whisper_ckpt / llama_ckpt    # required when stub: false
  adapter_pca_init:            # optional PCA init .pt
  gradient_checkpointing: true          # Llama transformer layers
  encoder_gradient_checkpointing: true  # Whisper encoder blocks (no-op while frozen)
  init_from:                   # weights-only warm start (handoff ckpt / sidecar)
  bridge_type: mlp             # encoder→llama bridge: mlp | swiglu
  audio_adapter_r: 0           # gated per-layer audio adapter width; 0 = disabled
  audio_adapter_type: mlp      # mlp | swiglu
  input_convention: flat       # flat | chat (Decision 011)
  audio_adapter_zero_writer_layers: 0   # swiglu only: first N layers zero down_proj
optim:
  betas, weight_decay, grad_clip: {enabled, max_norm}
metrics:
  eval_every, eval_batches, eval_batch_size, train_periods, wer: {...}
checkpoint: {dir, save_every}
run:
  max_steps                    # global hard cap across all stages
  instruction_mode             # run default; stages may override
stages:                        # ordered list — THE training programme
  - name: ...
    trainable: [adapter]       # subset of {encoder, adapter, llama}
    lrs: {adapter: 1.0e-4}     # per-module peak LRs (must cover trainable)
    schedule: {warmup_steps: 0}
    batch_size, accum_steps
    instruction_mode:          # optional stage override
    optimizer_init: fresh      # "inherit" raises NotImplementedError
    exit: {strategy, threshold, min_steps}
      # strategies: first_token_below | eval_loss_below | max_steps | epochs | never
      # epochs: threshold = completed epochs (int >= 1), checked at the epoch
      #         boundary before the next loader is built
```

Shard shuffle per epoch: `random.Random(cfg.seed + stage_idx*10000 + epoch)`.
WebDataset loaders are FINITE (v1.0.2, resampled=False) — one pass per epoch,
the epoch loop rebuilds the loader.

---

## Component Interfaces (essentials)

```python
# model/whisper_encoder.py
WhisperEncoder.forward(mel: (B, 80, T_mel)) -> (B, T_mel//2, 768)
WhisperEncoder.load_openai_weights(path)      # shape-asserted, fail-loud
log_mel_spectrogram(audio: (T,)) -> (80, T//160)   # no padding/trimming here

# model/adapter.py
AudioAdapter(llama_dim=4096, pca_init_path=None)      # also AudioSwiGLUBridge
AudioAdapter.forward((B, T, 768)) -> (B, ceil(T/4), llama_dim)
.audio_bos / .audio_eos                               # learned (d_model,) markers

# model/sequence.py
prepare_input(...)      -> (inputs, labels)                  # flat, unchanged
prepare_input_chat(..., chat, audio_bos, audio_eos)
    -> (inputs, labels, audio_mask (B, L, 1))
assemble_inputs(..., terminator_id, chat=None, audio_bos=None, audio_eos=None)
    -> (inputs, labels, audio_mask | None)   # chat=None -> flat, mask None
ChatTemplate.from_tokenizer(tokenizer)       # .audio_offset, .eot_token_id
EvalPrefixBatch(..., chat=, audio_bos=, audio_eos=)   # .audio_mask grows with gen
                       # batched generation; inserts tokens at gen_pos, never
                       # exposes padding to causal attention history
ChatTemplate.seg_user_continuation           # user opener for turns 2..N: the
                       # same header with [bos] swapped for the [eot] that
                       # closes the assistant turn before it (EVAL ONLY)
ConversationPrefix(chat, embed, audio_bos, audio_eos)     # multi-turn, EVAL ONLY
    .add_user_turn(audio_emb, instruction_ids) / .add_assistant_text(ids)
    .embeddings / .audio_flags / .trace / .render(decode)
ConversationPrefixBatch(prefixes, audio_flags)   # EvalPrefixBatch's multi-turn
                       # sibling: same padding + insert-at-gen_pos discipline,
                       # but handed finished prefixes (a conversation's prefix
                       # accumulates and cannot be rederived from one utterance)

# model/llama.py
Llama(LlamaConfig).forward(inputs_embeds, labels=None, audio_lengths=None,
                           audio_mask=None) -> (logits, loss|None)
    # audio_mask wins over audio_lengths when both are given
Llama.load_meta_weights(dir, vocab_map=None)   # HF safetensors or Meta .pth
Llama.enable_gradient_checkpointing()

# build.py
build_models(cfg, device, train=True, apply_init_from=True)
    -> (encoder, adapter, llama, init_from_loaded)
    # arch → pretrained → init_from overlay; prints which chat-special embedding
    # rows came from the checkpoint (Decision 011 depends on those being real)

# stages.py
Stage(stage_cfg, ctx).setup(encoder, adapter, llama) -> (optimizer, scheduler)
Stage.make_loader(epoch, stage_idx) -> DataLoader
Stage.should_advance(metrics, step_in_stage, epoch) -> bool

# data.py
build_dataloader(...)              # training loader (finite; caller shuffles shards)
read_eval_shard(shard_path) -> list[EvalSample]   # one tarfile read, member
    # order; the shared front end of the WER loader and run_longform.py
build_sorted_eval_dataloader(...)  # WER loader: tarfile, sorted, length-filtered
PrunedTokenizer(tokenizer_path).encode/.decode
    # + .vocab_size/.terminator_id/.bos_token_id/.start_header_id/
    #   .end_header_id/.eot_token_id  (what ChatTemplate.from_tokenizer reads)
load_pruned_config(tokenizer_path) -> PrunedConfig   # the pruned_config.json reader
INSTRUCTION_VARIANTS               # canonical two prompt strings

# utils/checkpoint.py
save_checkpoint / save_adapter_checkpoint     # kind="periodic" | "handoff"
read_checkpoint / apply_full_checkpoint -> ResumeState
load_weights(path, ...) == apply_weights(read_checkpoint(path), ...)
    # use the two-call form when the metadata is wanted too — one read

# utils/generate.py + utils/evaluate.py
greedy_generate(..., stop_token_id, chat=None) -> list[list[int]]
    # batched; stops on the terminator (SEP flat / <|eot_id|> chat)
greedy_generate_from_prefix(llama, pfx, stop_token_id, max_new_tokens,
                            audio_lengths=None) -> list[list[int]]
    # the decoding loop itself; greedy_generate and tools/run_longform.py both
    # call it, so single-turn and multi-turn generation cannot drift
evaluate_all_splits(...) -> (wer_dict, sample_rows)   # no text normalisation
    # sample_rows: {key, split, type, reference, hypothesis}, capped at n_samples
    # return_all_transcriptions=True adds a 3rd return, every evaluated pair
```

Batch tuple from build_dataloader (pass straight to prepare_input):
```
(mel (B,80,T_max), audio_lengths (B,), instruction_ids, instruction_lengths,
 transcript_ids, transcript_lengths)
```

---

## Diagnostics (metrics.py)

MetricCollector routes observations by input family (logits / grads / loss /
generation) and emits on per-metric periods; `render()` prints, the collector
never prints. Signals: token budget, first-token vs rest loss decomposition,
healthy/collapse flag, per-module grad norms (enc/ada/llm), first-tok top-5,
terminator fraction, normalised entropy, mean max logit. (The W&B key keeps its
`collapse/*_sep_fraction` name for dashboard continuity; under the chat
convention it counts `<|eot_id|>`. Entropy is normalised by log(V) read from
pruned_config at run time, so a vocabulary rebuild does not silently rescale
it.) Full interpretation guide,
healthy ranges, and baseline ladder live in TRAINING_DIAGNOSTICS_REFERENCE.md
(project knowledge base); generated-output failure modes are catalogued in
DEGENERACY_TAXONOMY.md. Baselines per corpus come from tools/compute_baselines.py and
are logged as flat W&B reference lines.

Key milestone semantics: eval first-token loss below the unigram baseline is
the "model uses audio" signal — the central open metric at small scale.

---

## Data Pipeline Execution Order

```
1. tools/download_data.py       # LibriSpeech splits (+ eval splits if needed)
2. tools/label_formatted.py     # LLM labelling run → labels.jsonl  (or
                                #   tools/precompute_labels.py for the old
                                #   BasicTextNormalizer/spaCy pipeline)
3. tools/build_vocab.py         # --labels_file labels.jsonl → pruned tokenizer
4. tools/preprocess.py          # mel + labels → shards (fast with --labels_file)
5. tools/create_subset.py / make_dev_dataset.py / make_eval_subset.py  # as needed
```

Labels come FIRST: the vocabulary is derived from the label text (Decision 005),
and sharding is minutes with `--labels_file` against ~1 hr/split inline.

**Relabelling an existing corpus does not re-run this pipeline.** The mel is a
pure function of the audio, so tools/relabel_shards.py streams the existing
shards, swaps the two label members, drops `validation: failed` samples, and
rewrites manifest.jsonl — minutes of I/O against hours of mel recompute. It
preserves shard filenames and shard membership, so shard-list files stay valid.
Rebuild the vocabulary afterwards, from the same labels.jsonl. Both
tools/preprocess.py and tools/relabel_shards.py drop failed records, so the two
paths produce the same corpus.

---

## Conventions

- Python 3.11, type hints and docstrings everywhere; comments explain WHY.
- No HuggingFace transformers in model code — permitted only in data.py
  (tokenizer) and tools/ (normalizers, PunctuationModel).
- Fail loudly on shape mismatches; assert before every `copy_`.
- No hardcoded paths or secrets; pathlib + CLI flags + env vars.
- Tools are independently readable; duplication across tools is acceptable.
- Self-tests live in `if __name__ == "__main__"` blocks (see stages.py,
  make_eval_subset.py) — extend them when touching those files; keep them
  CPU-runnable with mocks.
- Container-first: code must run identically in the venv and on RunPod/GCP.
- Readable over clever (e.g. the additive shuffle-seed formula was chosen over
  hashing for legibility).

---

## Current Status

| Area | Status |
|---|---|
| Model components (encoder / adapter / llama / sequence) | ✅ Complete |
| Config-driven multi-stage training loop (training.py) | ✅ Complete |
| Resume/checkpoint semantics (kind tagging, init_from) | ✅ Complete |
| Metrics + diagnostics + baselines | ✅ Complete |
| WER engine + run_wer.py sweep tool | ✅ Complete |
| Transcription dumps + degeneracy / A-B tooling | ✅ Complete |
| Data pipeline + full-corpus shards on GCS | ✅ Complete |
| Batch-size sweep tooling (find_max_batch, probe_accum) | ✅ Complete |
| Experiment configs (both_formats_*) | ✅ Authored |
| Pre-refactor ckpt resumed with post-refactor code | ✅ Complete |
| Gated audio-layer adapters inside Llama | ✅ Complete (C-series banked) |
| Instruct backbone + chat convention (Decision 011) | 🔨 Run complete @15040; terminator regression open |
| Eval slicing / degeneracy tooling (analyze_slices) | ✅ Complete, D1 + instruct-chat backfilled |
| Long-form multi-turn eval harness (run_longform) | ✅ Complete; E1 @17280 run banked — Finding E-iv (capability absent) |
| OOD protocol: shards, coverage, Whisper-small control | ✅ Complete (TED-LIUM 3, Common Voice, Earnings-22) |
| OOD protocol: our model's decode | ⛔ Blocked — no chat checkpoint survives (see below) |
| Instruction-generalisation probe suite (P1-P5) | ✅ Complete — run banked @15040; Findings E-v … E-viii |
| fp16 master-weights underflow experiment | ⏸ Deferred (post-single-GPU) |
| Multi-GPU / multi-node | ⏸ Deferred |

---

## Active Task — Instruct Backbone + Chat Convention

Decision 011's code is on master (self-tested; flat path verified unchanged
per-utterance against the banked D1 transcriptions). What remains is the run:

1. `python tools/build_vocab.py --labels_file data/labels.jsonl --llama_dir
   <Instruct dir> --output_dir data/pruned_tokenizer_chat/`
   — writes the four `*_id` keys and verifies the splice boundaries. Expect
   40,039 tokens and eot → 40,038. A Meta-format `--llama_dir` prints a notice
   that the `apply_chat_template` layout cross-check was skipped; that is normal
   and the build is still verified (see 011).
2. `python tools/compute_baselines.py --tokenizer data/pruned_tokenizer_chat/ …`
   — V moved 40,034 → 40,039, so the old `baselines.json` reference lines are wrong.
3. `python training.py --config configs/stub_chat.yaml` (local RTX 3060) — a
   20-step stub run with this line's chat convention / swiglu bridge / adapters.
4. `python tools/find_max_batch.py --config configs/instruct-chat-3stage.yaml
   --stage 3` — stage 3's `batch_size: 64 / accum_steps: 2` is an UNTESTED
   placeholder; the chat scaffold lengthens every sequence. find_max_batch
   assembles under the config's own `input_convention`, so its number is for the
   sequence this run actually builds.
5. Short real-GPU run, then the full launch (3 × 4680 steps).
6. Results: `run_wer.py --splits dev-clean dev-other --full` over the stage
   handoffs, then `count_degeneracies.py` + `analyze_slices.py` over the emitted
   JSONLs and `plot_wer.py` over the CSV.

**Outcome of that comparison (step 15040, all three stages complete): the
`<|eot_id|>` termination prior is WORSE than the accidental `锦`, not better.**
Final WER 9.14/12.76 (dev-clean un/formatted) and 16.43/20.05 (dev-other) vs D1's
5.09/9.16 and 11.92/16.38 — but the aggregate hides the shape, and the shape is
two opposite terminator failures:

- **Premature stop, formatted only:** 18/479 dev-clean and 22/480 dev-other empty
  hypotheses (3.8% / 4.6%). D1 had ZERO empties anywhere. Zero in unformatted.
- **Failure to stop, short utterances:** overrun (hyp > 1.3x ref) is 13-14% in the
  0-8 word bucket and decays monotonically to <1% at 17+ words. The model
  transcribes correctly, then rambles or restarts instead of emitting `<|eot_id|>`
  ("illustration the lemon the lemon the lemon the lemon."). 0-8 bucket WER is
  40.79 dev-clean vs D1's 17.09.
- **Adherence regression:** unformatted style-flip 81/479 (16.9%) vs D1's 8/479
  (1.7%), and it is 71/81 a trailing terminal period on otherwise-correct words —
  the Instruct backbone will not suppress sentence-final punctuation. Costs only
  ~0.6 WER overall but ~6 points in the 0-8 bucket, where one period is a large
  fraction of a short reference.

Numeral format-switching did improve slightly (dev-clean 11/23 -> 13/23 clean
switches; dev-other flat at 8/11) — small n, directional only. So the chat
convention helps instructability at the margin while the terminator regresses
badly; the next move is the termination prior, not the scaffold.

Artifacts: `results/instruct-chat/` (fetched from the pod) and `out/instruct-*`.

Do NOT rebuild `data/pruned_tokenizer/` in place — the flat line still reads it.

---

## Long-Form Eval Protocol — transcription across chat turns

Eval-only, zero-shot: the model is trained SINGLE-TURN, and this asks what it
does when audio arrives as successive user turns and the transcript continues
across assistant turns. `tools/run_longform.py` is the whole harness; nothing in
training changes.

**Conversations.** LibriSpeech utterance ids encode reading order
(`speaker-chapter-utterance`). Utterances are grouped by (speaker, chapter),
sorted by index, and cut into runs of `--turns` CONSECUTIVE utterances. A run
that would span a gap never forms — a gap means the "conversation" would jump
forward in the book mid-context. Selection is deterministic: groups in
(speaker, chapter) order, runs in reading order, first `--max-convs` taken. No
RNG anywhere.

The per-utterance ≤41-token reference filter runs FIRST, so per-utterance WER
stays comparable with the single-turn sweeps — and it is what punches most of
the holes that then drop runs. Measured on the full split shards at
`--turns 6`:

| split | utterances | dropped ≤41 | groups | candidate runs | consecutive | dropped for gaps |
|---|---:|---:|---:|---:|---:|---:|
| dev-clean | 2,693 | 296 | 97 | 359 | 275 | 84 |
| dev-other | 2,855 | 212 | 91 | 404 | 338 | 66 |

⚠ **Point `--shard` at the FULL split shards** (`data/eval_shards/dev-clean-000000.tar`),
not `data/eval_subset_shards/*-480.tar`: those are random 480-sample draws and
contain almost no consecutive runs. The tool warns and then errors on zero
conversations rather than reporting a number from three conversations.

⚠ **The literal "first N runs" rule concentrates speakers.** 100 conversations
comes from 15 dev-clean speakers (of 40) and 10 dev-other speakers (of 33).
`--max-runs-per-group 2` widens that to 30 / 22 speakers at the same n, and is
the better setting if the claim involves speaker breadth. Default stays 0
(unlimited) so the documented selection is the literal one.

**Sequence.** `ConversationPrefix` (model/sequence.py) composes ChatTemplate's
OWN segments, so the multi-turn layout cannot drift from the trained one:

```
[bos][start_header]"user"[end_header]"\n\n" ▸BOS [audio_1] ▸EOS "\n" {instruction}
[eot][start_header]"assistant"[end_header]"\n\n" {transcript_1}
[eot][start_header]"user"[end_header]"\n\n" ▸BOS [audio_2] ▸EOS "\n" {instruction}
[eot][start_header]"assistant"[end_header]"\n\n" {transcript_2}   …
```

- **The instruction repeats in every user turn**, after that turn's audio —
  exactly where the single-turn convention puts it (Decision 011: audio first,
  instruction last). It is NOT hoisted to a system turn; this convention emits
  no system block at all.
- The `[eot]` closing each assistant turn is the FIRST id of the next turn's
  `seg_user_continuation`, mirroring `seg_pre_transcript`'s leading `[eot]`.
  A conversation's prefix therefore ends with no trailing `[eot]` — the model
  must generate the final one, which is a large part of what this measures.
- **Turn 1 is byte-identical to the single-turn prefix.** Asserted in
  `model/sequence.py`'s self-test against `EvalPrefixBatch` (embeddings AND
  audio mask), and confirmed on real audio: turn-1 hypotheses match across all
  three context modes. If turn 1 differed, every later turn would be measuring
  the harness rather than the model.
- The audio mask is now several disjoint spans, one per turn, and the in-layer
  audio adapters gate on it. That is why the stub config sets
  `audio_adapter_r: 16` — with the adapters off, nothing reads the mask.

**Three context modes**, all decoded in the same run over the same conversations:

| mode | previous assistant turns hold | isolates |
|---|---|---|
| `self` | the model's own generated transcript | the realistic condition; errors propagate |
| `oracle` | the gold transcript (teacher-forced) | whether context HELPS, independent of error accumulation |
| `single` | nothing — each utterance decoded alone | the matched control |

`single` is decoded HERE, not lifted from a `run_wer.py` sweep, and that is
load-bearing. Batches here are formed by conversation, not by ascending audio
length; mel is padded per batch and the Whisper encoder attends bidirectionally
ACROSS that padding, so batch composition perturbs the audio embeddings. The
control has to sit in the same batches, meet the same generation cap
(longest reference in the batch + `--max-new-slack`), and run through the same
untouched `greedy_generate` / `EvalPrefixBatch` path. Compare within this
file's output; compare to a WER sweep only at the order-of-magnitude level.

**Context guard.** The worst-case assembled length (every assistant turn
budgeted at its own generation cap) is computed for every selected conversation
and checked against `llama.config.max_seq_len` before any decoding, because
overrunning it would not error — the RoPE tables reach far past it and attention
would simply run. At 6 turns the worst case is ~1,364 tokens against a 131,072
context, so the guard is there for a `--turns 40` run, not this one.

**Analysis.** The JSONL carries run_wer.py's field names plus `conv_id`,
`turn_index` (1-based) and `context_mode`, so the existing tools read it as-is:

```bash
python tools/analyze_slices.py --jsonl out/longform-dev-clean.jsonl \
    --group-by turn_index --out-md out/longform-dev-clean-by-turn.md \
    --out-json out/longform-dev-clean-by-turn.json
python tools/analyze_slices.py --jsonl out/longform-dev-clean.jsonl \
    --group-by context_mode --out-md out/longform-dev-clean-by-ctx.md
```

`--group-by turn_index` on the combined file mixes the three context modes;
for the per-mode WER-and-degeneracy-by-turn curves, split the JSONL by
`context_mode` first and run it on each.

### Result: the capability is absent (Finding E-iv)

Run on the pod against `checkpoints/instruct-chat-3stage/step0017280.pt`,
10 conversations × 6 turns × 2 formats × 3 context modes per split. Artefacts,
tables and the verbatim gallery: `results/longform/` (see its README).

| split | mode | t1 | t2 | t3 | t4 | t5 | t6 | all |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dev-clean | single | 4.1 | 4.3 | 9.5 | 4.3 | 5.2 | 6.4 | **5.7** |
| dev-clean | self | 4.1 | 116.4 | 95.2 | 96.3 | 98.5 | 100.0 | **85.2** |
| dev-clean | oracle | 4.1 | 115.7 | 94.2 | 96.8 | 100.0 | 100.0 | **85.3** |

- **Not error propagation** — `oracle` collapses identically to `self`.
- **Not a harness bug** — `single` uses the identical mel batch at every turn and
  scores 4–14%.
- **The model re-decodes audio block #1** at turn 2 rather than the current turn,
  then decays into silence (17 → 8 → 4 → 2 → 1 → 0 mean output words;
  `del_frac_of_err` 0.85 against single's 0.06).

Suspected cause: the constant audio offset. The model learned "transcribe the
audio at position 6", never "transcribe the most recent audio". A **text-only
history ablation** would separate that from the terminator-prior explanation
(EXPERIMENT_LOG.md backlog item 9); a re-run at step 15040 with `--max-convs 100`
would remove the post-peak and small-n caveats (item 10).

`data/pruned_tokenizer_chat/` was built locally as part of this work, from
`data/labels.jsonl` and the Meta tokenizer in `data/pruned_tokenizer/` (base and
Instruct share a tokenizer). It reproduces the documented numbers — 40,039
tokens, `<|eot_id|>` → 40,038 — and matches the pod's, which the run confirms.

## Instruction-Generalisation Probe Suite

Eval-only, zero-shot, Speech-IFEval style: every probe is an **output constraint
that does not depend on what the audio says**, so instruction-following is
measured separately from perception. A model can transcribe perfectly and still
fail all of them. Training only ever saw the two strings in
`data.INSTRUCTION_VARIANTS`, in one fixed single-turn layout — none of these
instructions, output formats or question texts appears anywhere in it.

Three tools plus a shared spec:

```
tools/probe_spec.py               every prompt string, exactly once (stdlib)
tools/check_vocab_feasibility.py  THE GATE — run first, before any decoding
tools/run_probes.py               generation → one JSONL per probe
tools/score_probes.py             scoring → one Markdown report (stdlib)
```

| probe | asks | pass criterion |
|---|---|---|
| P1 uppercase | transcribe in ALL CAPITALS | ≥95% of alphabetic chars upper case |
| P2 word_count | transcript, then the word count on a new line, as digits | the number equals the word count of the model's OWN transcript |
| P3 last_word | answer with only the last word spoken | exact match (§6-normalized) to the reference's last word |
| P4 ignore_audio | ignore the audio; answer one of 10 fixed questions in one word | exact match, **plus a text-only control** |
| P5 prompt_variation | the trained task in 3 unseen paraphrases per mode | per-variant WER and adherence, against the canonical string |

Two numbers per probe, and they come apart in the informative direction:
**Following Rate** (the headline) is the fraction of outputs that are even the
right *kind* of response; **pass rate** is the fraction that additionally got it
right. A model answering "the last word is today" has a P3 Following Rate of 0
even when the word is correct.

### The feasibility gate is load-bearing, and it already found something

The vocabulary is pruned to the LibriSpeech label ids (Decision 005), so a probe
whose expected output falls outside it is impossible **by construction** and
scoring it would report a property of `tools/build_vocab.py` as a failure of
instruction-following. `check_vocab_feasibility.py` decides this first and writes
a gate JSON; `run_probes.py` refuses to run a probe the gate has not cleared
(`--force` overrides and stamps every affected row `gate_forced: true`, so a
forced number can never later be mistaken for a clean one).

Measured against `data/pruned_tokenizer_chat/` (40,039 tokens):

| candidate group | gates | fully reachable | verdict |
|---|---|---:|---|
| uppercase_transcripts (50 real dev-clean refs, upper-cased) | P1 | **0.0%** | FAIL |
| probe_prompts (21 instruction strings) | all | 100% | pass |
| digits_0_99 (bare and newline-prefixed) | P2, P4 | 100% | pass |
| p4_answers | P4 | 100% | pass |

- **P1 is DROPPED: upper case is unreachable corpus-wide.** Not one of 50
  upper-cased references survives — `HELLO` breaks into `H`+`ELLO`, and every
  such subword was pruned away. That is a finding about the pruning decision,
  not about the model: the vocabulary was built from lower-cased and
  sentence-cased audiobook text, so an all-caps constraint cannot be asked of
  this model at all. It is also why P4's accepted answers deliberately exclude
  all-caps forms.
- **The prompt side matters more than the output side.** `PrunedTokenizer.encode`
  drops unmapped ids silently, so an unreachable word in an *instruction* raises
  nothing — the model just receives the prompt with words deleted. The gate
  caught exactly this: `" transcription"` is unreachable, so P2's wording is
  `"...the number of words in your transcript..."`. `" transcript"` and
  `" verbatim"` are fine. **The probe wording is dictated by the gate, not by
  taste** — re-run it after editing any prompt in `probe_spec.py`.

### Load-bearing design decisions

- **One utterance sample drives every probe, in identical batches.** The Whisper
  encoder attends bidirectionally across mel padding, so batch composition
  perturbs the audio embeddings. Probes compared with each other — and P5's
  paraphrases compared with the canonical string — must sit in the same batches.
- **P5 decodes the canonical instruction HERE** rather than lifting a number
  from a banked WER sweep, for that same reason. The spread across wordings is
  then prompt sensitivity and nothing else. Spread is computed *within* a mode:
  the formatted task is harder by design, and pooling the two would report that
  difference as prompt sensitivity.
- **P4's text-only control is the leakage measurement.** The same 10 questions
  are asked with no audio block in the context at all — `seg_pre_audio` +
  `"\n"` + instruction + `seg_pre_transcript`, the trained user turn minus
  exactly the markers and the audio, and nothing else. The backbone is frozen,
  so the text-only rate is what this LLM does with the question alone and the
  drop under audio is what the audio conditioning costs. Batched through
  `ConversationPrefixBatch` (its audio mask is all zeros, which is correct —
  there is no audio for the in-layer adapters to fire on). It is 1 generation
  per question, so a single question moves that rate 10 points; the report says
  so.
- **P3's generation cap is the FULL transcript length + slack, not one word.**
  A model that dumps the whole transcript instead of answering must be visible
  as a dump, not truncated into looking like a near-miss. Its Following Rate is
  "short AND not a transcript dump" — a short utterance's whole transcript can
  fit inside the word limit, and counting that as compliance would hide the
  collapse the probe exists to catch.
- **P2's pass is SELF-consistency**, because that is the instruction as written
  ("the number of words in your transcript"). Agreement with the reference's
  count folds in transcription errors and is reported separately.
- **Scoring imports WER, normalisation and the adherence rule from
  `analyze_slices.py`** (and the degeneracy verdicts from
  `count_degeneracies.py`) rather than reimplementing them, so P5's canonical row
  is directly comparable with the banked slice reports. WER is raw-token and
  UN-normalized, the house convention.
- **`--max-ref-tokens 41`** by default, the same filter the WER sweeps and
  `run_longform.py` apply, so a WER measured here is comparable with the banked
  numbers.

### Results (step 15040, dev-clean, 200 utterances / 37 speakers, 2,210 generations)

Artefacts: `results/probes/` (see its README). P1 dropped by the gate; P2-P5 run.

| probe | n | Following Rate | pass rate | headline |
|---|---:|---:|---:|---|
| P2 word_count | 200 | **0.0%** | 0.0% | 0/200 emitted a newline, let alone a count |
| P3 last_word | 200 | **0.0%** | 0.0% | 100% dumped the transcript; 0% answered with a word |
| P4 ignore_audio | 200 | 3.5% | **0.0%** | ≥49% transcribed the audio instead; control unusable |
| P5 prompt_variation | 1,600 | 97.1% | 71.5% | unformatted WER 6.94→20.18 across wordings; formatted flat |

**Finding E-v — instruction-following outside the two trained strings is absent.**
Not weak: absent. Every unseen instruction collapses to "transcribe the audio".
P2 asked for a transcript plus a count line and got 0/200 newlines. P3 asked for
one word and got the full transcript 200/200 times. P4 asked ten general-knowledge
questions over 200 clips and got 0 answers — it transcribed the audio instead
(49% an exact normalised match to the reference, the rest transcripts carrying
ordinary ASR errors, so 49% is a FLOOR on the collapse rate, not the rate).

**Finding E-vi — the instruction functions as a two-way style switch, not as an
instruction.** The collapse is not to a single default: an unseen instruction is
snapped to one of the two TRAINED modes, and which one tracks surface cues in the
prompt rather than its meaning.

| probe | instruction contains | formatted-style output |
|---|---|---:|
| P2 | "…as **digits**" (a formatted-canonical cue) | 94.5% |
| P4 | no cue from either trained string | 90.5% |
| P3 | "…last word **spoken**" (an unformatted-canonical cue) | 24.0% |

⚠ This is a pattern across three instructions, i.e. n=1 per condition — suggestive,
not established. The cheap test is an ablation that varies one cue word at a time
in an otherwise fixed instruction; nothing in the harness needs changing to run it.

**Finding E-vii — paraphrase robustness is strongly asymmetric, and the
asymmetry is informative.** P5 held the task constant and varied only the wording:

| mode | canonical | worst paraphrase | WER spread | adherence range |
|---|---:|---:|---:|---|
| formatted | 10.79 | 10.79 | **0.77** | 96.5-97.0% |
| unformatted | 6.94 | 20.18 (`u2`) | **13.24** | **1.0-90.0%** |

Formatted output survives every paraphrase; unformatted collapses under two of
three. Adherence goes 90.0% (canonical) and 89.5% (`u3`) → 4.0% (`u1`) → 1.0%
(`u2`) — the model reverts to capitalised, punctuated text, i.e. to the formatted
default. All three paraphrases contain the words "lowercase" and "punctuation",
so this is not a vocabulary-coverage effect (the gate cleared all of them).

The reading: the model has ONE default behaviour — formatted transcription, which
is also what a Llama Instruct backbone does with speech-like text unprompted —
and ONE learned deviation from it. Paraphrasing the instruction that requests the
deviation loses the deviation; paraphrasing the instruction that requests the
default costs nothing, because falling back lands in the same place. **Formatted
mode's apparent robustness is therefore not evidence of instruction-following at
all**, and reporting the two modes' spreads together would have hidden that.

**Finding E-viii — the frozen backbone does not produce coherent text without
audio in the context.** P4's text-only control (the trained user turn minus
exactly the markers and the audio block) came back 10/10 degenerate loops:
"Understanding friction simmer simmer simmer … mushrooms mushrooms". So:

- The intended leakage measurement is **void** for this run — a gap against a
  broken control is not a measurement, and `score_probes.py` now detects the
  degeneracy and refuses to report a gap rather than printing "+0.0 points"
  (`CONTROL_MAX_DEGENERATE`).
- It is a finding in its own right, and it converges with E-iv (multi-turn: the
  model re-decodes audio block #1, then decays into silence). Both say the model
  is welded to "there is audio, at this offset". The audio positions are doing
  work beyond carrying content.
- What it does NOT show is that the pruned Instruct backbone cannot answer these
  questions; that has not been tested. The clean control is the same ten
  questions through the SAME pruned vocabulary and scaffold on the untrained
  base model — one cheap pod job, and the obvious next step if P4 is to be
  quoted as a leakage number.

None of this contradicts the WER story: at the same checkpoint the model
transcribes dev-clean at 6.94/10.79 WER under its two trained prompts. It is a
good ASR model that is not, in any general sense, instructable — which is
precisely the distinction the suite was built to draw, and the reason the probes
are scored separately from WER.

### Running it

```bash
# 0. the gate — FIRST, and again after any prompt edit
python tools/check_vocab_feasibility.py \
    --tokenizer data/pruned_tokenizer_chat/ \
    --shard data/eval_subset_shards/dev-clean-480.tar \
    --out-json out/probe-feasibility.json --out-md out/probe-feasibility.md

# 1. local stub sanity — prints every prompt, scaffold specials and all
python tools/run_probes.py --config configs/stub_probes.yaml \
    --shard data_stub/stub-diag-000000.tar \
    --feasibility out/probe-feasibility.json \
    --probes P3 P4 --n 5 --print-prompts 1 --decode-anyway \
    --out-dir out/probes-stub

# 2. the real run (pod)
python tools/run_probes.py --config configs/instruct-chat-3stage.yaml \
    --checkpoint checkpoints/instruct-chat-3stage/step0015040.pt \
    --shard data/eval_shards/dev-clean-000000.tar \
    --feasibility out/probe-feasibility.json --out-dir results/probes/

# 3. the report
python tools/score_probes.py --dir results/probes/ \
    --out-md results/probes/probe-report.md \
    --out-json results/probes/probe-report.json
```

Cost at the default `--n 200`, with P1 dropped: 200 (P2) + 200 (P3) + 200 + 10
(P4) + 1,600 (P5, 8 variants × 200) ≈ **2,210 generations** — P5 dominates, and
`--n` scales the whole suite. Well under one WER-sweep cell.

⚠ **Step 15040 is the endpoint for this line, not 17280** (Finding E-iii:
training past the handoff makes it worse). Point `--checkpoint` there.

Reporting rules the tools enforce: every probe is stated as zero-shot; the gate's
outcomes including any dropped probe are printed in the report; and the probes
are never aggregated into one number — the per-probe rates are the result.

## Out-of-Distribution Eval Protocol

Eval-only. Three test sets, one axis each, all decoded through the existing WER
engine (`run_wer.py --eval-tar NAME=PATH --dataset TAG`) — no second decode path
exists, so an OOD number and an in-domain number come from the same code.

| set | axis | n | hours | reference forms |
|---|---|---:|---:|---|
| TED-LIUM 3 test | read → spontaneous lecture speech | 998 (≤41 tok) / 1,139 (≤64) | 2.0 / 2.5 | verbatim only |
| Common Voice 17 en test | accent + channel diversity (CC-0) | 5,000 | 8.2 | verbatim + formatted |
| Earnings-22 subset | far-domain real-world audio | 1,789 | 3.1 | verbatim + formatted |

**The reported number is the Δ against `openai/whisper-small` (encoder+decoder,
greedy, no LM) run on the same segments under the same scoring.** An absolute WER
on this project's un-normalized house convention, its own references and a
length-filtered subset is not comparable to anything published, and the report
does not pretend otherwise.

### Load-bearing decisions

- **The control decodes the same audio, not the same mel.** Every OOD shard
  carries a 4th member, `{key}.flac` (`tools/ood_shard.py`), so
  `run_whisper_control.py` runs Whisper's own front end over byte-identical
  segments. Reusing our natural-length mel would put Whisper slightly off its
  training distribution and understate the control — a bias in the flattering
  direction.
- **Verbatim pairing normalises BOTH sides with FORMATTING_SPEC §6; formatted
  pairing normalises neither.** Whisper always writes cased, punctuated text, so
  scoring it raw against a verbatim reference measures punctuation, and
  normalising only the control hands it a free pass we pay for. §6 on both sides
  of both systems is the only symmetric arrangement. It is worth ~3 WER on
  TED-LIUM: the house verbatim reference keeps apostrophes (`it's`) and §6
  deletes them, so a hypothesis-only normalisation charges one error per
  correctly transcribed contraction. The un-normalized house numbers are kept in
  `ood_report.py`'s JSON under `house_convention`.
- **§6 does not expand digits, so every verbatim row is scored twice.** A control
  writing `60,000` where the reference says `sixty thousand` takes three errors
  it did not earn — an asymmetry that favours US, since a model trained on this
  convention writes the words. `ood_report.py` re-scores the digit-free subset,
  excluding the same utterances from both systems. Earnings-22 is 22%
  digit-bearing; read its digit-free row.
- **Coverage is measured before any decoding and reported next to every WER.**
  `check_vocab_coverage.py` encodes each reference with the FULL tokenizer and
  tests id survival in `vocab_map.json`. An unreachable word is a guaranteed
  error that measures Decision 005, not the model. A row whose pruning floor is
  ≥25% of its Δ gets flagged in the report.
- **TED-LIUM has no formatted reference.** Its `formatted.txt` is a copy of the
  verbatim text so the three-member house format holds; the manifest says
  `has_formatted_ref: false` and it must be decoded `--formats unformatted`.
- **Common Voice references are PROMPT text**, not transcripts — they diverge on
  misreads, and both systems pay that floor. CV also has ZERO digit-bearing
  sentences, so it does not exercise the §4 numeral rule at all.
- **Earnings-22 needed no aligner.** Its segmentation is reused from
  `anton-l/earnings22_baseline_5_gram` (the source the official loader itself
  pulls): per-segment wavs plus start/end timestamps. The subset samples 40 of
  125 calls, which samples the corpus's accent breadth too — widen `--sources`
  if the accent axis is the claim.
- **Earnings-22 references carry two things LibriSpeech labels never did:**
  filled pauses ("um"/"uh", 28.7% of segments) and `<inaudible>` markers (6.6%).
  Both are a reference-convention floor, not an ASR error — a model trained on
  read audiobooks will never emit either, and most ASR systems suppress fillers
  too, so the control pays part of the same floor. They are FLAGGED per segment
  (`has_disfluency`, `has_inaudible`), never cleaned out: rewriting a human
  transcript to flatter the systems under test is exactly the wrong move.

### Measured coverage (`data/pruned_tokenizer/`, 40,034 tokens)

| set | ref form | utterances fully reachable | words reachable |
|---|---|---:|---:|
| LibriSpeech dev-clean-480 | verbatim | 100.00% | 100.000% |
| LibriSpeech dev-other-480 | verbatim | 100.00% | 100.000% |
| TED-LIUM 3 test | verbatim | 80.33% | 98.959% |
| Common Voice en test | verbatim | 89.16% | 98.749% |
| Common Voice en test | formatted | 79.62% | 97.053% |
| Earnings-22 subset | verbatim | 70.43% | 97.483% |
| Earnings-22 subset | formatted | 49.69% | 94.894% |

In domain the pruning is free; out of domain one utterance in five (TED-LIUM) to
one in two (Earnings-22 formatted) contains a word the model **cannot emit at
all**. The top offenders name the cause — the vocabulary was frozen on
19th-century public-domain audiobooks: `online`, `gamers`, `wikipedia`,
`sustainable`, `digital`, `software`, `internet`, `2021`, `2022`, `covid`,
`cybersecurity`. This is the single most important number in the OOD report and
it qualifies every WER beside it. `check_vocab_coverage.py --out-examples` makes
it concrete by round-tripping references through the map: "using websites blogs
and online videos to demonstrate…" has a ceiling of "using and to demonstrate…".

### Banked control numbers (openai/whisper-small, greedy, no LM)

| set | mode | n | control WER | digit-free |
|---|---|---:|---:|---:|
| tedlium3-test-le41 | verbatim | 998 | 6.07% | 4.69% |
| tedlium3-test-le64 | verbatim | 1,139 | 5.91% | 4.59% |
| commonvoice-en-test | verbatim | 5,000 | 16.48% | 15.97% |
| commonvoice-en-test | formatted | 5,000 | 21.72% | — |
| earnings22 | verbatim | 1,789 | 20.77% | 21.81% |
| earnings22 | formatted | 1,789 | 28.23% | — |

The digit-free TED-LIUM and Common Voice figures land within a point of published
whisper-small numbers for those corpora — the harness measures what it thinks it
does. Earnings-22 moves the OTHER way digit-free (20.77 → 21.81): numbers in an
earnings call are enunciated carefully, so the digit-bearing segments are the
easy ones there.

Control degeneracy (tools/count_degeneracies.py over the same JSONLs) is the
graceful-degradation baseline our model will be read against: TED-LIUM 0.8%
truncation / 0.1% hallucination, Common Voice 1.0% / 1.7%, Earnings-22 8.2% /
4.1%. Zero empty hypotheses anywhere.

### Status: control complete, our decode blocked

The Whisper-small control, the shards and the coverage are done and banked.
**Our model's OOD decode has not run: no `instruct-chat` checkpoint is reachable
from this machine.** Not `checkpoints/`, not `gs://speechllm-data/checkpoints/`
(only the flat-line runs are there), and neither the Instruct backbone nor
`data/pruned_tokenizer_chat/` is here — that run's checkpoints stayed on the pod
and only `results/instruct-chat/` was fetched.
`configs/instruct-chat-3stage_resume.yaml` still points at
`checkpoints/instruct-chat-3stage/step0014040.pt` and
`/workspace/.llama/checkpoints/Llama3.1-8B-Instruct/`, so if that network volume
is alive the weights are there. The local RTX 3060 (12 GB) cannot hold an 8B
model regardless, so this step is a pod job in any case.

The full decode path IS verified end to end against the OOD shards with a stub
model (`run_wer.py --config configs/stub.yaml --eval-tar …`), through the
per-utterance JSONL and into `analyze_slices.py` / `count_degeneracies.py`. What
is missing is weights, not code.

### Running it — `tools/run_ood_eval.sh`

The whole protocol for one checkpoint is one command. It preflights the resolved
config, smoke-decodes one batch per set, runs the three `run_wer.py` invocations,
then `ood_report.py`, `analyze_slices.py` and `count_degeneracies.py`:

```bash
tools/run_ood_eval.sh --checkpoint checkpoints/staged-full-stack/step0018720.pt
tools/run_ood_eval.sh --config configs/instruct-chat-3stage_resume.yaml \
    --checkpoint checkpoints/instruct-chat-3stage/step0018720.pt --tag chat
```

`--dry-run`, `--smoke-only`, `--skip-smoke`, `--max-batches N`, `--tag`,
`--shards-dir`, `--device`, `--force`. Outputs land in `results/ood-<tag>/` and
`out/ood-<tag>-*`, so the base and chat lines never collide.

Three groupings in that script are deliberate, not incidental:
- **TED-LIUM is decoded `--formats unformatted`.** Its `formatted.txt` is a
  placeholder copy; scoring it would report a number for a reference form that
  does not exist.
- **One `run_wer.py` invocation per corpus** (TED's two caps share one), so each
  carries a correct `--dataset` tag and a late failure does not lose the earlier
  summary CSVs. The cost is two extra model loads.
- **The preflight is not a formality.** A config pointed at the wrong vocabulary
  or the wrong `input_convention` produces a plausible bad WER and no error —
  `PrunedTokenizer.encode` silently drops ids it does not have. The two checks
  that match on path names (`*_chat`, `*Instruct*`) are heuristics; `--force`
  downgrades them to warnings.

**Vocabulary coverage transfers between the two lines.** Reachability is
membership of the ORIGINAL ids in `vocab_map.json`, and both builds prune from
the same `labels.jsonl`; the chat build only forces in four scaffold specials
that never appear in reference text. The 40,034 → 40,039 renumbering matters for
embedding rows, not for what is sayable, so `out/ood-vocab-coverage.json` is
valid for both. Re-run `check_vocab_coverage.py` only if the two vocabularies
were ever built from different label files.

---

## Completed Task — Gated Parallel Audio Adapters

Bottleneck adapters INSIDE Llama layers 0..n_layers-2, active only at audio
token positions, leaving every pretrained parameter untouched:

```
x_hat = audio_mask * tanh(gate) * audio_adapter(adapter_norm(x))   # NEW, reads BLOCK INPUT
x = x + self_attn(input_layernorm(x), cos, sin)
x = x + mlp(post_attention_layernorm(x))
x = x + x_hat                                                      # parallel to the WHOLE layer
```

- `audio_adapter`: two variants selected by `cfg.model.audio_adapter_type`
  (`build_audio_adapter` dispatches; width `r` from `cfg.model.audio_adapter_r`,
  default 0 = disabled, experiments use 128):
  - `"mlp"` (default, C1/C2) — `AudioLayerAdapter`: Linear(d_model→r) → GELU →
    Linear(r→d_model), all bias-free. No-op via **zero-init `up_proj`** (the
    writer); `down_proj` keeps default init.
  - `"swiglu"` (C3-C6) — `AudioSwiGLUAdapter`: SwiGLU bottleneck named like
    `SwiGLUMLP` (`gate_proj`/`up_proj` are d_model→r, `down_proj` is the r→d_model
    writer). Writer gets the GPT-2 scaled init (std 0.02/√L) so the branch writes
    at transformer-layer scale; no-op instead comes from **zero-init `gate_proj`**
    (silu(0)=0), which still trains at step 0 because silu′(0)=0.5. ~3·d_model·r
    params/layer vs the MLP's 2·d_model·r (+50% at equal r; r≈88 is budget-matched).
  Each owns its own RMSNorm (`norm`) — pretrained norms untouched.
- `cfg.model.audio_adapter_zero_writer_layers: N` (swiglu only, default 0) flips
  the first N adapter layers to the LoRA-style scheme instead: **zero `down_proj`**,
  both input projections at default init. The live tensor at step 0 is then
  `down_proj` rather than `gate_proj`. Split is by layer index (`layer_idx < N`);
  `build_audio_adapter(config, layer_idx)` owns the choice.
- Exactly ONE tensor per adapter may be zero-initialised, under either scheme.
  Zeroing a second one makes every gradient in the branch identically zero and
  the adapter never trains — asserted by the llama.py self-tests.
- The branch is parallel to the entire transformer layer (it reads the block
  input, not the post-attention stream) — the deliberate design, replacing the
  earlier MLP-parallel placement.
- **No scalar gate.** The per-layer `tanh(gate)` multiplier was removed from both
  variants: the input projection (`gate_proj` / `down_proj`) already scales the
  branch per-channel and input-dependently, which subsumes one scalar per layer.
- `audio_mask` computed once in Llama.forward and threaded through blocks AND
  gradient checkpointing. Under the flat convention it is derived from
  `audio_lengths` (audio is the per-sample prefix `[0, L)`); Decision 011 added
  an explicit `audio_mask=` argument that takes precedence, because the chat
  scaffold puts audio at an offset this module cannot infer. Both None →
  adapters inactive (stub/self-test paths unchanged).
- New pseudo-module `"audio_adapters"` in stages (name-substring selection
  within llama); its param group gets **weight_decay=0.0** (decay would shrink
  the freshly-zeroed projections and pull the adapter norms away from 1).
- Checkpoint back-compat: loading pre-change llama state_dicts must tolerate
  missing `audio_adapter` keys (warn + keep fresh init); any other missing key
  stays a hard error. Sidecar extended to carry audio-adapter tensors.
- Thread `audio_lengths` through EVERY llama call site: train step, eval/diag
  pass, greedy_generate, run_wer.
- W&B per-depth diagnostic: `Llama.audio_adapter_scales()` →
  `adapter_scale/layer_XX_{writer,input}`, the RMS of each adapter's writer and
  gate projections (replaces the removed `gates/layer_XX` trace). Whichever
  tensor a layer zero-initialised starts at 0, so its rise is the
  this-depth-is-engaging signal.
Experiment series (identifiers per ARCHITECTURE_DECISIONS.md 017; full results
and findings in EXPERIMENT_LOG.md):

| ID | bridge | adapter net | init | trainable | config |
|---|---|---|---|---|---|
| C1 | MLP | MLP r=128, ∥ MLP sub-block | zero writer | adapters | (superseded) |
| C2 | MLP | MLP r=128, ∥ whole layer | zero writer | adapters + bridge | `configs/audio_layer_adapters.yaml` |
| C3 | MLP | SwiGLU r=88 | zero `gate_proj`, GPT-2 writer | adapters | `configs/audio_layer_adapters_swiglu.yaml` |
| C4 | MLP | SwiGLU r=88 | layers 0-14 zero writer, 15-30 as C3 | adapters + bridge | `configs/audio_layer_adapters_swiglu_v4.yaml` |
| C5 | MLP | SwiGLU r=88 | as C3 | adapters + bridge | `configs/audio_layer_adapters_swiglu.yaml` |
| C6 | SwiGLU | SwiGLU r=88, no scalar gate | as C4 | adapters + bridge | `configs/audio_layer_adapters_swiglu_v5.yaml` |

Config filenames still carry a retired `v1-v5` numbering; the IDs above are
authoritative. C3 and C5 share a config and differ only in `trainable`.

C1-C5 warm-start from the bridge-only stage-1 checkpoint (step 4680, run B2).
**C6 cannot**: its bridge is a different module and that checkpoint holds
MLP-bridge tensors, so `init_from` is null and C6 trains its bridge from
scratch. Its early loss curve is not comparable to C2-C5.

Checkpoint compatibility: mlp↔swiglu variants are incompatible by construction
(different key names/shapes) and fail loudly, which is intended. C3/C5 and C4 share
a state-dict layout, so an `init_from` between them loads silently and overwrites
the split init — don't. Checkpoints written BEFORE the scalar gate was removed
(the in-flight C2/C3/C4 runs) carry `audio_adapter.gate` keys and will NOT load
into current code; converting one means folding tanh(gate)≈0.762 into the writer
and dropping the key.

---

## Project Knowledge Base

Documents in the planning-chat knowledge base, referenced by name:

| Document | Holds |
|---|---|
| `CLAUDE.md` (this file) | Operative spec — what the system does now |
| `claude_project_documents/ARCHITECTURE_DECISIONS.md` | Narrative decision log — why, and what was ruled out |
| `claude_project_documents/EXPERIMENT_LOG.md` | Every run, its question, its result, its confounds |
| `claude_project_documents/DATA_PROVENANCE.md` | Corpus lineage, relabelling, drops, vocabulary invariants |
| `claude_project_documents/FORMATTING_SPEC.md` | The formatted-label contract (spec wins over derived code) |
| `claude_project_documents/TRAINING_DIAGNOSTICS_REFERENCE.md` | Training-time signals: what each means, how to read it |
| `claude_project_documents/DEGENERACY_TAXONOMY.md` | Generated-output failure modes and their interventions |
| `claude_project_documents/ENGINEERING_REFERENCE.md` | Optimizer internals, VRAM budget, attention |
| `claude_project_documents/INFRASTRUCTURE_REFERENCE.md` | RunPod/GCP setup, commands, gotchas |
| `claude_project_documents/TRAINING_COST_ESTIMATES.md` | Hardware pricing and run cost accounting |
| `claude_project_documents/LITERATURE_SURVEY.md` | Prior art, comparison class, eval methodology |

If this file and one of the above disagree about current behaviour, this file
wins; if they disagree about rationale, `ARCHITECTURE_DECISIONS.md` wins.