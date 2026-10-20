# speech-llm — Claude Code Project Brief

The persistent brief for Claude Code. Re-read it at the start of every session.
It describes what the code does **now**: architecture, sequence layout,
invariants, configs, stages, data pipeline, eval tools and conventions. Do not
make architectural decisions that contradict it without explicit instruction
from the project lead.

If this file and the code disagree, the code is ground truth — flag the drift,
don't silently follow the stale text.

Pre-cleanup state (flat convention, MLP bridge/adapters, B/C/D-series configs,
full experiment narrative): tag **`phase1-final`**, plus commit `b93c661`.

---

## Project overview

An end-to-end instructable SpeechLLM trained on LibriSpeech 960h. Raw audio in,
transcript out, conditioned on a natural-language instruction (unformatted vs
formatted output). Training is multi-stage, declared as a YAML `stages:` list.

Priorities, in order: (1) educational clarity — every component readable and
self-contained; (2) cost efficiency (RunPod/GCP); (3) open-source
reproducibility; (4) WER quality.

The project lead is the sole developer. Design happens in the planning chat;
implementation is delegated here via explicit prompts. Flag untested paths and
prefer root-cause fixes over patches.

---

## Architecture (run E1, the only supported structure)

```
Raw audio (16 kHz, ≤ 30 s) → log-mel (80, T_mel)
        ↓
[ Whisper small encoder ]  88M   d_model=768, 12 layers, Pre-LN
        ↓  (B, T_mel//2, 768)    conv stem stride 2
[ Temporal mean pool ×4 ]  —     (B, ceil(T_enc/4), 768)   80 ms / token
        ↓
[ SwiGLU bridge ]          ~14M  down_proj(silu(gate_proj(x)) ⊙ up_proj(x)), 768→2048→4096
                                 + learned audio_bos / audio_eos marker vectors
        ↓  spliced into the chat sequence (below)
[ Llama 3.1 8B Instruct ]  8B    32 layers, d_model 4096, GQA 32/8, SwiGLU, RoPE θ=500k
  + SwiGLU audio adapters        layers 0..30, parallel to each whole layer, audio positions only
        ↓
[ LM head (tied) ]               loss on transcript tokens + trailing <|eot_id|>
```

All model code is from scratch (`model/`); no `transformers` in model code.
Weights: Whisper from the OpenAI checkpoint (`load_openai_weights`,
shape-asserted); Llama from HF safetensors or Meta `consolidated.*.pth`
(`load_meta_weights`), embedding rows taken through `vocab_map.json`.

### Sequence layout (`model/sequence.py`)

```
[bos][start_header]"user"[end_header]"\n\n"
    AUDIO_BOS [audio embeddings] AUDIO_EOS "\n" [instruction]
[eot]
[start_header]"assistant"[end_header]"\n\n"
    [transcript]
[eot]                                          ← EOS target (a label)
```

- `ChatTemplate.from_tokenizer(tokenizer)` precomputes the scaffold id segments
  once per run (`seg_pre_audio`, `seg_pre_instruction`, `seg_pre_transcript`,
  `eot_token_id`). It is the single source of truth for assembly and is
  id-space agnostic, so `tools/build_vocab.py` builds the same scaffold in
  original id space to verify it. Scaffold texts live in
  `model.sequence.CHAT_SCAFFOLD_TEXTS`.
- `prepare_input(...)` → `(inputs, labels, audio_mask)`. Labels are -100
  everywhere except transcript tokens and the **trailing** eot; the user-turn
  eot is input-only. Per-sample padding is stripped before concatenation; the
  batch is then right-padded. No pad token enters the loss.
- Audio comes first in the user turn, so audio content always starts at the
  constant `ChatTemplate.audio_offset = len(seg_pre_audio) + 1`. `audio_mask`
  (B, S, 1) covers audio **content** only (not the markers, not scaffold) and
  is what the in-layer adapters gate on. Llama never infers it.
- `Llama.forward(inputs_embeds, labels=None, audio_mask=None)` always takes
  embeddings. `audio_mask=None` (or all zeros) leaves the adapters inactive.
- Generation: `EvalPrefixBatch` builds the prefix up to the assistant header,
  right-pads, and inserts each generated token at `gen_pos[i]` so causal
  attention never sees padding inside real history; it grows the audio mask
  with zeros. `utils/generate.greedy_generate_from_prefix` is THE decoding loop;
  `greedy_generate` (single-turn) and `tools/run_longform.py` (multi-turn,
  `ConversationPrefix` / `ConversationPrefixBatch`, eval only) both call it.

### Instructions

`data.INSTRUCTION_VARIANTS` holds the two trained prompts:

```
"Transcribe the audio exactly as spoken, in lowercase with no punctuation."           → unformatted.txt
"Transcribe the audio as written text, with capitalization, punctuation, and numbers as digits." → formatted.txt
```

`instruction_mode` (`unformatted` | `formatted` | `both`) is set at run level
and overridable per stage; `both` picks one variant uniformly per sample.
`tools/build_vocab.py` keeps its own copy of the strings (data.py's self-test
checks they match). This mechanism is due to be replaced in a later phase.

---

## Invariants (break one and training/eval silently degrades)

- **Bridge output projection is NEAR zero, never exactly zero** (std 0.02/√6,
  zero bias). Bridge output is the entire content of the audio positions and
  Llama is bias-free, so an exactly-zero write stays zero through all layers;
  each RMSNorm then adds a 1/√eps ≈ 316× backward gain, gradients hit ~1e7,
  fp16 overflows and GradScaler skips every step. Asserted in adapter.py.
- **In-layer adapters zero exactly one tensor: `gate_proj`.** silu(0)=0 makes
  the branch an exact no-op; silu′(0)=0.5 keeps gate_proj training at step 0.
  `down_proj` has GPT-2 scaled init (0.02/√L). Zeroing a second projection kills
  every gradient in the branch. Each adapter owns its own RMSNorm; no scalar
  gate. The audio_adapters param group has weight_decay 0.
- **Markers** `audio_bos`/`audio_eos` are `nn.Parameter(d_model)` on the bridge,
  normal(0, 0.02) init. They are not vocabulary tokens (no logit row, never
  generated) and ride in the `adapter` param group and state_dict.
- **`<|eot_id|>` is the only terminator.** Its embedding row must come from the
  Instruct checkpoint through vocab_map; `build.py` prints each chat-special
  row's provenance and refuses a vocabulary without the four specials.
- **Vocabulary is built from the same label text the shards hold**
  (`build_vocab.py --labels_file data/labels.jsonl`). `PrunedTokenizer.encode`
  silently DROPS unmapped ids, so a mismatched vocabulary — or an unreachable
  word in a new instruction — deletes tokens instead of failing.
- **Rebuilding the vocabulary reassigns every id.** That is safe only while no
  checkpoint stores a trained `llama` (checkpoints save llama only when it is in
  `modules_dirty`). It also invalidates `baselines.json` (uniform baseline is
  log V) and `metrics._VOCAB_SIZE` (self-test default only).
- **`--llama_dir` must not be a previous pruned_tokenizer dir**; build_vocab
  refuses to copy a stale `vocab_map.json`/`pruned_config.json`.
- **bf16 is ruled out for training**: fp16 autocast + GradScaler, fp32 master
  weights. Without scaling, adapter gradients underflow through frozen Llama.
- **audio_lengths** = `(T_mel // 2 + 3) // 4` (encoder stride-2 floor, pool-4
  ceil). Used for assembly, masks and throughput (1 token = 80 ms).
- **Batch composition perturbs embeddings**: the Whisper encoder attends across
  mel padding, so comparisons between prompts/modes must decode in identical
  batches (run_probes, run_longform do this deliberately).

### Vocabulary (`data/pruned_tokenizer_chat/`)

40,039 tokens; the four Llama 3.1 chat specials are forced in by canonical id:
`<|begin_of_text|>` 128000 → 40,035, `<|start_header_id|>` 128006 → 40,036,
`<|end_header_id|>` 128007 → 40,037, `<|eot_id|>` 128009 → 40,038.
`pruned_config.json` carries those four ids plus `vocab_size`; `sep_token_id`
is written as an alias of eot. `data.load_pruned_config()` is the only reader.
By id, not by name: a Meta-format directory loads through transformers'
TikToken fallback, which exposes no specials. `data/pruned_tokenizer/` is the
legacy 40,034-token vocabulary (no specials); encode-only tools can still read
it, nothing can train or decode with it.

---

## Configs (`utils/config.py`)

Load order: `configs/base.yaml` → run YAML (deep-merged; lists replaced) → thin
CLI overrides (`--config`, `--resume`, `--wandb/--no-wandb`, `--run-name`,
`--checkpoint-dir`). Result: frozen dataclasses, validated at load.

**Every YAML key is whitelisted.** An unknown key is an error; a key removed in
the cleanup (`input_convention`, `bridge_type`, `audio_adapter_type`,
`audio_adapter_zero_writer_layers`, `adapter_pca_init`, `paired_prompts`) fails
with "this setting was removed". Unknown CLI flags also fail (there is no
`--stub` or `--max-steps`: stub mode and step caps live in YAML).

```yaml
seed: 42                    # also keys the shard shuffle; CHANGE on resume for fresh data order
data:
  shards_file | shards      # shard list file OR brace/glob pattern (exactly one)
  tokenizer: data/pruned_tokenizer_chat/
  diag_shard:               # training-time diagnostic shard
  eval: {dev_clean, dev_other, test_clean, test_other}
  num_workers: 4
model:
  stub: false               # true → tiny random Llama (stub_dims)
  stub_dims: {n_layers: 6, d_model: 512, n_heads: 8, n_kv_heads: 2, intermediate_size: 1024}
  whisper_ckpt: weights/whisper_small.pt
  llama_ckpt: weights/Llama3.1-8B-Instruct/
  gradient_checkpointing: true          # Llama layers (ignored in stub mode)
  encoder_gradient_checkpointing: true  # Whisper blocks; self-gating while frozen
  init_from:                # weights-only warm start (handoff ckpt / sidecar)
  audio_adapter_r: 0        # SwiGLU in-layer adapter width; 0 = none (E1 uses 128)
optim: {betas, weight_decay, grad_clip: {enabled, max_norm}}
logging: {wandb, project, run_name}
checkpoint: {dir, save_every}
metrics: {train_periods, eval_every, eval_batches, eval_batch_size, compute_only_on_emit, wer: {...}}
run: {max_steps, instruction_mode}     # max_steps is a global cap across stages
stage_defaults: {...}                  # merged under every stage
stages:
  - name: ...
    trainable: [adapter]    # subset of {encoder, adapter, llama, audio_adapters}
    lrs: {adapter: 1.0e-4}  # keys must equal trainable exactly; replaced, not merged
    schedule: {warmup_steps: 0}
    batch_size, accum_steps
    instruction_mode:       # optional per-stage override
    optimizer_init: fresh   # "inherit" raises NotImplementedError
    exit: {strategy, threshold, min_steps}
      # first_token_below | eval_loss_below | max_steps | epochs | never
```

`adapter` = the bridge (incl. markers); `audio_adapters` = the name-selected
in-layer adapter subset of llama's parameters.

| config | what |
|---|---|
| `base.yaml` | every default |
| `instruct-chat-3stage.yaml` | run E1: bridge_only → audio_adapters_only → full_stack [encoder, adapter, audio_adapters]; 3 × 4680 + 1000 steps; stage-3 batch 60 × accum 2. Step 15040 is the line's endpoint |
| `example.yaml` | illustrative two-stage config; read by the config/stages self-tests |
| `stub_chat.yaml` | 20-step, two-stage stub; the model `tools/equivalence_check.py` captures |
| `stub_chat_3stage.yaml` | the E1 curriculum in miniature (18 steps), incl. full_stack with the encoder |
| `stub_longform.yaml` / `stub_probes.yaml` | stub models for run_longform.py / run_probes.py |

---

## Training (`training.py`, `stages.py`)

- `for stage_idx in range(start_stage, len(stages))`: each `Stage.setup` sets
  `requires_grad` in place, builds a fresh AdamW8bit (bitsandbytes) with named
  param groups in canonical order (encoder, adapter, llama, audio_adapters) and
  its own warmup scheduler. `Stage.make_loader(epoch, stage_idx)` shuffles
  shards with `random.Random(seed + stage_idx*10000 + epoch)`; loaders are
  finite (one pass per epoch). `Stage.should_advance` evaluates the exit
  strategy; `epochs` is checked at the epoch boundary.
- Checkpoints carry a `kind`:
  - `periodic` — full state (optimizer, scaler, scheduler, run counters); the
    only kind `--resume` accepts. Resume derives the start stage from the
    checkpoint's `stage_index`; there is no batch fast-forward.
  - `handoff` — weights-only, written at every stage advance (plus an
    `-adapter.pt` sidecar); consumed by `model.init_from`.
  Checkpoints are deltas over pretrained: encoder/llama are written only once
  they are in `modules_dirty`; bridge and audio adapters are always written.
  Bridge state_dicts without the markers load with a warning; any other
  missing/unexpected key is a hard error.
- Memory: 8-bit AdamW is mandatory for full-model single-GPU training; gradient
  checkpointing (`use_reentrant=False`) on Llama layers and Whisper blocks;
  attention via `F.scaled_dot_product_attention`. bnb allocates optimizer state
  lazily over the first steps — warm up before measuring memory
  (`tools/find_max_batch.py` does). Keep `accum_steps=1` unless the effective
  batch genuinely must exceed VRAM.
- Diagnostics (`metrics.py`): `MetricCollector` routes observations by family
  (logits / grads / loss) and `render()` prints. Signals: first-token vs rest
  loss, per-module grad norms, first-token top-5, terminator fraction (W&B key
  still `collapse/*_sep_fraction`), normalised entropy (log V from
  pruned_config), mean max logit, per-layer `adapter_scale/layer_XX_{writer,input}`.
  Eval first-token loss below the unigram baseline = "model uses audio".
  Baselines come from `tools/compute_baselines.py` → `baselines.json`.
- The diag shard is consumed by a PERSISTENT iterator across eval passes; do not
  rebuild it per eval.

---

## Data pipeline

```
1. tools/download_data.py      LibriSpeech splits (MD5-verified)
2. tools/label_formatted.py    LLM labelling run (Batches API): prepare/submit/poll/
                               validate/retry/finalize → data/labels.jsonl
3. tools/build_vocab.py        --labels_file data/labels.jsonl --llama_dir <Instruct>
                               --output_dir data/pruned_tokenizer_chat/
4. tools/preprocess.py         FLAC → mel → WebDataset shards (fast with --labels_file)
5. tools/create_subset.py / make_eval_subset.py   shard lists / eval subsets
```

Labels come first because the vocabulary is derived from them. Shards hold
`{key}.mel.npy` (float16, natural length), `{key}.unformatted.txt`,
`{key}.formatted.txt`; sized by audio duration (~30 min). Collation pads mel to
`ceil(max/8)·8`. **Relabelling** does not rerun this: `tools/relabel_shards.py`
swaps the two label members in existing shards and drops `validation: failed`
records (preprocess drops them too); rebuild the vocabulary afterwards.
The formatted-label contract is `claude_project_documents/FORMATTING_SPEC.md`.

---

## Eval tools

All decoding goes through `greedy_generate` / `greedy_generate_from_prefix`, so
every number below comes from the same code path. WER is jiwer on raw text — no
normalisation — so it measures instruction adherence too.

**WER sweep** — `tools/run_wer.py --config C --checkpoints 'glob' --output X.csv`
evaluates checkpoints on `build_sorted_eval_dataloader` (tarfile, finite, sorted
by audio length, references ≤41 tokens, stable order → reproducible batches).
Writes the summary CSV plus a FULL per-(step, split) transcription JSONL
(utterance key, per-sample WER). `--splits` restricts; `--eval-tar NAME=PATH
--dataset TAG` evaluates arbitrary shards. Rejects `-adapter.pt` sidecars.
Downstream, all stdlib-only:
- `plot_wer.py` — WER-vs-step curve with stage spans.
- `count_degeneracies.py` — empty / truncation / hallucination / loop /
  style-flip. It OWNS the degeneracy rules; `analyze_slices.py` imports them.
- `analyze_slices.py` — WER, degeneracy and adherence by mode, reference length,
  numeral presence, rare words (`--train-vocab` from `build_vocab_freq.py`),
  `--group-by` any JSONL field; plus an example gallery.
- `compare_hyps.py A.jsonl B.jsonl` — per-utterance A/B; the regression gate for
  a refactor (greedy decode is deterministic).

**Long-form multi-turn** (eval only, zero-shot) — `tools/run_longform.py`
groups consecutive utterances (`speaker-chapter-utterance` order, no gaps) into
conversations of `--turns` user turns and decodes three context modes in the
same batches: `self` (own previous transcripts), `oracle` (gold) and `single`
(each utterance alone — the matched control, decoded here, not lifted from a
sweep). The instruction repeats in every user turn; turn 1 is asserted
byte-identical to the single-turn prefix. Point `--shard` at FULL split shards
(subsets have almost no consecutive runs). `--max-runs-per-group` widens
speaker coverage. JSONL adds `conv_id`, `turn_index`, `context_mode`.

**Instruction-generalisation probes** (eval only, zero-shot):
`probe_spec.py` (every prompt, once) → `check_vocab_feasibility.py` (THE GATE:
can the pruned vocabulary express each probe's output AND its prompt; run it
first and after any prompt edit) → `run_probes.py` (refuses ungated probes;
`--force` stamps rows `gate_forced`; P4 includes a text-only control with no
audio block) → `score_probes.py` (Following Rate + pass rate per probe, never
aggregated; refuses to report a gap against a degenerate control). P1
(uppercase) is dropped by the gate: upper-case text is unreachable in this
vocabulary.

**Out-of-distribution** — TED-LIUM 3 test, Common Voice 17 en test, Earnings-22
subset, built by `prepare_ood_*.py` to the shared layout in `ood_shard.py` (adds
a `{key}.flac` member). `check_vocab_coverage.py` measures reference
reachability before decoding; `run_whisper_control.py` decodes the same audio
with openai/whisper-small; `ood_report.py` joins ours + control + coverage
(verbatim pairing normalises BOTH sides with FORMATTING_SPEC §6; digit-free
re-score; TED-LIUM is verbatim-only). `tools/run_ood_eval.sh --checkpoint X`
runs the whole protocol for one checkpoint (preflight → smoke → decode → report
→ slices); the preflight refuses a vocabulary without the chat specials, and a
backbone path that does not look like Instruct unless `--force`.

**VRAM** — `tools/find_max_batch.py --config C --stage N` (or `--eval`)
binary-searches the largest batch for the exact stack and stage the config
builds.

**Equivalence** — `tools/equivalence_check.py` captures one CPU/fp32 forward
pass + short greedy decode of the `stub_chat` model under `out/equivalence/`;
`--verify` reloads with `strict=True` and requires logits/loss within 1e-5 and
identical ids. Run it before and after any refactor that should not change the
model.

---

## Stub workflow (local RTX 3060)

```bash
python tools/make_stub_shard.py          # once per checkout → data_stub/ (gitignored)
python training.py --config configs/stub_chat.yaml
python training.py --config configs/stub_chat_3stage.yaml
python tools/run_longform.py --config configs/stub_longform.yaml \
    --shard stub=data_stub/stub-diag-000000.tar --turns 3 --max-convs 2 \
    --print-layout 1 --out-dir out/longform-stub
python tools/check_vocab_feasibility.py --tokenizer data/pruned_tokenizer_chat/ \
    --shard data_stub/stub-diag-000000.tar \
    --out-json out/probe-feasibility.json --out-md out/probe-feasibility.md
python tools/run_probes.py --config configs/stub_probes.yaml \
    --shard data_stub/stub-diag-000000.tar --feasibility out/probe-feasibility.json \
    --probes P3 P4 --n 5 --print-prompts 1 --decode-anyway --out-dir out/probes-stub
python tools/equivalence_check.py --verify
```

Stub shards are random mel + real labels from `tests/fixtures/stub_labels.jsonl`
(8 runs of 8 consecutive utterances). Hypotheses are garbage by construction;
the plumbing is under test.

Self-tests live in `if __name__ == "__main__"` blocks. Run directly: `model/*.py`,
`stages.py`, `utils/config.py`, `utils/checkpoint.py`, `metrics.py`,
`tools/ood_shard.py`. With `--self-test`: `data.py` and most tools (`git grep -l
-- --self-test`). All are CPU-runnable with mocks. Extend them when touching a
file.

---

## Environment

- Local: WSL2 Ubuntu 22.04, Python 3.11, venv `source ~/.venvs/aidev/bin/activate`,
  RTX 3060 12 GB — stub runs and self-tests only (cannot hold the 8B model).
  `data/` and `checkpoints/` are symlinks onto a separate data disk. Host RAM is
  ~7 GB: stream shards, don't read a large one whole.
- Experiments: RunPod A100 80GB, network volume at /workspace; data, weights and
  checkpoints sync with `gs://speechllm-data/`. Local-only artefacts
  (`results/`, `out/`, `runs/`, `checkpoints_aux/`) are backed up under
  `gs://speechllm-data/artifacts/`.
- Tracking: W&B; run configs use project `speechllm-auto-stage`, base
  `speech-llm-dev`.

---

## Repo structure

```
speech-llm/
├── CLAUDE.md
├── training.py            # multi-stage training loop (entry point)
├── build.py               # build_models(): construct + init encoder / bridge / llama
├── data.py                # loaders, PrunedTokenizer, load_pruned_config, INSTRUCTION_VARIANTS
├── stages.py              # Stage / StageContext: trainability, optimizer, loader, exit
├── metrics.py             # MetricCollector + render()
├── model/
│   ├── whisper_encoder.py # Whisper small encoder (+ log_mel_spectrogram)
│   ├── adapter.py         # AudioSwiGLUBridge (pool + SwiGLU + audio markers)
│   ├── llama.py           # Llama 3.1 8B + AudioSwiGLUAdapter + weight loading
│   └── sequence.py        # ChatTemplate, prepare_input, EvalPrefixBatch,
│                          #   ConversationPrefix[Batch] (multi-turn, eval only)
├── utils/
│   ├── config.py          # frozen-dataclass config + key whitelist
│   ├── checkpoint.py      # save/load, kind tagging, delta saving, ResumeState
│   ├── evaluate.py        # evaluate_all_splits, compute_wer
│   ├── generate.py        # greedy_generate, greedy_generate_from_prefix
│   └── optim.py           # build_adamw8bit, make_warmup_scheduler
├── tools/
│   ├── download_data.py, download_weights.py
│   ├── label_formatted.py, relabel_shards.py, build_vocab.py, build_vocab_freq.py
│   ├── preprocess.py, create_subset.py, make_eval_subset.py, make_stub_shard.py
│   ├── compute_baselines.py, find_max_batch.py, equivalence_check.py
│   ├── run_wer.py, plot_wer.py, count_degeneracies.py, analyze_slices.py, compare_hyps.py
│   ├── run_longform.py
│   ├── probe_spec.py, check_vocab_feasibility.py, run_probes.py, score_probes.py
│   └── ood_shard.py, prepare_ood_{tedlium,commonvoice,earnings22}.py,
│       check_vocab_coverage.py, run_whisper_control.py, ood_report.py, run_ood_eval.sh
├── configs/               # base.yaml, instruct-chat-3stage.yaml, example.yaml, stub_*.yaml
├── tests/fixtures/        # stub_labels.jsonl
├── docs/                  # narrative write-ups (Parts 0-3)
├── README.md, Dockerfile, requirements.txt, baselines.json
├── data/                  # pruned_tokenizer_chat/, shards, eval shards, ood_shards/ (gitignored)
└── weights/               # whisper_small.pt, Llama3.1-8B-Instruct/ (gitignored)
```

---

## Conventions

- Python 3.11, type hints and docstrings everywhere; comments explain WHY.
- No HuggingFace `transformers` in model code — only in data.py (tokenizer) and
  tools/.
- Fail loudly on shape mismatches; assert before every `copy_`.
- No hardcoded paths or secrets; pathlib + CLI flags + env vars.
- Tools are independently readable; duplication across tools is acceptable.
- Container-first: identical behaviour in the venv and on RunPod/GCP.
- Readable over clever.

---

## Knowledge base

In `claude_project_documents/` (planning-chat knowledge base; untracked here,
backed up to `gs://speechllm-data/artifacts/claude_project_documents/`):
ARCHITECTURE_DECISIONS.md (why, and what was ruled out), EXPERIMENT_LOG.md
(every run, result and finding, incl. Series E), FORMATTING_SPEC.md (label
contract; spec wins over derived code), TRAINING_DIAGNOSTICS_REFERENCE.md,
INFRASTRUCTURE_REFERENCE.md, TRAINING_COST_ESTIMATES.md, LITERATURE_SURVEY.md,
OPTIMIZER_REFERENCE.md, VRAM_BUDGET_REFERENCE.md, FLASH_ATTENTION_REFERENCE.md.
The planning chat also holds DEGENERACY_TAXONOMY.md and DATA_PROVENANCE.md. If this file and one of them disagree about
current behaviour, this file wins; about rationale, ARCHITECTURE_DECISIONS.md
wins.
