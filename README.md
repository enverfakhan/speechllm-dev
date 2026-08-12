# speech-llm

**An instructible SpeechLLM, built from scratch: Whisper encoder → MLP adapter → Llama 3.1 8B, trained on LibriSpeech 960h.**

Every model component in this repository — the Whisper encoder, the Llama 3.1 transformer, the adapter, the training loop, the batched greedy decoder — is implemented from scratch in plain PyTorch. No `transformers` model classes, no training framework. The goal is a codebase where every design decision is visible, explained, and small enough to read in an afternoon.

This README is the entry point. It explains why the project exists, what was built, and how the work evolved. The detailed engineering story lives in the numbered documents under [`docs/`](#documentation-map).

---

## Why this project

I wanted hands-on, end-to-end experience training a model that is state-of-the-art *in its approach* within my field (ASR): an **instructible SpeechLLM** — a model whose transcription style follows the instruction it is given. The goals are, in order:

1. **Experience the full arc** — from raw data and pretrained checkpoints, through pipeline construction and debugging, to multi-stage training on rented cloud GPUs.
2. **Keep the implementation as simple as possible** — every component from scratch, readable, and educational. Complexity that doesn't teach something is avoided.
3. **Be cost-conscious throughout** — the entire experimental program runs on a single rented A100 80GB (~$1.56/hr all-in), with data and checkpoints on GCS and a deliberately small network volume. Total spend to date: **~€475**, with a projected **~€700** for the complete project. The engineering choices (8-bit optimizer, gradient checkpointing, mixed precision, pruned vocabulary) exist to make an 8B model trainable on this budget.

### Making "instructible" concrete

**Dataset:** LibriSpeech 960h. It's clean, open, easy to access, large enough to support real experimentation, and small enough that a full multi-epoch training run is actually achievable on a constrained budget.

**Instructions:** Two — minimal, well-defined, and different in output space:

| Instruction | Target output |
|---|---|
| *"Transcribe the audio exactly as spoken, in lowercase with no punctuation."* | lowercase, no punctuation (LibriSpeech's native format) |
| *"Transcribe the audio as written text, with capitalization, punctuation, and numbers as digits."* | punctuation, sentence caps, proper-noun caps, numbers as digits |

LibriSpeech ships unformatted, so the formatted labels had to be created. They are generated with open-source tools (a BERT-based punctuation restorer + spaCy for proper nouns). The labels don't need to be perfect — they need to be **consistent**. The goal isn't a SOTA formatter; it's the experience of training a model to follow instructions on audio.

---

## How this codebase was written

The implementation code in this repository was written by Claude Code, from prompts I wrote. My work is the layer above the code: researching the options, making and testing the design decisions, specifying each implementation task, and reviewing the results. The design didn't arrive fully formed — it started with specific ideas and evolved through experimentation, and the documents in `docs/` trace that evolution and the reasoning behind each decision.

---

## Architecture

```
log-mel spectrogram (80 × T)
        │
        ▼
Whisper small encoder (from scratch, 88M params, pretrained OpenAI weights)
        │  (B, 1500, 768)
        ▼
Adapter: mean-pool ×4 → Linear(768→2048) → GELU → Linear(2048→4096)   [~6M params]
        │  (B, 375, 4096) — one token per 80ms of audio
        ▼
Llama 3.1 8B (from scratch: RMSNorm, RoPE, GQA, SwiGLU; pretrained Meta weights)
        │
        ▼
[audio | SEP | instruction | SEP | transcript | SEP]
                            loss on transcript + final SEP only
```

Key implementation facts:

- **From-scratch model code.** `model/whisper_encoder.py`, `model/llama.py`, `model/adapter.py` have zero dependency on HuggingFace `transformers` or `openai-whisper` for model logic. Pretrained weights are loaded via explicit, shape-asserted key mapping.
- **Pruned vocabulary (~40K of 128K tokens).** The output head only needs LibriSpeech's vocabulary under both normalizations. This shrinks the LM head and embedding matrices and is one of several deliberate cost levers.
- **Sequence assembly with no padding leakage.** Per-sample padding is stripped before concatenation; causal attention never sees a pad token in the history of a real token — during training *or* batched greedy generation.
- **Single-GPU training of an 8B model** via bitsandbytes 8-bit AdamW, gradient checkpointing, and fp16 autocast with GradScaler. Why each of these is necessary is covered in [Part 1](docs/01-building-the-pipeline.md), with measurements.

---

## How the work evolved

The project followed a strict sequencing discipline: **local → single-GPU cloud → (next) multi-GPU.**

1. **Artifacts first.** Download and verify weights, build the WebDataset shards, precompute formatted labels, prune the tokenizer. ([Part 0](docs/00-creating-the-artifacts.md))
2. **Pipeline on a stub.** The full encoder→adapter→LLM interface was developed locally on an RTX 3060 using a tiny randomly-initialised Llama stub — same code paths, same loss, 34M instead of 8B parameters. A memorization experiment (deliberately overfit a tiny subset) validated the pipeline end-to-end before any cloud spend. ([Part 1](docs/01-building-the-pipeline.md))
3. **Instrument, then train.** Before scaling up, I built a diagnostic framework: corpus-derived loss baselines (uniform / unigram / bigram / first-token), first-token loss as the primary "is the audio actually being used?" signal, collapse detection, gradient-norm tracking. WER is deliberately *not* a training-time metric — it's an evaluation artifact computed after the fact. ([Part 2](docs/02-training-signals.md))
4. **Staged training.** A single joint training run doesn't work here (a randomly-initialised adapter between two pretrained networks needs to be brought online first). Training is organised as explicit stages — Stage 1: adapter-only; Stage 2: wider unfreezing — with checkpoint semantics (resume vs. warm-start) designed around that structure and around Spot-instance preemption. ([Parts 2–3](docs/02-training-signals.md))

---

## Documentation map

| Doc | Contents |
|---|---|
| [**Part 0 — Creating the artifacts**](docs/00-creating-the-artifacts.md) | Downloading & verifying weights; LibriSpeech → WebDataset shards; generating formatted labels; pruning the tokenizer; compute setup and the GCP → RunPod pivot |
| [**Part 1 — Building the pipeline**](docs/01-building-the-pipeline.md) | The forward pass end-to-end; sequence assembly & loss masking; 8-bit AdamW; gradient checkpointing; mixed precision & GradScaler; the memorization experiment; finding the max batch size |
| [**Part 2 — Memorization & training signals**](docs/02-training-signals.md) | From memorization to a 6-speaker generalisation testbed; why WER fails as a training-time signal; the generation-free diagnostic suite and its corpus-derived baselines; the local experiments (freezing, learning rates, initialisation, adapter-only vs. full fine-tune) that produced the staged training strategy |
| [**Part 3 — Stage 1 on the full dataset**](docs/03-stage1-training.md) | The batch size sweep and its multi-GPU implications; the hit/miss metric decomposition; Stage-1 experiments across instruction formats with W&B curves; the criterion for advancing to Stage 2 |

---

## Status

- **Stage 1 (adapter-only) training: complete and validated.** A ~6M-parameter adapter alone, with encoder and Llama frozen, produces coherent transcription — the audio bridge works.
- **Stage 2 experimentation: in progress.** Current experiments compare unfreezing strategies and stage-transition timing (see [Part 3](docs/03-stage1-training.md)).
- **Full evaluation pending.** WER for both instruction formats, plus analysis of model behavior across audio length and other variables, will be published here once the finalized training runs complete.

---

## What this project demonstrates

- **From-scratch implementation of modern architectures** — Whisper encoder and Llama 3.1 (RMSNorm, RoPE, GQA, SwiGLU), with pretrained weight loading via explicit key/shape mapping.
- **Memory engineering for large models on constrained hardware** — measured VRAM budgets, 8-bit optimizer states, gradient checkpointing, and diagnosed mixed-precision failure modes (bf16 vs fp16, GradScaler underflow).
- **Training diagnostics** — designing signals and information-theoretic baselines that indicate whether the model is learning from audio well before WER can.
- **Experiment design and infrastructure pragmatism** — staged training, resumable checkpointing built for Spot preemption, and cost-tracked runs on rented cloud GPUs.
- **AI-assisted development** — implementation delegated to Claude Code from written specifications, with design, testing, and review done by me.

---

## Repository structure

```
speech-llm/
├── model/                  # from-scratch model code
│   ├── whisper_encoder.py  #   Whisper small encoder (conv stem, pre-LN blocks)
│   ├── llama.py            #   Llama 3.1 8B (RMSNorm, RoPE, GQA, SwiGLU)
│   ├── adapter.py          #   mean-pool + 2-layer MLP bridge
│   └── sequence.py         #   sequence assembly, loss masking, batched generation
├── utils/                  # checkpoint, config, evaluate, generate, optim
├── tools/                  # artifact creation & analysis
│   ├── download_weights.py, download_data.py, precompute_labels.py,
│   ├── build_vocab.py, preprocess.py, create_subset.py, make_dev_dataset.py,
│   └── compute_baselines.py, find_max_batch.py, run_wer.py, …
├── configs/                # YAML configs — sparse overrides on base.yaml
├── docs/                   # Parts 0–3 (this README's companion documents)
├── build.py                # shared model construction (training + eval see identical init)
├── data.py                 # WebDataset pipeline, tokenizer, dataloaders
├── stages.py               # multi-stage training definitions
├── metrics.py              # training-time diagnostic signals
├── training.py             # training entry point
├── smoke_test.py           # end-to-end pipeline verification
└── Dockerfile
```

## Quick start

See [Part 0](docs/00-creating-the-artifacts.md) for the full artifact-creation walkthrough. The short version:

```bash
python tools/download_weights.py --output_dir weights/            # Whisper (verified) + Llama instructions
python tools/download_data.py                                     # LibriSpeech 960h
python tools/precompute_labels.py --librispeech_dir data/librispeech/LibriSpeech/ --output data/labels.jsonl
python tools/build_vocab.py --librispeech_dir data/librispeech/LibriSpeech/ --llama_dir weights/llama3.1-8b/ --output_dir data/pruned_tokenizer/
python tools/preprocess.py --input_dir … --output_dir data/shards/ --labels_file data/labels.jsonl
python training.py --config configs/base.yaml
```