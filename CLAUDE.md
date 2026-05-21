# speech-llm — Claude Code Project Brief

This document is the persistent brief for Claude Code. Re-read it at the start of
every session. It contains the full architecture, all confirmed decisions, repo
structure, component specs, and the active task. Do not make architectural decisions
that contradict anything in this file without explicit instruction from the project lead.

---

## Project Overview

An end-to-end instructable SpeechLLM trained on LibriSpeech 960h on GCP.
The model takes raw audio as input and produces a transcript, conditioned on a
natural-language instruction prompt. Training is joint (no staged freezing).

**Core goals (in priority order):**
1. Educational clarity — every component is readable and self-contained
2. Cost efficiency — all decisions should minimise GCP training cost
3. Open-source reproducibility — code, weights, and write-ups will be released
4. WER quality — secondary; sufficient to show meaningful loss decrease and a
   cost/WER trade-off narrative

**Project leads:** sole developer, WSL2 (Ubuntu 22.04), RTX 3060 12GB VRAM.

---

## Architecture

```
Raw audio (16kHz, up to 30s)
        ↓
[ Whisper small encoder ]   88M params   d_model=768   12 layers
        ↓  (B, 1500, 768)
[ Temporal mean pool ×4 ]   no params    (B, 1500, 768) → (B, 375, 768)
        ↓  (B, 375, 768)
[ MLP Adapter ]             ~10M params  Linear(768→2048)→GELU→Linear(2048→4096)
        ↓  (B, 375, 4096)
        ┌─────────────────────────────────────────┐
        │  audio tokens │ instruction │ transcript │   (sequence fed to Llama)
        └─────────────────────────────────────────┘
        ↓
[ Llama 3.1 8B ]            8B params    32 layers, d_model=4096
        ↓
[ LM head ]                 next-token prediction loss on transcript tokens only
```

**Total parameters:** ~8.19B

### Input sequence format

```
[audio embeddings: 375 tokens] [SEP] [instruction tokens] [SEP] [transcript tokens]
```

- Audio embeddings come from the adapter output — they are not token IDs, they are
  float vectors injected directly into Llama's residual stream before layer 0.
- Instruction is one of two prompt strings (see Training section below).
- SEP is a single separator token repurposed from Llama's reserved vocabulary
  (chosen to be absent from all LibriSpeech corpora under both normalizations).
- Loss mask: -100 for all audio positions and instruction positions;
  true token IDs for transcript positions only.

### Two instruction variants (instructable training)

```
"Transcribe the following audio without formatting."
"Transcribe the following audio with proper formatting."
```

One variant is assigned randomly per sample per training step.
The corresponding label is the transcript under the matching normalization (see Data).

---

## Confirmed Architecture Decisions

### Decision 001 — LLM Backbone: Llama 3.1 8B
Cleanest modern transformer architecture for from-scratch educational implementation.
Components: RMSNorm, RoPE, GQA (num_heads=32, num_kv_heads=8), SwiGLU MLP.
License: Llama 3.1 Community License (compatible with open-source release).
Config: 32 layers, d_model=4096, intermediate_size=14336, original vocab_size=128,000
(pruned to 40,148 for LibriSpeech training), max_seq_len=131072.

### Decision 002 — Whisper Encoder: Whisper small (encoder only)
88M params, d_model=768, 12 layers, 1500 output tokens per 30s chunk (20ms/token).
Decoder is discarded entirely. Weights initialised from OpenAI pretrained checkpoint.
Implemented from scratch — no dependency on openai-whisper or transformers.

### Decision 003 — Adaptation Layer: 2-layer MLP with temporal pooling
Temporal pool factor=4 (mean) reduces 1500→375 tokens (80ms/token).
MLP: Linear(768→2048) → GELU → Linear(2048→4096). ~10M params. Randomly initialised.

### Decision 004 — Training: joint from scratch
All three components (encoder, adapter, LLM) are trained jointly from step 0.
Differential learning rates: encoder LR=1e-5, adapter LR=1e-4, LLM LR=1e-4.
Rationale: pretrained encoder weights are stable enough; adapter and LLM need
higher LR as they are randomly initialised.

### Decision 005 — Vocabulary: pruned Llama tokenizer (40,148 tokens)
The Llama 3.1 8B tokenizer has 128,000 tokens. Only 40,148 are needed for the
LibriSpeech corpus (both transcript normalizations) plus the two instruction strings.
A contiguous re-indexing (`vocab_map.json`) maps old IDs → new IDs. Embedding matrix
and LM head are initialised at pruned size from scratch. SEP token is old ID 127,999
(Chinese character, unused in English text), mapped to new ID 40,147.

### Decision 006 — Shard format: variable-length mel
Mels are stored at their natural length `(80, T)` where `T = floor(duration_s * 100)`.
No zero-padding at shard time. The DataLoader collation function pads each batch to
`T_max = ceil(max(T_i) / 8) * 8` at training time, saving ~50% of shard disk space.
Shards are sized by total audio duration (default 30 min/shard) not sample count.

### Decision 007 — audio_lengths formula
`audio_lengths[i] = (T_mel_i // 2 + 3) // 4`
This is the adapter output token count per sample: encoder conv stride-2 halves
T_mel (floor division), then adapter mean-pool-4 reduces further (ceiling division
because partial groups are not dropped). Used to mask padding in attention.

---

## Environment

- OS: WSL2, Ubuntu 22.04
- Python: 3.11.15, venv at `~/.venvs/aidev`
  - Activate: `source ~/.venvs/aidev/bin/activate`
- PyTorch: installed for CUDA 12.6
  ```
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
  ```
- GPU: RTX 3060 12GB VRAM (local dev and smoke tests only)
- GCP: used for full training runs (instance type TBD — pending parallelisation decision)

---

## Repo Structure

```
speech-llm/
├── CLAUDE.md                        # this file
├── README.md
├── requirements.txt
├── Dockerfile
│
├── model/
│   ├── __init__.py
│   ├── whisper_encoder.py           # Whisper small encoder, fully from scratch
│   ├── adapter.py                   # Temporal pool + MLP adapter
│   └── llama.py                     # Llama 3.1 8B, fully from scratch
│
├── train.py                         # Single-GPU training loop (no parallelisation)
├── data.py                          # WebDataset dataloader for preprocessed shards
│
├── scripts/
│   ├── download_data.py             # Download full LibriSpeech splits
│   ├── download_weights.py          # Fetch Whisper small checkpoint; instructions for Llama
│   ├── build_vocab.py               # Prune Llama tokenizer to LibriSpeech vocabulary
│   ├── precompute_labels.py         # Pre-compute transcript labels for all splits (run once)
│   ├── preprocess.py                # Audio → mel + labels → WebDataset shards
│   └── create_subset.py             # Write a shard list for a reproducible training subset
│
├── smoke_test.py                    # End-to-end forward+backward pass on tiny stub model
│
├── data/
│   ├── pruned_tokenizer/            # Output of build_vocab.py (vocab_map.json etc.)
│   ├── labels.jsonl                 # Output of precompute_labels.py (all 281k samples)
│   ├── shards/                      # Output of preprocess.py (.tar files + manifest.jsonl)
│   └── subset_shards.txt            # Output of create_subset.py (10 shards for dev loop)
│
└── weights/                         # Local checkpoint storage (gitignored)
    ├── whisper_small.pt
    └── llama3.1-8b/                 # Meta checkpoint (downloaded separately)
```

---

## Data Pipeline Execution Order

**Must be run in this order** — each step depends on the previous:

```
1. scripts/download_data.py          # download all 3 LibriSpeech splits (~60 GB)
2. scripts/build_vocab.py            # scan ALL raw trans.txt files → pruned tokenizer
3. scripts/precompute_labels.py      # run full formatting pipeline once → labels.jsonl
4. scripts/preprocess.py             # audio → mel + labels → shards (per split, fast with --labels_file)
5. scripts/create_subset.py          # pick N shards for dev loop → subset_shards.txt
```

**Why build_vocab before preprocess:** vocab must cover the full 960h corpus to
guarantee no OOV tokens at training time. Scanning raw trans.txt files is fast
and requires no shards to already exist.

**Why precompute_labels before large-split sharding:** The full formatting pipeline
(EnglishTextNormalizer → PunctuationModel → _capitalize_sentences → spaCy PROPN)
takes ~1 hour per large split when run sample-by-sample. Running it once across all
splits and saving results to labels.jsonl reduces per-split sharding time to a few
minutes (O(1) dict lookup per sample via `--labels_file`).

---

## Component Specifications

### `model/whisper_encoder.py`

Full from-scratch Whisper small encoder. Zero dependency on `openai-whisper` or
`transformers`. All subcomponents implemented explicitly in this single file:
conv stem, multi-head self-attention, layer norm, MLP, learned positional
embeddings, transformer blocks.

**Architecture constants (Whisper small):**
```python
N_MELS       = 80
N_AUDIO_CTX  = 1500
N_STATE      = 768      # d_model
N_HEAD       = 12
N_LAYER      = 12
```

**Key interfaces:**
```python
class WhisperEncoder(nn.Module):
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel:    (B, 80, T)  — variable length; padded to multiple of 2 in collation
        # return: (B, T//2, 768)

    def load_openai_weights(self, checkpoint_path: Path) -> None:
        # Maps OpenAI checkpoint keys to this implementation's parameter names.
        # Asserts on every shape before loading — fail loudly on mismatch.

def log_mel_spectrogram(audio: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
    # audio:  (T,) float32, normalised to [-1, 1]
    # return: (80, T_frames) — NOT padded to 30s; T_frames = len(audio) // 160
```

**Notes:**
- Pre-LN (LayerNorm before attention and MLP, not after) — matches OpenAI implementation.
- Positional embeddings are learned, not sinusoidal — `nn.Embedding(1500, 768)`.
- Conv stem: Conv1d(80, 768, kernel=3, stride=1, pad=1) → GELU → Conv1d(768, 768, kernel=3, stride=2, pad=1).
  The stride-2 conv halves the time dimension: T_frames → T_frames // 2.


### `model/adapter.py`

Temporal pooling followed by 2-layer MLP. Simple and self-contained.

**Key interfaces:**
```python
class AudioAdapter(nn.Module):
    def forward(self, encoder_out: torch.Tensor) -> torch.Tensor:
        # encoder_out: (B, T//2, 768)  — variable-length encoder output
        # return:      (B, T//8, 4096) — T//8 = ceil(T//2 / 4) audio tokens

def prepare_input(
    adapter_out: torch.Tensor,           # (B, L_audio, 4096)  L_audio = T//8
    audio_lengths: torch.Tensor,         # (B,)  real audio token count per sample
    instruction_ids: torch.Tensor,       # (B, T_inst)
    instruction_lengths: torch.Tensor,   # (B,)
    transcript_ids: torch.Tensor,        # (B, T_trans)
    transcript_lengths: torch.Tensor,    # (B,)
    embed_layer: nn.Embedding,           # Llama's token embedding layer
    sep_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Returns:
    #   inputs:  (B, L_audio + 1 + T_inst + 1 + T_trans, 4096)  — full embedded sequence
    #   labels:  (B, L_audio + 1 + T_inst + 1 + T_trans)        — -100 except transcript
```

**Pooling implementation:**
```python
# Mean pool factor=4 with ceiling: (B, T//2, 768) → (B, ceil(T//2/4), 768)
B, T, D = x.shape
pad = (4 - T % 4) % 4
if pad:
    x = F.pad(x, (0, 0, 0, pad))
x = x.reshape(B, (T + pad) // 4, 4, D).mean(dim=2)
```


### `model/llama.py`

Full from-scratch Llama 3.1 8B. Zero dependency on `transformers`. All subcomponents
implemented explicitly in this single file: RMSNorm, RoPE, GQA attention, SwiGLU MLP,
transformer blocks, LM head.

**Architecture config (Llama 3.1 8B):**
```python
@dataclass
class LlamaConfig:
    n_layers:          int   = 32
    d_model:           int   = 4096
    n_heads:           int   = 32
    n_kv_heads:        int   = 8        # GQA
    intermediate_size: int   = 14336
    vocab_size:        int   = 40148    # pruned; set from pruned_config.json at runtime
    max_seq_len:       int   = 131072
    rms_norm_eps:      float = 1e-5
    rope_theta:        float = 500000.0  # Llama 3.1 uses 500k, not 10k
```

**Key interfaces:**
```python
class Llama(nn.Module):
    def forward(
        self,
        inputs_embeds: torch.Tensor,     # (B, S, 4096) — pre-embedded sequence
        labels: torch.Tensor | None,     # (B, S) — -100 masked; used to compute NTP loss
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Returns (logits, loss). Loss is None if labels is None.
        # Always takes inputs_embeds, never raw token IDs —
        # embedding happens in prepare_input() in adapter.py.

    def load_meta_weights(self, checkpoint_dir: Path) -> None:
        # Loads Meta's sharded Llama 3.1 8B checkpoint.
        # Asserts on every shape before loading.
```

**Notes:**
- RoPE: implement with rope_theta=500000.0 (Llama 3.1 change from 3.0's 10000.0).
- GQA: K and V projections have shape (d_model, n_kv_heads * head_dim);
  keys and values are repeated n_heads // n_kv_heads times before attention.
- Use `torch.nn.functional.scaled_dot_product_attention` for the attention
  kernel (uses FlashAttention-2 automatically when available).
- LM head shares weights with the token embedding matrix (tied embeddings).


### `data.py`

WebDataset-based dataloader consuming preprocessed shards from disk.
Does not do any preprocessing — mel computation and transcript normalization
happen offline.

**Shard format (written by `scripts/preprocess.py`):**
Each sample in a shard `.tar` contains:
```
{key}.mel.npy          float16 numpy array, shape (80, T)  — variable length
{key}.unformatted.txt  plain text, BasicTextNormalizer output
{key}.formatted.txt    plain text, full 4-pass formatted transcript
```

**Key interfaces:**
```python
def list_shards(pattern: str) -> list[str]:
    # Expand a brace range or glob pattern to a sorted list of existing local paths.
    # e.g. "data/shards/train-clean-100-{000000..000200}.tar"

def build_dataloader(
    shard_pattern: str | list[str],  # brace/glob pattern OR explicit shard list
    tokenizer_path: Path,            # path to pruned tokenizer (build_vocab.py output)
    sep_token_id: int,               # from pruned_config.json
    batch_size: int,
    num_workers: int,
    instruction_variants: list[str], # the two prompt strings
    shuffle_buffer: int = 1000,
) -> torch.utils.data.DataLoader:
    ...
```

**Batch tuple returned:**
```
(mel, audio_lengths, instruction_ids, instruction_lengths, transcript_ids, transcript_lengths)
 (B,80,T_max)  (B,)       (B,T_inst_max)       (B,)          (B,T_trans_max)      (B,)
```

**Epoch-level shard shuffling is caller's responsibility:**
```python
all_shards = list_shards("data/shards/train-clean-100-{000000..000200}.tar")
for epoch in range(n_epochs):
    epoch_shards = list(all_shards)
    random.Random(base_seed + epoch).shuffle(epoch_shards)
    loader = build_dataloader(epoch_shards, ...)
    for batch in loader:
        ...
```


### `train.py`

Single-GPU training loop. No parallelisation. This is the baseline that will be
extended for GCP multi-GPU/TPU runs in a later phase.

**Features:**
- Reads shard list from `--shards_file` (text file, one path per line — e.g. `data/subset_shards.txt`)
  or falls back to `--shards` glob pattern.
- AdamW with three parameter groups (differential LRs):
  ```python
  [
    {"params": encoder.parameters(), "lr": 1e-5},
    {"params": adapter.parameters(),  "lr": 1e-4},
    {"params": llama.parameters(),    "lr": 1e-4},
  ]
  ```
- Gradient accumulation (configurable `--accum_steps`)
- Mixed precision via `torch.amp.autocast` + `GradScaler`
- Checkpoint saving every N steps to `./checkpoints/`
- Logging: loss to stdout every step; optional W&B via `--wandb`
- `--max_steps` argument for smoke runs (e.g. `--max_steps 100`)

**CLI:**
```
python train.py \
  --shards_file    data/subset_shards.txt \
  --tokenizer      data/pruned_tokenizer/ \
  --whisper_ckpt   weights/whisper_small.pt \
  --llama_ckpt     weights/llama3.1-8b/ \
  --batch_size     4 \
  --accum_steps    8 \
  --max_steps      100 \
  --wandb
```


### `scripts/download_data.py`

Downloads all three LibriSpeech training splits using `urllib` or `requests`
(no HuggingFace datasets dependency for raw download).

**Splits to download:**
```
train-clean-100   ~6.3 GB
train-clean-360   ~23 GB
train-other-500   ~30 GB
```

Saves to a configurable `--output_dir`. Verifies MD5 checksums after download.
Prints progress with `tqdm`.


### `scripts/build_vocab.py`

Prunes the Llama tokenizer to the vocabulary actually needed for LibriSpeech training.
Scans raw `*.trans.txt` files from **all three splits** directly — no shards needed.

**Steps:**
1. Scan all `*.trans.txt` files under `--librispeech_dir` (recursive, all splits)
2. Run the full formatting pipeline on every transcript (batched, ~3–4 min):
   - `BasicTextNormalizer` → unformatted string
   - `EnglishTextNormalizer` → `PunctuationModel` → `_capitalize_sentences` → spaCy PROPN → formatted string
3. Tokenize both strings per transcript; collect union of all token IDs
4. Also tokenize the two instruction prompt strings
5. Select SEP: backward scan from token 127,999; pick highest unused ID
6. Build contiguous re-indexing; save `vocab_map.json` and `pruned_config.json`
7. Copy tokenizer files (excluding weight files) to `--output_dir`

**Result (empirical, full 960h corpus):**
- Original vocab size: 128,000
- Pruned vocab size: 40,148
- SEP old ID: 127,999 (`éĶ¦`) → new ID: 40,147

**CLI:**
```
python scripts/build_vocab.py \
  --librispeech_dir  data/librispeech/LibriSpeech/ \
  --llama_dir        /home/goivagoi/.llama/checkpoints/Llama3.1-8B/ \
  --output_dir       data/pruned_tokenizer/
```


### `scripts/precompute_labels.py`

Runs the full 4-pass formatting pipeline **once** across all LibriSpeech splits and
writes a JSONL file for fast label lookup during sharding. Without this, each of the
large splits (train-clean-360, train-other-500) would take ~1 hour to shard.

**Output format (one JSON per line):**
```json
{"key": "1234-56789-0001", "unformatted": "...", "formatted": "..."}
```

**CLI:**
```
python scripts/precompute_labels.py \
  --librispeech_dir  data/librispeech/LibriSpeech/ \
  --output           data/labels.jsonl
```

Pass `--labels_file data/labels.jsonl` to `preprocess.py` to skip model loading.


### `scripts/preprocess.py`

Converts raw LibriSpeech FLAC files into WebDataset `.tar` shards.

**Per-sample processing:**
1. Load FLAC audio via soundfile, resample to 16 kHz if needed
2. Compute log-mel spectrogram → `(80, T)` float16 (variable length, NOT padded to 30s)
3. Look up precomputed labels by key from `--labels_file` (fast path), or
   run the full 4-pass pipeline inline (slow path, used only for train-clean-100 dev work):
   - `BasicTextNormalizer` → unformatted
   - `EnglishTextNormalizer` → `PunctuationModel` → `_capitalize_sentences` → spaCy PROPN → formatted

**Output:**
- Duration-based shards: new shard opened when cumulative audio ≥ `--shard_duration_mins` (default 30 min)
- Shard naming: `{split}-{000000}.tar`
- Alongside shards: `manifest.jsonl` — one line per sample with key, shard, duration_s, n_mel_frames, unformatted, formatted

**CLI (fast, with precomputed labels):**
```
python scripts/preprocess.py \
  --input_dir   data/librispeech/LibriSpeech/train-other-500 \
  --output_dir  data/shards/ \
  --split       train-other-500 \
  --labels_file data/labels.jsonl \
  --seed        42
```

**Current state:** train-clean-100 sharded (201 shards, 28,539 samples, seed=42).
train-clean-360 and train-other-500 pending — run after precomputing labels.


### `scripts/create_subset.py`

Writes a shard list for a reproducible training subset. Simulates the shard
assignment the training loop would use for a specific rank and epoch.

**Shuffling formula:** `random.Random(base_seed + epoch).shuffle(all_shards)`
**Rank split:** `epoch_shards[rank::num_ranks]` (interleaved, matching WebDataset split_by_node)

**CLI:**
```
python scripts/create_subset.py \
  --shard_dir   data/shards/ \
  --split       train-clean-100 \
  --n_shards    10 \
  --base_seed   42 \
  --epoch       0 \
  --rank        0 \
  --num_ranks   1 \
  --output      data/subset_shards.txt
```

**Current state:** `data/subset_shards.txt` contains 10 shards (rank=0, epoch=0, seed=42).


### `scripts/download_weights.py`

1. Downloads OpenAI Whisper small checkpoint directly from OpenAI's CDN.
   Saves to `weights/whisper_small.pt`. Verifies SHA256.

2. Prints step-by-step instructions for downloading Llama 3.1 8B from Meta.
   (requires licence acceptance — cannot be automated)
   **Llama tokenizer location (local):** `/home/goivagoi/.llama/checkpoints/Llama3.1-8B/`


### `smoke_test.py`

Verifies the full forward and backward pass on a tiny stub model that fits on the
RTX 3060. Uses random weights throughout (no checkpoints needed).

**Stub Llama config for smoke test:**
```python
LlamaConfig(n_layers=2, d_model=512, n_heads=8, n_kv_heads=2,
            intermediate_size=1024, vocab_size=512)
```

**What it checks:**
1. Instantiate WhisperEncoder, AudioAdapter, Llama (stub config)
2. Print parameter count for each component and total
3. Generate a dummy batch: B=2, variable-length mel, random transcripts
4. Run full forward pass: mel → encoder → adapter → prepare_input → llama → loss
5. Call `loss.backward()`
6. Assert gradients are non-None and non-zero for at least one parameter in each component
7. Print peak VRAM usage via `torch.cuda.max_memory_allocated()`
8. Assert peak VRAM < 10GB

**Run with:**
```
source ~/.venvs/aidev/bin/activate
python smoke_test.py
```


### `Dockerfile`

Base image: `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime`

```dockerfile
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

VOLUME ["/weights", "/data"]

ENTRYPOINT ["python", "train.py"]
```

`requirements.txt` is kept minimal and pinned. No HuggingFace `transformers`
in the model implementation — `transformers` is only needed for `AutoTokenizer`
in `data.py` and `scripts/build_vocab.py`.

---

## Conventions

- **Python 3.11.15**. Type hints on all functions and methods. Docstrings on all
  classes and public functions.
- **No HuggingFace transformers in model components.** `transformers` is permitted
  only in `data.py` (tokenizer) and `scripts/` (normalizers, PunctuationModel).
- **No hardcoded paths.** Use `pathlib.Path` and CLI arguments everywhere.
- **No hardcoded secrets.** GCS bucket names, W&B keys etc. come from environment
  variables or CLI flags.
- **Fail loudly on shape mismatches.** Every weight loading function asserts shapes
  before `copy_`. A silent shape broadcast is a silent bug.
- **Code duplication across scripts is acceptable.** Each script is independently
  readable without importing from siblings.
- **Comments explain why, not what.**
- **Container-first mindset.** Code must run identically inside the Dockerfile and
  locally in the `aidev` venv.

---

## Current Status

| Component | Status |
|-----------|--------|
| Architecture decisions | ✅ Complete |
| `model/whisper_encoder.py` | ✅ Complete |
| `model/adapter.py` | ✅ Complete |
| `model/llama.py` | ✅ Complete |
| `smoke_test.py` | ✅ Complete |
| `scripts/download_weights.py` | ✅ Complete |
| `scripts/download_data.py` | ✅ Complete |
| `scripts/build_vocab.py` | ✅ Complete — pruned tokenizer at `data/pruned_tokenizer/` |
| `scripts/precompute_labels.py` | ✅ Complete — ready to run on full corpus |
| `scripts/preprocess.py` | ✅ Complete — train-clean-100 sharded (201 shards) |
| `scripts/create_subset.py` | ✅ Complete — `data/subset_shards.txt` (10 shards) |
| `data.py` | ✅ Complete |
| `train.py` | ⬜ Not started — next task |
| `Dockerfile` | ⬜ Pending (after train.py works locally) |
| Shard train-clean-360 + train-other-500 | ⬜ Pending (run precompute_labels.py first) |
| Upload shards to GCS | ⬜ Pending (Phase 2) |
| Parallelisation strategy | ⬜ Pending (Phase 2) |
| GCP infrastructure | ⬜ Pending (Phase 2) |

---

## Active Task

**Implement `train.py`** and validate the full pipeline end-to-end on `data/subset_shards.txt`.

### Step 1 — implement `train.py`

Implement in this order within the file:
1. CLI argument parsing (`argparse`) — all flags listed in the spec above
2. Load `pruned_config.json` from tokenizer dir to get `sep_token_id` and `vocab_size`
3. Instantiate `WhisperEncoder`, `AudioAdapter`, `Llama` (with pruned vocab_size)
4. Load pretrained weights: `encoder.load_openai_weights(...)`, `llama.load_meta_weights(...)`
5. Build AdamW with three parameter groups (differential LRs)
6. Build DataLoader from `--shards_file` using `build_dataloader`
7. Training loop:
   - `torch.amp.autocast` + `GradScaler` for mixed precision
   - Gradient accumulation every `--accum_steps` steps
   - `loss.backward()` → `scaler.step(optimizer)` → `scaler.update()`
   - Log loss to stdout every step
   - Save checkpoint every `--save_every` steps (default 500)
   - Stop after `--max_steps` if specified

### Step 2 — smoke-run on subset

```bash
source ~/.venvs/aidev/bin/activate
python train.py \
  --shards_file  data/subset_shards.txt \
  --tokenizer    data/pruned_tokenizer/ \
  --whisper_ckpt weights/whisper_small.pt \
  --llama_ckpt   /home/goivagoi/.llama/checkpoints/Llama3.1-8B/ \
  --batch_size   4 \
  --accum_steps  8 \
  --max_steps    20
```

### Step 3 — shard remaining splits (after train.py verified)

```bash
# 1. Generate labels for all splits (~3-4 min with GPU)
python scripts/precompute_labels.py \
  --librispeech_dir data/librispeech/LibriSpeech/ \
  --output          data/labels.jsonl

# 2. Shard train-clean-360
python scripts/preprocess.py \
  --input_dir   data/librispeech/LibriSpeech/train-clean-360 \
  --output_dir  data/shards/ \
  --split       train-clean-360 \
  --labels_file data/labels.jsonl \
  --seed        42

# 3. Shard train-other-500
python scripts/preprocess.py \
  --input_dir   data/librispeech/LibriSpeech/train-other-500 \
  --output_dir  data/shards/ \
  --split       train-other-500 \
  --labels_file data/labels.jsonl \
  --seed        42
```
