# Part 0 — Creating the Artifacts

*[← README](../README.md) · [Part 1 — Building the pipeline →](01-building-the-pipeline.md)*

Before any training code can run, four artifacts have to exist: **pretrained weights**, **the dataset**, **the labels**, and **the tokenizer**. This document walks through how each is created, and closes with the compute setup the project runs on — including where the original plan changed.

Everything here is reproducible from the `tools/` directory. The commands are shown inline; the focus is on the reasoning around them.

---

## 1. Pretrained weights

The model code in `model/` is written from scratch; the weights are pretrained. The first artifact is the pair of checkpoints.

```bash
python tools/download_weights.py --output_dir weights/
```

**Whisper small** is fetched directly from OpenAI's CDN — no `openai-whisper` package dependency, just a URL and a SHA256 check. The script verifies the hash after download (and before skipping a re-download of an existing file), so a truncated or corrupted checkpoint fails here instead of surfacing later as a shape error or silently degraded features.

**Llama 3.1 8B** can't be automated: Meta requires license acceptance per user. The script prints step-by-step instructions for the HuggingFace Hub route (accept license → token → `huggingface-cli download`, pulling the original Meta-format checkpoint, not the HF-converted one — the from-scratch `model/llama.py` loads Meta's `consolidated.*.pth` layout directly).

Weight loading throughout the codebase is **explicit and shape-asserted**: `WhisperEncoder.load_openai_weights()` maps every checkpoint key to a module parameter by hand and raises on any shape mismatch, so a wrong or reshaped tensor fails at load time rather than broadcasting silently.

## 2. The dataset

```bash
python tools/download_data.py
```

LibriSpeech 960h — `train-clean-100`, `train-clean-360`, `train-other-500`, plus the dev/test splits. The choice was made in the project premise (clean, open, large enough to experiment, small enough to train for multiple epochs on a budget), so Part 0's job is just to get it on disk: ~1000 hours of FLAC plus transcripts, roughly 60GB extracted.

## 3. The labels — making "instructible" real

This is where the project's core idea becomes a data engineering task. The model is trained on two instructions:

- *"Transcribe the following audio without formatting."* → lowercase, no punctuation
- *"Transcribe the following audio with proper formatting."* → punctuation, sentence caps, proper-noun caps

LibriSpeech transcripts are natively unformatted (all-caps, no punctuation), so the **unformatted** target is a one-liner: Whisper's `BasicTextNormalizer`. The **formatted** target has to be manufactured. The pipeline is four passes:

1. `EnglishTextNormalizer` (Whisper) — canonicalises numbers and symbols to spoken form
2. `PunctuationModel` (deepmultilingualpunctuation, a BERT token classifier) — restores punctuation
3. Sentence-initial capitalisation — a regex pass, because the punctuation model's decode step doesn't reliably capitalise
4. spaCy `en_core_web_sm` PROPN tagging — capitalises proper nouns

The ordering matters: proper-noun tagging runs *after* punctuation and sentence caps are in place, because the POS tagger performs noticeably better with sentence structure to condition on.

These labels are not perfect — a BERT punctuator and a small spaCy model make mistakes. What the training objective needs is *consistency*: the same input distribution mapping to the same output convention, so the model has a learnable formatted style. The goal is an instructible model, not a SOTA formatter.

### Precomputing the labels

The first version ran this pipeline inline, per-sample, inside the sharding loop. For `train-clean-100` that was tolerable; for `train-other-500` it made preprocessing prohibitively slow — BERT inference one utterance at a time is a poor access pattern for the hardware.

The fix: **precompute the labels once, look them up during sharding.**

```bash
python tools/precompute_labels.py \
  --librispeech_dir data/librispeech/LibriSpeech/ \
  --output data/labels.jsonl
```

`precompute_labels.py` streams every transcript in the corpus through the pipeline in **batched** form — all texts in a chunk go through the BERT punctuator in a single batched pipeline call, and through spaCy via `nlp.pipe()`. The output is one JSONL record per utterance: `{"key": ..., "unformatted": ..., "formatted": ...}`.

Sharding then does an O(1) dict lookup per sample instead of a model inference, cutting shard-build time from ~1 hour to a few minutes per large split — and as a bonus, the sharding step no longer needs the NLP models installed at all.

## 4. The tokenizer — pruning 128K down to ~40K

Llama 3.1's tokenizer has 128,256 entries. The LibriSpeech corpus, under *both* normalisations plus the two instruction strings, uses about 40K of them. Carrying the full vocabulary means carrying a 128K × 4096 embedding matrix and a 128K-way output softmax for tokens that can never appear in a label.

```bash
python tools/build_vocab.py \
  --librispeech_dir data/librispeech/LibriSpeech/ \
  --llama_dir       weights/llama3.1-8b/ \
  --output_dir      data/pruned_tokenizer/
```

The script:

1. Streams every raw transcript from every split (directly from `*.trans.txt` files — no dependency on sharding having run first)
2. Applies **both** normalisation pipelines to every transcript and collects the union of token IDs — a token used only by formatted text still has to survive the prune
3. Adds the token IDs of the two instruction strings
4. Selects a **SEP token**: the first Llama special token that doesn't appear anywhere in the collected set. SEP is the sequence delimiter and EOS target (Part 1 covers the sequence layout), so it must be a token the corpus can never emit organically
5. Writes a contiguous re-indexing (`vocab_map.json`: old ID → new ID) plus `pruned_config.json` with the final vocab size and SEP ID

The result is a ~40K vocabulary — roughly a **3× reduction in embedding and LM-head parameters**, which matters when the whole point is fitting an 8B-model training run on one GPU.

Two honest caveats, both discussed later in the series: the pruned model can only ever emit LibriSpeech-vocabulary text (an accepted constraint for this project's scope), and remapping token IDs interacts with pretrained-embedding loading in a way that had consequences (Part 1).

## 5. The shards — WebDataset

Training streams data as WebDataset `.tar` shards rather than reading FLAC files at train time:

```bash
python tools/preprocess.py \
  --librispeech_dir data/librispeech/LibriSpeech/ \
  --output_dir      data/shards/ \
  --labels_file     data/labels.jsonl \
  --seed            42
```

Per sample, the script loads the FLAC (via soundfile), resamples to 16 kHz if needed, computes the **log-mel spectrogram at its natural length** — using the exact same `log_mel_spectrogram()` from `model/whisper_encoder.py` that the model sees, so there is no train/preprocess skew — and stores it as float16 alongside both transcript variants:

```
1234-56789-0001.mel.npy           # (80, T) float16 — natural length, not padded
1234-56789-0001.unformatted.txt
1234-56789-0001.formatted.txt
```

Three decisions:

**Mels are precomputed, in float16.** Recomputing a deterministic transform every epoch is wasted compute; float16 halves the storage, and the log-scaled, normalised values sit comfortably in fp16 range. Total shard footprint: ~55GB for 960h.

**Samples are globally shuffled before sharding (`--seed 42`).** LibriSpeech is organised by speaker and chapter. Sharding in directory order would produce shards that are internally single-speaker, which would put an unreasonable burden on the training-time shuffle buffer to produce diverse batches. Shuffling at *shard-creation* time means every shard is a cross-speaker mix, and the runtime shuffle buffer can stay small.

**Shards are sized by audio duration, not sample count** (default: 30 minutes of audio per shard). Duration is a better unit of work than sample count when utterance lengths vary this much — and the *size* itself is a three-way balance. Shards must be small (and therefore numerous) enough that shard-level shuffling produces meaningful re-randomization across epochs — and, in the planned multi-node setting, enough shards to distribute disjoint subsets across workers without starving any of them. They must be large enough to preserve the point of sharding in the first place: long sequential reads instead of per-sample random I/O. And they must not be so large that a single shard's internal ordering dominates long stretches of training and imprints spurious structure on the batch sequence. Thirty minutes of audio — a few hundred utterances per shard, ~2000 shards for the full corpus — sits comfortably inside all three constraints.

Additionally, samples longer than 30 seconds are **skipped, not truncated** — they exceed Whisper's positional embedding range, and trimming audio without trimming the transcript would inject label noise. There are only a handful in the corpus; dropping them is cheaper than corrupting them. A `manifest.jsonl` records every sample's key, split, shard, duration, and both labels, so downstream tools (eval subset construction, corpus statistics, the loss baselines in Part 2) never need to re-open the tars.

## 6. Compute — the plan vs. what happened

The original infrastructure plan was GCP: A100 80GB Spot instances for development, scaling to multi-GPU later, with data in a GCS bucket in the same region as the compute.

Part of the plan survived. **GCS remained the storage backbone** — training data shards, pretrained weights, and every checkpoint live in a bucket. **The compute side changed**: A100 80GB Spot capacity was chronically unavailable (stockouts across zones, in every region tried), which makes iterative development impractical. GCP also gates GPU access behind multiple quota approvals.

The pragmatic answer was **RunPod**: instant provisioning, per-second billing, and a single A100 80GB at **$1.49/hr** (≈ **$1.56/hr all-in** with the attached network volume). The working setup:

- **A100 80GB pod** with a **400GB network volume** mounted at `/workspace` — deliberately small, sized to hold the shards, weights, and active checkpoints and nothing more, since volume storage bills continuously even when no pod is running
- **GCS as the source of truth**: the pod pulls shards and weights from the bucket via a read-scoped service account, and pushes checkpoints back after each run — so the pod itself is fully disposable
- **Runs are launched manually**: SSH into the pod, `python training.py --config configs/<experiment>.yaml`. No orchestration layer — at single-GPU scale, an orchestrator would be complexity without payoff. (Multi-node training, where that calculus changes, is planned future work.)

This split — *rented, disposable compute + durable cloud storage* — is what makes spot-priced experimentation safe: losing a pod loses nothing but the time since the last checkpoint upload.

**Cost so far: ~€475**, with a projected **~€700 total** for the complete project including the finalized training runs — covering end-to-end training of an 8B-parameter multimodal model, experimental program included. The memory-engineering choices that make this possible are the subject of [Part 1](01-building-the-pipeline.md).

---

## Artifact inventory

At the end of Part 0, the following exists (locally and mirrored in GCS):

| Artifact | Produced by | Size |
|---|---|---|
| `weights/whisper_small.pt` | `tools/download_weights.py` (SHA256-verified) | ~460MB |
| `weights/llama3.1-8b/` | manual (Meta license) | ~16GB |
| `data/labels.jsonl` | `tools/precompute_labels.py` | ~100MB |
| `data/pruned_tokenizer/` | `tools/build_vocab.py` | small |
| `data/shards/*.tar` + `manifest.jsonl` | `tools/preprocess.py` | ~55GB |

*Next: [Part 1 — Building the pipeline](01-building-the-pipeline.md), where these artifacts meet the training loop.*