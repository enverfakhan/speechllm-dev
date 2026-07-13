# Part 3 — Stage 1 on the Full Dataset

*[← Part 2 — Memorization and training signals](02-training-signals.md)*

This document covers the ongoing full-scale phase: Llama 3.1 8B on a rented A100 80GB (setup in [Part 0](00-creating-the-artifacts.md)), LibriSpeech 960h, and the training strategy decided in [Part 2](02-training-signals.md). It is written as a running record and will be updated as the finalized runs complete. Experiments live in the `speechllm-instructible` W&B project; runs are compared on `runtime/cumulative_audio_hours` rather than steps, so arms with different effective batch sizes remain comparable.

---

## 1. The metric suite at full scale

Two changes accompany the move to the real model.

**Hit/miss decomposition.** Each of `loss_first_tok` and `loss_rest` is reported separately for **hit** positions (the model's argmax equals the target token) and **miss** positions (it doesn't). The decomposition separates two kinds of progress that a single mean conflates: the hit-rate rising (more positions predicted correctly) versus the miss-loss falling (the model getting *less wrong* where it still fails). Early in a stage, most movement is in the miss population; a stalling mean loss can hide that the hit population is still growing.

**The diagnostic shard.** At full scale, the EVAL DIAG shard is a **1920-sample randomized-order subset drawn from all four eval splits** (dev-clean, dev-other, test-clean, test-other), fully swept every 3 eval cycles with a persistent iterator. Per-split detail is deliberately not this signal's job — that belongs to the WER tool, which reports per split.

## 2. The batch size sweep

The per-GPU memory ceilings from Part 1 §7 say what *fits* (micro-batch 64 for Stage 1); the sweep asks what batch size is *worth paying for*. Method: short Stage-1 runs at effective batch sizes 64 / 128 / 256 (gradient accumulation over the fixed micro-batch), compared on loss versus samples seen and loss versus update steps.

Results: 128 matched 64 in sample-efficiency terms — no loss of information per sample, so still inside the critical-batch regime — while reaching the same loss in roughly half the update steps. In update-step terms, 64's eval loss also plateaued earlier than 128's. 256 showed no further gain. The Stage-1 setting: **effective batch 128, lr 1e-4**.

The sweep also estimates the return on additional parallelism before buying it. Optimal GPU count ≈ B_crit / B_per_GPU; provisioning beyond that pays for compute that no longer improves sample efficiency. At the current Stage-1 numbers this is a small multiple of one GPU — one of the reasons multi-node work is sequenced after, not before, the single-GPU program.

## 3. Making WER evaluation affordable

With real checkpoints worth evaluating, generation-based WER re-entered the picture — and at 0.5 samples/sec it was unaffordable. Two changes brought it to **4.8 samples/sec**: a dedicated finite eval dataloader that sorts samples by adapter audio-token length, so batches are length-homogeneous and short utterances don't wait on long ones during batched greedy decoding; and filtering to references ≤41 tokens (~90th percentile), capping worst-case generation length. `tools/run_wer.py` runs both instruction formats and reports them separately, with no text normalisation applied — the scores measure whether the model actually follows the formatting instruction.

## 4. Stage 1 experiments

The Stage-1 experimental question is *how long to train the adapter alone, and in what setup*. Three arms, as sparse config overrides on `base.yaml`:

- **Unformatted-only from scratch** — the simpler target and the probe arm for stage-exit questions.
- **Both formats from scratch** (`configs/both_formats_scratch.yaml`).
- **Both formats warm-started from the unformatted baseline** (`configs/both_formats_from_baseline.yaml`, via `model.init_from`).

The unformatted-only arm was used to work out what a Stage-1 exit criterion should look like: exit as soon as eval first-token loss crosses the uniform baseline, or hold the stage until model confidence reaches a target level (on the order of 0.9 likelihood on the correct token)? To bound that question from the far end, the arm was also run without exiting, to convergence, and the converged model inspected through generation:

- **Length-dependent quality.** The converged adapter-only model transcribes short utterances (below roughly 15–16 seconds) markedly better, and hallucinates significantly on longer ones — the adapter-only bridge has a practical audio-length horizon.
- **No instruction sensitivity.** Trained on a single instruction, the model ignores the instruction entirely: swapping in the other instruction at generation time produces the exact same output. In this setup, instruction sensitivity does not emerge from the pretrained LLM recognising the wording — it has to be trained in, by data that contains the contrast.

The both-formats arms have been trained but not yet deeply analysed; their comparison — instruction contrast from step 0 versus introduced after an unformatted warm start — is the next analysis item, and this section will be extended with the results.

## 5. Stage 2 — beyond the single bridge

Stage 2 means going beyond training only the adapter bridge. What has been tested so far:

- **Encoder + adapter trainable (Llama frozen):** no meaningful contribution to the health signals over adapter-only.
- **Full fine-tuning:** impractical on a single GPU — the memory ceiling forces a very small batch, and gradient accumulation does not compensate well in this regime.

The planned next design moves the adaptation *inside* the backbone instead of leaving it all at the entrance. A transformer block computes `x + f(x)`, with `f` the attention + FFN path; the idea is to extend the residual path to `α·x + β·g(x)` per block, where `g` is a small FFN, `α + β = 1`, and the adapters are active **only on audio token positions**. This gives every block a learned, position-gated correction for the audio modality while leaving the text path — and the pretrained text behavior — untouched. Llama's token embeddings stay frozen throughout, to preserve the pretrained input space the instructions rely on.

The deciding evidence: whether any of these configurations moves `eval_first_tok` in a way adapter-only training cannot.

## 6. Status

- Stage 1 complete and validated; finalized re-runs of the experimental program are in progress on the cleaned-up codebase.
- Analysis of the both-formats arms is next, followed by full WER evaluation (both formats, per split, per length bucket).
- Planned: the per-block audio-adapter Stage-2 experiment, and the multi-GPU scaling program (FSDP) sized by the sweep in §2.