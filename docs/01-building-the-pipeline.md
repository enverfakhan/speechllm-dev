# Part 1 — Building the Pipeline

*[← Part 0 — Creating the artifacts](00-creating-the-artifacts.md) · [Part 2 — Memorization and training signals →](02-training-signals.md)*

Part 0 ended with artifacts on disk. This document is about turning them into a working training pipeline: the forward pass, the sequence design, the memory engineering that fits an 8B-parameter model onto a single A100 80GB, and the experiments that verify the pipeline is correct before spending any GPU-hours on real training on the rented VMs.

Where a design choice was motivated by an observed failure (zero gradients, an OOM, a logit explosion), the failure is described alongside the choice.

---

## 1. Develop cheap, validate cheap, then scale

The pipeline was developed on a local RTX 3060 (12GB) using a **stub mode** built into the config system: `model.stub: true` swaps the full Llama for a tiny randomly-initialised one (6 layers, d_model=512, ~34M parameters) while keeping everything else identical — the same encoder, adapter code, sequence assembly, loss, and training loop.

```yaml
model:
  stub: true
  stub_dims: { d_model: 512, n_layers: 6, ... }
```

The stub surfaces the scale-independent bugs — wrong loss masks, padding leaking into attention, shape mismatches, silent dtype casts, dataloader misalignment — on the same code path the 8B run takes. Model construction lives in one shared function (`build.py:build_models`) used by training and evaluation alike, so the two cannot drift apart in initialisation logic.

The sequencing for the whole project follows from this: **local stub → single-GPU cloud → (future) multi-GPU**, advancing only when the previous step is validated.

## 2. The forward pass, end to end

One training step, with shapes:

```
WebDataset shard stream                              (data.py)
  → mel          (B, 80, T_max)   float32, zero-padded to a multiple of 8
  → audio_lengths, instruction_ids/lengths, transcript_ids/lengths

WhisperEncoder(mel)                                  (model/whisper_encoder.py)
  → (B, T_mel//2, 768)            20ms per token

AudioAdapter(enc_out)                                (model/adapter.py)
  → mean-pool ×4 → Linear(768→2048) → GELU → Linear(2048→4096)
  → (B, ⌈T/4⌉, 4096)              80ms per token — Llama-ready "audio tokens"

prepare_input(...)                                   (model/sequence.py)
  → per-sample: [audio | SEP | instruction | SEP | transcript | SEP]
  → inputs (B, L_max, 4096), labels (B, L_max) with -100 outside transcript

Llama(inputs, labels)                                (model/llama.py)
  → cross-entropy on transcript tokens + the final SEP only

scaler.scale(loss).backward() → scaler.step(optimizer)
```

The dataloader (`data.py`) does no preprocessing — mels and both transcript variants were computed offline in Part 0, so a training-time "sample" is a float16 numpy load, a tokenizer lookup into the pruned vocabulary, and padding bookkeeping. Each batch carries explicit *length* tensors alongside the padded ID tensors, which is what makes the next section possible.

One design decision at this layer: **epoch-level shard shuffling is owned by the training loop, not by WebDataset.** Each epoch, the shard list is reshuffled with a seed derived from `(base_seed, stage, epoch)` and a fresh loader is built. This is more code than flipping WebDataset's internal shuffle flags, but it makes the data order reproducible, inspectable, and — later — shareable with a prefetch process.

## 3. Sequence design

The sequence layout is the contract between the audio and text worlds:

```
[ audio tokens ] [SEP] [ instruction ] [SEP] [ transcript ] [SEP]
 ── continuous ─────── ── embedded discrete tokens ──────────────
    audio embeddings                        └── loss lives here ──┘
```

Three decisions:

**Loss is masked to the transcript plus the trailing SEP.** Labels are `-100` everywhere else — audio and instruction positions are conditioning, not targets.

**The trailing SEP is the EOS target.** The model is trained to predict SEP after the last transcript token, giving greedy decoding a learned stopping signal.

**Padding never enters a real token's causal history.** `prepare_input` strips each sample's padding *before* concatenation (using the per-sample length tensors), assembles the true sequence, and only then pads the *batch* to its longest member — at the end, where causal attention can't see it and the `-100` labels exclude it from the loss. The batched greedy decoder (`EvalPrefixBatch` in `sequence.py`) upholds the same invariant during generation, which is harder: prefixes differ in length across a batch, so generated tokens are *inserted at each sample's own write position* rather than appended at the tensor's end. The result is that the model never — in training or generation — attends to a padding zero sitting between real tokens, so the generation-time input distribution matches the training distribution exactly.

The assembly is a plain Python loop over the batch. This is deliberate: it is the most correctness-critical function in the codebase, and it is not the throughput bottleneck (the 8B forward pass is), so it is kept legible.

## 4. From-scratch models, pretrained weights

`model/whisper_encoder.py` and `model/llama.py` implement their architectures completely — no `transformers`, no `openai-whisper` in any model code. The Whisper encoder is the conv stem + 12 pre-LN transformer blocks; the Llama implementation covers RMSNorm, rotary position embeddings, grouped-query attention (32 query / 8 KV heads), and SwiGLU MLPs. Both route attention through `F.scaled_dot_product_attention` (Flash Attention under autocast).

Pretrained weights are loaded by explicit key mapping with shape assertions (Part 0's principle). For Llama, `vocab_map.json` remaps both the token embedding rows and the output-head rows from the original 128K vocabulary into the pruned 40K one, so the pretrained text representations carry over intact.

**Initialisation.** In stub mode there is nothing to remap — the tiny Llama is randomly initialised end to end — and the stub-era memorization experiments (§6) surfaced a bug through it: with default initialisation, logits exploded within the first optimizer steps. The fix was GPT-2-style scaled initialisation (`std = 0.02/√n_layers`), and the confirmation was quantitative: after the change, training loss started at ~10.6, which is `ln(40148)` — the uniform-distribution loss over the pruned vocabulary. The model now started from the uniform distribution, which located the original problem in the output layer's initial scale rather than in the training dynamics downstream of it.

The same reasoning was then extended to the adapter. Its output projection writes into Llama's residual stream — functionally, it behaves like one more residual branch of the model — so it was given the matching treatment: near-zero initialisation (`std = 0.02/√6`, following Llama's residual scaling convention), so that the adapter initially contributes almost no signal to the pretrained residual stream and grows into it during training. Early-training diagnostics were measurably healthier after the change.

## 5. Fitting 8B parameters into 80GB

The memory constraints, in the order they were hit:

### 5.1 The budget

Master weights are kept in **fp32** (the mixed-precision story below explains why this is forced, not chosen): ~29GB for the vocabulary-pruned model. A standard fp32 AdamW would add another **12 bytes per parameter** of optimizer state — ~88GB, alone exceeding the card. Full-precision Adam on a single A100 is not possible; either weight precision (see 5.3 for why that fails) or optimizer-state precision had to be reduced.

### 5.2 8-bit AdamW (bitsandbytes)

The optimizer states — Adam's two moment tensors — are where the memory goes, and they are also where precision is most compressible, because they are *smoothed statistics*, not weights. bitsandbytes' `AdamW8bit` stores both moments in 8 bits per element via block-wise quantisation with per-block fp32 scaling (~6 bytes/param total vs 12), cutting optimizer state to ~44GB; the update math itself runs in fp32, so the optimizer is a drop-in replacement.

### 5.3 Mixed precision — and why bf16 lost

Compute runs under fp16 autocast; master weights stay fp32; a `GradScaler` multiplies the loss before backward and unscales before the optimizer step. Each ingredient was verified by removing it:

**Without GradScaler, the adapter receives exactly zero gradient in Stage 1.** The gradient signal has to flow backward through 32 frozen Llama layers before reaching the adapter. Along that path it attenuates; in fp16, whose smallest normal magnitude is ~6×10⁻⁵, it attenuates *below representability* and flushes to zero. The scaler's loss multiplication shifts the entire gradient path up into representable range, and unscaling recovers correct magnitudes afterward. This was observed directly as `ada grad norm = 0.000` in the training diagnostics.

**bf16 removes the need for a scaler, but did not work here.** bf16 has fp32's exponent range, so gradient-path underflow largely disappears and no scaler is needed (`GradScaler` is incompatible with bf16). But bf16 buys that range by giving up mantissa: 8 bits against fp16's 11. In testing, bf16 Stage-1 runs also produced zero adapter gradients — the small gradient signals surviving the frozen-layer path lost their significance to rounding instead of to underflow. The configuration adopted as a firm constraint: **fp32 master weights + fp16 autocast + GradScaler**.

One subtlety is documented as an open experiment rather than resolved: GradScaler protects the *gradient path*, but not the *update step* — if `lr × update` is too small relative to an fp16 master weight, the addition rounds to zero and the parameter silently stops learning. This is one of the reasons master weights are fp32; "fp16 master weights" is deferred as a documented post-single-GPU experiment.

### 5.4 Gradient checkpointing

With weights and optimizer accounted for, activations are what remain. Gradient checkpointing here buys roughly **4× less activation memory for ~20–30% extra compute**. Measured on the A100 80GB:

| | without ckpt | with ckpt |
|---|---|---|
| Stage 1 (adapter-only), batch=1 peak | 46.1 GB | 30.0 GB |
| Stage 1, max batch | 15 | **72** |
| Stage 2 (full model), max batch | 5 | **12** |

A detail in the Stage-1 row: checkpointing saves 15GB at batch=1 even though the Llama layers are frozen and build no gradients. Without checkpointing, PyTorch still stores every intra-layer intermediate of the frozen layers — autograd doesn't know at forward time that nothing downstream will need them. With `checkpoint(..., use_reentrant=False)`, frozen-layer forwards effectively run under `no_grad`, and those intermediates are never kept. For Stage 1, gradient checkpointing is therefore load-bearing rather than optional: without it, the practical batch size collapses and the memory plan for training breaks. (Only transformer layers are wrapped; embeddings, final norm, and the LM head have negligible activation footprints and wrapping them buys nothing.)

### 5.5 A note on allocator fragmentation

Not every OOM reflects a real memory ceiling. Variable-length batches produce variable-sized allocations, and PyTorch's caching allocator can fragment — free VRAM exists, but not contiguously. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to eliminate this class of OOM; without it, measured maximum batch sizes come out lower than the hardware actually supports.

## 6. Proving correctness before paying for training

Two verification gates sit between "the code runs" and starting paid training:

**A smoke test** (`smoke_test.py`) runs the full pipeline — real shards, stub model, real loss — end to end, catching integration breaks in minutes on the local card.

**A memorization experiment**: train on a deliberately tiny subset past the point of overfitting and require near-exact reproduction of the training transcripts (config: `configs/stage1-overfit-5000.yaml`). Failures at this gate map directly to pipeline bugs — loss plateauing above ~zero: loss mask or data alignment; loss near zero with garbage generations: training and generation disagree about the sequence contract; adapter gradient norm of zero: the precision stack (this is where the GradScaler finding of §5.3 surfaced, and the initialisation bug of §4). Real training started only after the pipeline could memorize on demand.

## 7. How big a batch?

The last preparatory question: what batch size does the memory actually allow? `tools/find_max_batch.py` binary-searches the maximum micro-batch per stage (and per gradient-accumulation setting), using worst-case sequence lengths (30s audio → 375 audio tokens) so the answer holds for the longest batches the real data can produce. An `--eval` mode does the same for inference memory, which has a completely different profile (no optimizer, no gradients, no checkpointing).

Measured maxima — 72 (Stage 1) and 12 (Stage 2) — became production settings of **64 and 8**: round numbers below the ceiling, leaving headroom for the length variance of real LibriSpeech batches. Whether those batch sizes are *good for optimisation* — as opposed to merely *fitting* — is an empirical question about critical batch size, answered by the sweep in [Part 3](03-stage1-training.md).

---

*Next: [Part 2 — Memorization and training signals](02-training-signals.md): the diagnostic suite, the loss baselines, and the local experiments that set the training strategy.*