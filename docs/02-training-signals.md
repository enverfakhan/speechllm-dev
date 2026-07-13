# Part 2 — Memorization and Training Signals

*[← Part 1 — Building the pipeline](01-building-the-pipeline.md) · [Part 3 — Stage 1 on the full dataset →](03-stage1-training.md)*

Part 1 ended with a pipeline that could memorize on demand. This document covers what came next, still entirely on the local RTX 3060 with the stub Llama: growing the memorization setup into a small generalisation testbed, discovering that WER is unusable as a training-time signal, building a generation-free diagnostic suite in its place, and using that suite to make the training-strategy decisions — freezing, learning rates, initialisation, staging — that the full-scale runs in Part 3 inherit.

---

## 1. From memorization to a generalisation testbed

The memorization experiments used a single speaker — about 12 minutes of audio. The natural next step keeps the frame and changes the question: hold out part of that speaker's data as an eval set, add **five more speakers** to the training side, and ask whether anything learned transfers to the held-out utterances. The added speakers were deliberately all women, matching the original speaker, so that the eval set stays close to the training distribution and the experiments measure generalisation in the easiest setting first.

The resulting local corpus: **6 speakers, 638 utterances**, with the held-out slice of the original speaker (the same slice used during the memorization runs) as the evaluation set. Small enough for fast iteration on a 12GB card; real enough to expose the difference between fitting and learning.

## 2. WER is not a training-time signal

The first approach to tracking progress was to periodically generate transcriptions and compute WER. On a partially-trained model this fails in practice: hypotheses either never emit the SEP token — generation runs to the token cap on every sample — or emit nothing at all. Between the degenerate outputs and the cost of autoregressive decoding, WER measures almost nothing about a model in this regime while consuming most of the eval budget.

The conclusion: **training health has to be observable from the forward pass alone**, with generation-based metrics (WER) reserved for models that can already produce non-degenerate output. `metrics.py` implements that suite; every signal below is emitted for both a fixed held-out diagnostic shard (**EVAL DIAG**) and the current training micro-batch (**TRAIN DIAG**), every 10 steps.

## 3. The diagnostic suite

The central design idea is splitting the transcript loss by position:

**`loss_first_tok`** — cross-entropy on the *first* transcript token only, the position immediately after the final instruction SEP. No transcript context exists yet, so the model can only do better than corpus word-frequency priors by using the audio. This makes first-token loss the most direct measure of whether the audio bridge is contributing information.

**`loss_rest`** — cross-entropy averaged over the remaining transcript tokens. These positions can lean on the pretrained LM prior, so rest-loss descends much faster; it tracks language modelling quality conditioned on (possibly ignored) audio.

The split matters because a speech-LLM has a failure mode ordinary LMs don't: it can reduce total loss substantially while ignoring the audio entirely, purely by exploiting text statistics. Aggregate loss hides this; the first/rest split exposes it.

Around that core, the supporting signals:

| Signal | What it tracks | Failure it catches |
|---|---|---|
| first > rest ordering flag | first token should be *harder* than the rest | inverted ordering (`⚠ collapse signal`): model exploiting a spurious first-position correlation instead of audio |
| per-component grad norms (`enc / ada / llm`) + enc/llm ratio | how aggressively each component moves | adapter norm ~0 with non-zero LR (mask or precision bug); monotonic LLM norm growth (instability); ratio calibrates encoder LR |
| first-token top-5 tokens | what the model actually predicts at the audio-grounded position | top-5 stuck on `' the'`, `' of'`, `' and'` — unigram priors, audio ignored |
| SEP fraction | share of positions where SEP is argmax | persistent >15–20%: collapsing toward empty/short outputs |
| first-position entropy (normalised) | concentration of the output distribution | near 1.0 late in training: model knows nothing; rapid collapse toward 0: overconfident memorisation |
| mean max logit | logit scale over time | sudden jumps: numerical trouble ahead (this is the signal that exposed the initialisation bug in Part 1 §4) |
| token budget | transcript tokens per micro-step | context for per-step loss variance |

## 4. Baselines: giving the loss numbers meaning

Loss values are interpreted against reference points computed from the corpus itself. `tools/compute_baselines.py` scans the training shards and computes the loss of progressively stronger audio-blind models:

| Baseline | Loss (6-speaker corpus) | A model that... |
|---|---|---|
| Uniform | 10.600 (= ln 40148) | knows nothing |
| Unigram | 6.802 | knows word frequencies |
| First-token | 4.545 | knows which words start utterances — the best possible first-token loss *without audio* |
| Bigram | 2.717 | knows pairwise transitions (computed in-sample: optimistic) |

These turn the two loss curves into a milestone ladder: `eval_rest` below uniform is trivial; below unigram means the model uses sequential context; `eval_first_tok` below the first-token baseline is the unambiguous signature that **audio is informing prediction** — nothing audio-blind can beat 4.545 at that position.

## 5. Target-driven experimentation

With signals and baselines in place, the local experiments had a defined objective: **drive `eval_rest` across the unigram baseline before the eval curves diverge** (the point where a 638-utterance corpus is exhausted and further training is memorisation). Variables tested against that target, a few configurations at a time:

- **Encoder: frozen vs. unfrozen.** Unfrozen runs at encoder LR 1e-6 and 1e-7 produced enc/llm gradient-norm ratios of 0.15–0.33 — the encoder barely moved while adding cost and variance. Freezing it lost nothing at this scale.
- **Adapter learning rate.** Swept; 1e-4 was the working point.
- **Adapter initialisation.** Three variants: Kaiming, the GPT-2-style near-zero output init from Part 1 §4, and a **PCA initialisation** of the first linear layer — pooled encoder outputs over the corpus, SVD, top-2048 principal components as the initial weight matrix (`tools/compute_adapter_pca_init.py`), starting the adapter as the projection the encoder's feature geometry already suggests. PCA did not outperform the GPT-2-style init and was not adopted.
- **Adapter-only vs. full fine-tuning.** Updating everything from step 0 let the LLM absorb the training signal through text statistics while the randomly-initialised adapter was still noise; adapter-only training forced the optimisation pressure onto the audio bridge first.

The configuration that hit the target: **frozen encoder, adapter-only, adapter LR 1e-4** — `eval_rest` reached the unigram boundary around step 1500 and held flat rather than reversing, with only transient, self-correcting collapse signals (5 over 2000 steps). Two more findings from these runs carried forward:

- **`accum_steps=1` on small corpora.** Gradient accumulation runs showed isolated 3–5× gradient-norm spikes traced to batch composition artefacts (the largest, an LLM norm of 77, was followed by a measurable eval regression). Accumulation is for reaching statistically necessary batch sizes, not a default.
- **A data ceiling at this scale.** `eval_first_tok` never crossed the unigram baseline at 638-utterance scale under any configuration — the first-token top-5 stayed pinned to high-frequency function words. Audio grounding at that position appears to need more data than six speakers supply; the full 960h run exists to answer that question.

## 6. What the local era decided

The output of this phase is the training strategy itself:

1. **Staged training**, with an adapter-only first stage (frozen encoder, frozen Llama) — a randomly-initialised adapter between two pretrained networks must be brought online before anything else is allowed to move. This is the origin of the Stage 1 / Stage 2 structure in `stages.py`, where each stage declares its trainable modules, learning rates, and advancement criterion in the config.
2. **A generation-free health suite with corpus-derived baselines** as the primary steering instrument, with WER demoted to a post-hoc evaluation artifact (`tools/run_wer.py`).
3. **Concrete hyperparameter and initialisation choices** — frozen encoder in Stage 1, adapter LR 1e-4, near-zero adapter output init (PCA init tested, not adopted) — each attached to an observed result rather than convention.

A full-dataset training run with the stub model, still local, verified the pipeline at real data scale. Then the pipeline moved to RunPod and the real model: [Part 3](03-stage1-training.md).