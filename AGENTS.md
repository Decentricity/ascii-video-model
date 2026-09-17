# AGENTS.md

NeurASCII is an experimental neural video model whose native representation is **colored ASCII video**. The model operates directly on symbolic glyph/color grids and generates terminal-playable video.

> The network must operate on symbolic character/color grids directly. Do not silently turn this back into ordinary raster-video modeling.

# CURRENT AGENT PRIORITY — READ THIS FIRST

## Phase C blocker: expanding static blob / autoregressive attractor

The current full-scale experiment has now moved beyond the earlier `ApplyEyeMakeup` smoke test.

Current run context:

- dataset: full `bitmind/UCF101Fullvideo` corpus;
- approximately **7 GB** of video;
- **13,451 videos / 101 classes**;
- model: approximately **32M parameters**;
- warm-started from the earlier `videos_all` weights;
- `max_steps=10000`;
- best checkpoint observed around **step 8000**, validation loss approximately **2.849**;
- free-running examples have been generated from diverse UCF101 class seeds;
- the rollout shown in the current Phase C page uses **greedy decoding (`temperature=0`)**.

### Empirical failure pattern

This is the current problem agents should optimize against:

1. The seeded beginning of a rollout can retain recognizable scene structure.
2. During autonomous generation, connected colored regions progressively become **larger and coarser**.
3. Distinct objects/regions merge into broad ASCII blobs.
4. Once a large blob forms, the rollout becomes increasingly **static** rather than maintaining meaningful motion.
5. The result can remain locally stable while being globally wrong: a giant blocky attractor replaces the original scene dynamics.

In shorthand, this is the **"bug blob" failure mode: the blob grows, then freezes**.

Do **not** treat a good next-frame validation loss as success if free-running rollout enters this state.

### Working hypothesis

Treat this as a hypothesis to test, not established fact.

The leading explanation is an **autoregressive attractor / oversmoothing failure** caused by some combination of:

- one-step teacher forcing creating train/inference exposure mismatch;
- small token errors accumulating during free rollout;
- the model drifting off the training manifold and conditioning on its own degraded predictions;
- full-frame loss being dominated by static or slowly changing regions;
- under uncertainty, large low-frequency glyph/color regions being safer predictions than detailed moving structure;
- greedy decoding repeatedly selecting those modal predictions and driving the sequence toward a fixed point;
- patch scale or spatial bottlenecks encouraging coarse structures to merge.

Dataset size alone is **not** currently the main lever: the failure persisted after scaling to the full UCF101 corpus.

## Immediate experiments, in priority order

### 1. Train explicitly for free-running rollouts

Pure one-step teacher forcing is no longer enough.

Implement and compare:

- multi-step rollout loss;
- scheduled sampling;
- feeding a controlled fraction of model-generated frames back into training context;
- curriculum rollout horizons that increase during training.

Always compare teacher-forced next-frame quality against autonomous rollout quality. If teacher-forced metrics improve while rollout stability does not, the target task has not improved.

### 2. Predict changes, not entire frames

Test **frame-delta / changed-cell prediction**.

The model should explicitly learn what changes from `t` to `t+1` rather than repeatedly reconstructing every stable cell. This may preserve scene identity while focusing capacity and loss on actual motion.

Useful variants:

- changed-cell mask + new glyph/color values;
- residual/delta representation;
- copy-from-previous-frame as the default action with sparse edits;
- patch-level delta prediction.

### 3. Stop static background from dominating the objective

Measure the fraction of unchanged versus changed cells and test motion-aware weighting.

Candidates:

- upweight cells/patches that changed between ground-truth frames;
- upweight rare glyph/color transitions;
- separate losses for static preservation and motion prediction;
- report loss on changed cells separately from loss on unchanged cells.

Do not solve this by simply encouraging more random change. The desired result is **structured motion**, not flicker.

### 4. Test spatial granularity

Current blob growth may be amplified by patch/coarse spatial representation.

Compare:

- smaller patches;
- current patch size;
- hierarchical coarse-to-fine prediction;
- local refinement after coarse prediction.

A model should not gain rollout stability merely by merging nearby structures into increasingly large blocks.

### 5. Keep narrow-domain sanity runs

Maintain a small `ApplyEyeMakeup` or one/few-clip configuration alongside full UCF101.

Use it to answer a simpler question:

> Can this exact architecture/training objective sustain coherent autonomous motion when the visual distribution is narrow and memorization is easy?

If a deliberately overfit one/few-clip model also enters the blob attractor, fix the objective/architecture before blaming dataset diversity.

### 6. Sweep decoding, but do not mistake sampling fixes for training fixes

Run controlled inference sweeps over:

- greedy decoding;
- temperature;
- top-k;
- top-p if implemented.

Lower or nonzero stochasticity may change the onset of collapse, but the goal is a model whose learned dynamics are stable rather than a sampler that cosmetically hides failure.

Store exact decoding settings next to every generated sample.

# Required rollout diagnostics

Every serious train/eval should produce:

1. teacher-forced next-frame metrics;
2. free-running samples at fixed short/medium/long horizons, e.g. 16 / 48 / 96 frames;
3. per-step degradation curves;
4. frame-to-frame token-change percentage;
5. changed-cell versus unchanged-cell loss;
6. connected-region / blob-size statistics if practical;
7. motion or changed-patch persistence over rollout time;
8. exact sampler settings;
9. comparison with a previous-frame-copy baseline;
10. a one/few-clip overfit sanity run.

A useful additional diagnostic for the current failure is **largest connected region size versus generated frame number**. If the largest region consistently expands while total motion decays, that quantitatively captures the observed bug-blob attractor.

# Core project constraints

## 1. ASCII is the native representation

Input data is not screenshots of terminal output.

Each cell should retain symbolic values such as:

```text
glyph ID
foreground color ID
background color ID
```

The model receives these values directly. Do not rasterize characters into images before training unless implementing an explicitly named comparison baseline.

## 2. Output must remain terminal-playable

The primary output is colored ASCII video and should be playable headlessly without requiring X11, Wayland, browser rendering, or GPU display output.

ANSI/libcaca/ncurses-compatible streaming is desirable.

## 3. Photorealistic reconstruction is downstream

Do not introduce a diffusion/video renderer into the core model merely to improve apparent quality.

The project intentionally separates:

```text
temporal/content generation
```

from:

```text
photorealistic synthesis
```

A later model may conditionally reconstruct realistic video from generated ASCII sequences, but NeurASCII must remain independently useful and testable.

## 4. Target a single consumer workstation

Assume approximately:

```text
Ubuntu Linux
NVIDIA RTX 4070 12 GB
32 GB system RAM
Intel i9-class CPU
```

Prefer experiments that fit this machine. Use BF16/FP16, gradient accumulation, checkpointing, SDPA/FlashAttention, efficient dataset IO, and 8-bit optimizer states where useful.

Do not hide OOMs by silently changing meaningful experimental settings.

## 5. Keep scientific claims falsifiable

ASCII efficiency is a hypothesis, not an assumption.

Eventually compare against roughly compute/information-matched baselines such as:

- downsampled RGB video;
- quantized-color pixel video;
- ASCII without color.

# Representation and model guidance

A frame is a symbolic lattice over `(row, column)` with glyph and color attributes, extended over time.

Keep glyph identity explicit. Prefer separate embeddings/heads for glyph, foreground color, and background color unless a combined vocabulary demonstrably works better.

Candidate model families include:

- causal Transformer;
- factorized spatial-temporal Transformer;
- recurrent Transformer;
- Mamba/state-space models;
- ConvGRU-like symbolic models.

Streaming inference is a long-term goal, so recurrent/state-space approaches remain interesting after the current rollout objective is stabilized.

# Patching

At 80x48, a frame contains 3,840 cells; at 10 FPS that is 38,400 cell positions/sec. Spatial patches can reduce sequence length, but patching must be treated as an experimental tradeoff because the current coarse-blob failure may interact with spatial granularity.

Keep patch size configurable and benchmark at least two granularities before concluding that larger patches are desirable.

# Dataset discipline

Split by **source video**, not adjacent clips, to avoid temporal leakage.

Track:

- total duration;
- source video count;
- frame count;
- effective patch/token count;
- glyph and color entropy;
- average frame delta;
- changed-cell fraction;
- class/domain composition.

Do not assume more diverse data fixes rollout collapse. The full UCF101 Phase C experiment is evidence that scale/diversity alone is insufficient for the current architecture/objective.

# Baselines

Maintain trivial baselines:

1. previous-frame copy;
2. interpolation where meaningful;
3. most-common token prediction;
4. simple cell/patch Markov predictor.

A learned model that produces attractive first frames but collapses into a static blob has not meaningfully beaten a persistence baseline for long-horizon generation.

# Evaluation philosophy

Single-frame accuracy is insufficient.

Report both token metrics and rollout behavior:

- glyph cross entropy;
- foreground/background color loss;
- exact cell/patch accuracy;
- changed-cell accuracy;
- frame-to-frame change rate;
- flicker rate;
- motion consistency;
- largest connected-region growth;
- rollout stability;
- time-to-collapse / time-to-attractor;
- throughput, VRAM, checkpoint size, inference speed.

Always save terminal-playable samples at fixed training intervals.

# Avoid these failure modes

## Accidental rasterization

Do not feed terminal screenshots into the main model.

## Premature photorealism

Do not spend core-model effort on realistic upscaling while temporal dynamics remain unstable.

## Premature scale

Do not respond to the current failure merely by increasing parameter count or dataset size.

## Misleading validation success

A lower one-step validation loss does not imply better free-running video.

## Frozen-video "stability"

Do not optimize temporal regularization so aggressively that the model simply copies the prior frame. Stable motion is the goal, not still images.

## Sampling-only fixes

A sampler that delays the attractor is useful diagnostically but does not by itself prove the learned dynamics are correct.

# Agent behavior

When changing architecture, objective, representation, or inference:

1. state the hypothesis being tested;
2. keep previous configurations reproducible;
3. change one major variable at a time where practical;
4. add/update tests for serialization and tensor shapes;
5. save representative autonomous rollouts, not only teacher-forced metrics;
6. log throughput and VRAM for training-impacting changes;
7. preserve baselines;
8. explicitly report whether the **blob-growth/static-attractor** failure improved, worsened, or merely changed appearance.

When uncertain, favor the smallest experiment that can falsify the hypothesis.

The project should remain weird, simple, measurable, terminal-native, and runnable on one enthusiast workstation.
