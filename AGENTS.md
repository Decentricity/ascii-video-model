# AGENTS.md

This repository contains an experimental neural video model whose native representation is **colored ASCII video**.

Agents working on this repository should preserve the central research idea:

> The neural network must operate on symbolic character/color grids directly. Do not silently turn the problem back into ordinary raster-video modeling.

## Core constraints

### 1. ASCII is the native representation

Input data is not screenshots of terminal output.

Each frame must retain symbolic values such as:

```text
glyph ID
foreground color ID
background color ID
```

The model should receive these values directly.

Do not rasterize characters into images before training unless implementing an explicitly named comparison baseline.

### 2. Output must remain terminal-playable

The model's primary output is colored ASCII video.

A valid generated sequence should be playable in a headless terminal without requiring:

- X11
- Wayland
- framebuffer rendering
- browser rendering
- GPU display output

ANSI/libcaca/ncurses-compatible streaming is desirable.

### 3. Realistic-video reconstruction is a separate downstream model

Do not introduce a diffusion/video decoder into the core MVP.

The project intentionally separates:

```text
temporal/content generation
```

from:

```text
photorealistic synthesis
```

The ASCII generator must be useful and testable independently.

### 4. Optimize for a single RTX 4070 12 GB workstation

Assume the initial training environment is approximately:

```text
Ubuntu Linux
NVIDIA RTX 4070 12 GB
32 GB system RAM
Intel i9-class CPU
```

Avoid architectures whose minimum viable experiment requires datacenter GPUs.

Initial model target:

```text
~30M parameters
```

Reasonable exploratory range:

```text
30M-80M
```

Larger experiments may be added later.

### 5. Establish baselines

Do not claim efficiency advantages solely because ASCII feels compressed.

At minimum plan comparisons against:

- downsampled RGB video
- quantized-color pixel video
- ASCII without color

Try to match parameter count, compute, and approximate information bandwidth.

---

# Development priorities

Work in this order unless there is a strong technical reason not to.

## Priority 1 — canonical file format

Define a stable representation for ASCII video.

Requirements:

- frame dimensions
- frame rate metadata
- glyph vocabulary metadata
- foreground palette metadata
- background palette metadata
- deterministic indexing
- random-access frames if practical
- efficient sequential training reads
- lossless round-trip between stored symbolic frame and terminal display

Candidate storage schemes include:

- NumPy arrays
- compressed NumPy archives
- Zarr
- HDF5
- custom binary framing
- memory-mapped arrays

Favor simple, inspectable formats first.

The first format does not need to be maximally compressed.

## Priority 2 — deterministic video-to-ASCII conversion

Create a reproducible preprocessing pipeline:

```text
source video
-> decode
-> FPS normalization
-> resize
-> libcaca conversion
-> symbolic extraction
-> serialized dataset
```

Pin or record every renderer setting that can influence output.

Avoid nondeterministic glyph/color selection.

If libcaca APIs do not expose the symbolic output cleanly, investigate:

- direct canvas APIs
- export formats
- terminal/ANSI capture
- deterministic reimplementation of the renderer

Do not use OCR on rendered ASCII.

## Priority 3 — player

Before training anything, implement a player that reads the dataset format and reproduces the ASCII video.

Suggested CLI:

```bash
python -m ascii_video_model.play sample.caca-video
```

Desired flags:

```text
--fps
--loop
--no-color
--start
--end
```

The stored representation and the displayed output must agree exactly.

## Priority 4 — dataset tooling

Implement:

```text
dataset inspection
train/validation split
clip extraction
statistics
token-frequency counts
palette statistics
glyph statistics
temporal-change statistics
```

Report:

- total source duration
- frame count
- effective patch count
- estimated training tokens
- storage size
- average frame delta
- glyph entropy
- color entropy

## Priority 5 — trivial baselines

Before neural training implement:

1. previous-frame copy
2. frame interpolation if meaningful
3. most-common token prediction
4. simple per-cell Markov model

The neural model must outperform trivial baselines.

## Priority 6 — neural MVP

Start with next-frame prediction.

Recommended initial settings:

```yaml
frame:
  width: 80
  height: 48
  fps: 10

patch:
  width: 4
  height: 4

model:
  target_parameters: 30000000
  precision: bf16

context:
  frames: 8
```

Treat these as defaults, not immutable requirements.

---

# Model architecture guidance

Prefer a simple architecture first.

Good starting architecture:

```text
glyph embedding
color embeddings
      |
      v
cell/patch encoder
      |
      v
spatial + temporal positional encoding
      |
      v
causal Transformer
      |
      +--> glyph prediction
      +--> foreground-color prediction
      +--> background-color prediction
```

Do not combine all glyph/color combinations into a single giant vocabulary without benchmarking why that is superior.

Separate heads make the representation easier to reason about.

Possible later architectures:

- Mamba/state-space
- recurrent transformer
- factorized spatial-temporal transformer
- hierarchical transformer
- ConvGRU symbolic model

Streaming inference is an eventual design goal, so architectures with efficient recurrent state deserve later investigation.

---

# Patching

Avoid immediately modeling each terminal cell as a full autoregressive timestep.

At 80x48 there are:

```text
3840 cells/frame
```

At 10 FPS:

```text
38,400 cell positions/sec
```

Spatial patches can reduce sequence length dramatically.

Start with:

```text
4x4 cells
```

but keep patch dimensions configurable.

Potential implementations:

### Flattened patch features

Simple and easy to debug.

### Learned patch encoder

Small MLP/attention/conv-like symbolic encoder.

### Discrete patch vocabulary

Potentially very efficient if common patterns repeat.

Do not introduce vector quantization until the simpler version is working.

---

# Training objectives

Implement in incremental order.

## Objective A — next frame

Input previous N frames, predict frame N+1.

## Objective B — autoregressive continuation

Roll predictions back into the context and generate longer sequences.

## Objective C — masked reconstruction

Optional auxiliary objective.

## Objective D — multi-frame horizon

Predict multiple future frames.

## Objective E — frame deltas

Experiment with predicting symbolic changes rather than complete frames.

Frame-delta prediction is particularly interesting because ASCII video may contain strong temporal redundancy.

---

# Training engineering

Use VRAM carefully.

Preferred techniques:

- BF16 when supported
- FP16 otherwise
- PyTorch SDPA / FlashAttention where available
- gradient accumulation
- gradient checkpointing
- fused optimizer if stable
- optional 8-bit optimizer
- compile only after correctness is established

Log:

```text
training loss
validation loss
glyph loss
foreground-color loss
background-color loss
tokens/sec
VRAM usage
wall-clock step time
```

Never hide OOM problems by silently reducing meaningful experimental settings.

Record configuration changes.

---

# Dataset scale

Use staged scaling.

## Smoke test

```text
minutes of video
```

Purpose:

- verify format
- verify training loop
- intentionally overfit tiny sample

## Prototype

```text
1-5 hours
```

Purpose:

- determine whether coherent motion is learnable

## MVP

```text
10-20 hours
```

Purpose:

- serious first generative model

## Later research

```text
50-200 hours
```

and potentially:

```text
500+ hours
```

Do not preprocess hundreds of hours before validating the model on a small corpus.

---

# First expected research result

The first meaningful demo should be:

```text
source ASCII clip
model-conditioned continuation
```

played directly in the terminal.

Target clip length can initially be only a few seconds.

Success means the generated sequence exhibits visibly nontrivial temporal structure rather than:

- static copying
- random flicker
- palette noise
- rapid collapse
- training-example replay

---

# Evaluation

Track both token metrics and behavior.

## Token metrics

- glyph cross entropy
- foreground-color cross entropy
- background-color cross entropy
- exact cell accuracy
- patch accuracy

## Temporal metrics

- frame-to-frame change rate
- flicker rate
- rollout stability
- motion consistency

## Compute metrics

- training FLOPs if practical
- steps/sec
- patches/sec
- VRAM
- checkpoint size
- inference speed

## Human inspection

Always save terminal-playable samples at fixed training intervals.

A statistically improving loss is not sufficient if visual rollouts remain useless.

---

# Baseline experiment design

The strongest eventual comparison is:

```text
ASCII representation
vs.
low-bandwidth pixel representation
```

with roughly equal:

```text
parameter count
training time
compute
source duration
information rate
```

Do not compare a 30M ASCII model against a vastly larger or higher-bandwidth pixel baseline and infer architectural superiority.

---

# Reconstruction / upscaling

This is explicitly a later phase.

The eventual system may look like:

```text
ASCII sequence
+ optional prompt
+ optional keyframe
        |
        v
realistic-video reconstruction model
```

The reconstruction model cannot recover information destroyed by ASCII conversion.

Describe this as:

```text
conditional reconstruction
```

or:

```text
plausible visual synthesis
```

not lossless decoding.

Temporal consistency is more important than recreating the exact original source pixels.

---

# Research questions worth testing

Agents may create issues or experiments around:

1. Does color help temporal prediction enough to justify its token cost?
2. Does background color materially improve representation?
3. Which glyph vocabulary minimizes temporal flicker?
4. Does 2x2 patching outperform 4x4 per compute?
5. Are learned patches better than direct symbolic patches?
6. Does delta prediction improve long rollouts?
7. Does temporal smoothing before libcaca conversion improve learnability?
8. Does a state-space model outperform a Transformer for long streams?
9. How much source-video diversity is needed before generalization appears?
10. Does ASCII-space modeling outperform equal-bandwidth quantized pixels?

---

# Avoid these failure modes

## Accidental rasterization

Do not feed terminal screenshots into the main model.

## Premature photorealism

Do not spend MVP effort on the upscale/reconstruction stage.

## Premature scale

Do not train a huge model before a tiny one can overfit and generate.

## Uncontrolled renderer changes

A preprocessing change can alter the entire token distribution.

Version renderer configurations.

## Dataset leakage

Split by source video, not by random adjacent clips, otherwise near-identical neighboring frames can appear in both training and validation.

## Misleading metrics

Do not report only single-frame accuracy.

A model can score well while producing terrible autoregressive rollouts.

## Unjustified scientific claims

Treat all efficiency advantages as hypotheses until measured against baselines.

---

# Suggested repository layout

```text
ascii-video-model/
├── README.md
├── AGENTS.md
├── pyproject.toml
├── configs/
│   ├── preprocess.yaml
│   └── train_tiny.yaml
├── src/
│   └── ascii_video_model/
│       ├── __init__.py
│       ├── format.py
│       ├── preprocess.py
│       ├── dataset.py
│       ├── tokenizer.py
│       ├── model.py
│       ├── train.py
│       ├── generate.py
│       └── play.py
├── scripts/
│   ├── preprocess_video.py
│   ├── inspect_dataset.py
│   └── generate_sample.py
├── tests/
│   ├── test_format.py
│   ├── test_roundtrip.py
│   └── test_dataset.py
└── samples/
```

---

# Coding style

Prefer:

- Python
- PyTorch
- typed APIs where useful
- small CLI tools
- explicit configuration
- deterministic defaults
- Linux-first implementation

Keep dependencies minimal.

Do not introduce a web stack for functionality that can remain CLI-native.

---

# Agent behavior

When modifying architecture or data representation:

1. explain the experimental reason in the commit or PR;
2. keep previous configurations reproducible;
3. add/update tests for serialization;
4. produce a sample output where applicable;
5. record throughput/VRAM if the change affects training;
6. avoid deleting an existing baseline merely because a newer approach looks better.

When uncertain, favor the smallest experiment that can falsify the idea.

The project should remain weird, simple, measurable, and runnable on one enthusiast workstation.
