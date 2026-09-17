# ascii-video-model

A small neural video model that operates **natively on colored ASCII / libcaca-style video**, rather than on pixels.

The core idea is to use libcaca (or a compatible ASCII renderer) as a **fixed perceptual encoder / handcrafted visual tokenizer**. Raw video is first converted into a time series of colored character grids. The model is then trained directly on that symbolic representation.

Instead of learning:

```text
pixels -> visual features -> latent representation -> temporal dynamics -> pixels
```

the model starts from:

```text
video -> libcaca -> glyph/color lattice -> temporal model -> glyph/color lattice
```

The output is itself valid colored ASCII video and can therefore be played **headlessly in a terminal**, without requiring a framebuffer or conventional video renderer.

A later, separate model may reconstruct or upscale selected ASCII sequences into realistic video.

---

## Why this is interesting

Traditional video models spend a large fraction of their capacity modeling visual details that may not be necessary for learning temporal structure:

- texture
- fine illumination changes
- high-frequency detail
- subpixel movement
- photographic noise
- very high spatial resolution

ASCII conversion destroys much of that information intentionally while preserving a surprising amount of useful structure:

- silhouettes
- edges
- coarse color
- approximate depth cues
- large-scale motion
- object persistence
- scene transitions
- spatial relationships

That makes colored ASCII video an unusual but potentially useful **structured lossy representation** for temporal learning.

The hypothesis is that a small model may learn useful video dynamics more efficiently when the representation has already removed much of the visual entropy.

---

## Core hypothesis

> A neural sequence model trained directly on colored ASCII video can learn temporally coherent video generation using substantially less compute than a comparable pixel-space video model.

The important claim is **not** merely that a neural network can generate ASCII art.

The interesting question is whether the ASCII representation acts as a useful **inductive bias** for video modeling.

A successful experiment should therefore compare the ASCII model against a pixel-space baseline with approximately comparable information bandwidth, parameter count, and training compute.

---

## System architecture

```text
                  TRAINING
                  ========

Raw video corpus
       |
       v
libcaca / ASCII renderer
       |
       v
Colored ASCII frame sequence
(glyph + foreground color + background color)
       |
       v
Tokenizer / patch encoder
       |
       v
Small temporal neural network
       |
       v
Next-frame / next-token prediction


                  INFERENCE
                  =========

Prompt / seed frames
       |
       v
ASCII video model
       |
       +------------------------------+
       |                              |
       v                              v
Terminal playback               Optional render pipeline
(headless-native)                     |
                                      v
                               ASCII-conditioned
                               reconstruction /
                               video upscaler
                                      |
                                      v
                                realistic video
```

---

## Representation

Each video is represented as a tensor over:

```text
(time, row, column)
```

Each cell contains at minimum:

```text
glyph
foreground color
background color
```

Conceptually:

```python
cell = {
    "glyph": int,
    "fg": color,
    "bg": color,
}
```

Possible canonical representation:

```text
T x H x W x channels
```

where channels correspond to symbolic IDs rather than raster pixels.

### Glyphs

Glyph identity must remain explicit.

Do **not** rasterize the ASCII characters back into images for the model input. Doing so would discard the main point of the experiment.

Start with a restricted libcaca-compatible glyph vocabulary.

Possible initial choices:

- printable ASCII
- CP437-style subset
- libcaca's selected character set
- custom reduced edge/density glyph vocabulary

A reduced vocabulary may improve efficiency.

### Colors

Initial experiments should use a quantized palette.

Candidates:

- ANSI 16 colors
- xterm 256 colors
- RGB332
- learned palette
- small fixed RGB codebook

The model may predict:

```text
glyph token
foreground-color token
background-color token
```

with separate output heads.

This is preferable to combining every glyph/color combination into one enormous vocabulary.

---

## Spatial tokenization

A naive cell-by-cell autoregressive model creates unnecessarily long sequences.

Example frame:

```text
80 x 45 = 3,600 cells
```

At 10 FPS:

```text
36,000 cells / second
```

Instead, group neighboring cells into spatial patches.

Suggested starting points:

```text
2 x 2 cells
4 x 4 cells
```

At 80x48 using 4x4 patches:

```text
20 x 12 = 240 patches / frame
```

At 10 FPS:

```text
2,400 patch positions / second
```

This is much easier to model.

Possible patch schemes:

1. Flatten cell attributes inside each patch.
2. Encode each patch with a tiny learned encoder.
3. Vector-quantize patch representations.
4. Build a discrete patch vocabulary from corpus statistics.

The first implementation should favor simplicity.

---

## Temporal model

The initial model should be deliberately small.

Recommended first serious range:

```text
30M - 80M parameters
```

Possible architectures:

- decoder-only Transformer
- temporal Transformer with spatial factorization
- Mamba / state-space model
- recurrent transformer
- ConvGRU-like symbolic model
- hybrid spatial encoder + temporal sequence model

The simplest useful baseline is likely:

```text
patch embedding
+ spatial position embedding
+ temporal position embedding
+ decoder-only Transformer
+ prediction heads
```

Longer-term, a state-space model may be attractive for streaming generation because the target use case includes live headless playback.

---

## Training objectives

### 1. Next-frame prediction

Given:

```text
frames t-n ... t
```

predict:

```text
frame t+1
```

This is the primary MVP task.

### 2. Autoregressive continuation

Generate future ASCII video conditioned on previous frames.

### 3. Masked spatiotemporal reconstruction

Randomly remove cells or patches and reconstruct them.

This can provide a useful auxiliary loss.

### 4. Multi-step prediction

Train the model to predict several future frames instead of only one.

This may improve longer-horizon dynamics.

### 5. Delta prediction

Because much of video is temporally redundant, experiments should test predicting changes relative to the previous frame:

```text
frame_delta(t -> t+1)
```

This may substantially reduce effective entropy.

---

## Headless-native output

A major property of the project is that the model output is itself a playable media format.

The model does not need to generate pixels.

Inference output can be streamed directly to:

- ANSI terminals
- libcaca
- ncurses
- framebuffer-independent SSH sessions
- headless Linux machines
- serial consoles
- lightweight embedded Linux devices

Conceptually:

```bash
ascii-video-model generate | ascii-video-player
```

or:

```bash
ascii-video-model generate --stream --ansi
```

This makes the system interesting independently of photorealistic reconstruction.

The ASCII video is not merely an intermediate latent representation; it is also a legitimate final output format.

---

## Two-stage rendering idea

The long-term architecture separates **content/dynamics generation** from **expensive visual synthesis**.

### Stage 1 — ASCII world model

Small model generates:

```text
characters
colors
movement
scene structure
temporal continuity
```

Output is immediately viewable.

### Stage 2 — optional visual reconstruction

A separate larger model receives the generated colored ASCII sequence and reconstructs realistic video.

Possible reconstruction inputs:

```text
ASCII glyph IDs
foreground colors
background colors
temporal context
optional text prompt
optional reference frame
```

The reconstruction model should preserve motion and composition from Stage 1 rather than inventing unrelated video.

This resembles a learned decoder for an aggressively compressed symbolic video representation.

---

## Why separate the stages?

A conventional video model must simultaneously learn:

```text
"What happens?"
+
"What does every pixel look like?"
```

This project deliberately separates those jobs:

```text
ASCII model:
    what happens, where, and how things move

reconstruction model:
    what those events look like photorealistically
```

If this works, generation of long sequences may remain inexpensive while high-quality rendering is only performed for footage worth keeping.

---

## Dataset

Input corpus:

```text
ordinary video
```

Preprocessing:

```text
decode video
-> normalize FPS/resolution
-> libcaca conversion
-> capture glyph/color grid
-> serialize symbolic frames
```

The training corpus should store the actual symbolic representation, not terminal screenshots.

### Suggested MVP dataset size

Start with:

```text
1-5 hours
```

for pipeline verification.

Then:

```text
10-20 hours
```

for the first meaningful model.

Approximate useful ranges discussed:

```text
10-20 h   : MVP / proof of concept
50-200 h  : serious generative experiments
500+ h    : broad-domain experimentation
```

These are experimental targets, not established requirements.

The optimal amount will depend heavily on:

- resolution
- FPS
- patch size
- palette
- model size
- domain diversity
- augmentation
- temporal window

---

## Approximate token scale

At roughly:

```text
80 x 45 characters
10 FPS
4 x 4-cell spatial patches
```

a 100-hour corpus can already approach the order of **hundreds of millions to roughly a billion patch-time observations**, depending on exact tokenization.

That means the representation can create a surprisingly large training sequence even from a modest number of source-video hours.

Dataset size should therefore be measured in both:

```text
hours of source video
```

and:

```text
effective training tokens
```

---

## Target development machine

Initial experiments are intended to be feasible on a single consumer workstation.

Reference hardware discussed:

```text
Intel i9-class CPU
NVIDIA RTX 4070 12 GB
32 GB RAM
Ubuntu Linux
```

Recommended training techniques:

- BF16 or FP16 mixed precision
- gradient accumulation
- gradient checkpointing
- FlashAttention / SDPA where appropriate
- memory-mapped datasets
- packed sequences
- optional 8-bit optimizer states

Expected practical model sizes:

```text
30-80M parameters:
    comfortable target

100-200M:
    plausible with memory optimization

300M+:
    increasingly inconvenient on 12 GB VRAM

1B:
    not an appropriate initial target for this machine
```

The objective is not to scale prematurely.

The representation itself is supposed to provide efficiency.

---

## First experiment

The first experiment should answer only one question:

> Can a small model learn coherent short-term motion in libcaca-space?

Use approximately:

```text
1-5 hours source video
small homogeneous dataset
low resolution
8-12 FPS
small palette
30M-ish model
short context
```

Train a next-frame predictor.

Evaluate generated rollouts visually in a terminal.

Do not begin with photorealistic reconstruction.

---

## Recommended first configuration

A reasonable starting point:

```yaml
video:
  width_chars: 80
  height_chars: 48
  fps: 10

representation:
  glyph_vocab: reduced_ascii
  color_palette: xterm256
  foreground: true
  background: true

patching:
  width: 4
  height: 4

model:
  type: transformer
  parameters: ~30M
  precision: bf16

training:
  objective:
    - next_patch
    - next_frame
  context_frames: 8-16
  gradient_accumulation: true

inference:
  mode: autoregressive
  output: ansi
  terminal_playback: true
```

These are proposed experimental defaults rather than fixed architectural decisions.

---

## Baselines

A credible experiment needs baselines.

### Baseline A — downsampled RGB

Train an equal-size model on aggressively downsampled pixel video.

Try to match overall information rate.

### Baseline B — quantized pixel video

Use a similarly restricted color palette but no glyph transformation.

### Baseline C — ASCII without color

Measure how much foreground/background color contributes.

### Baseline D — frame-independent ASCII

Compare full temporal learning against trivial frame prediction/interpolation.

---

## Evaluation

Pixel metrics alone are inappropriate because the model operates in symbolic ASCII space.

Potential metrics:

### Symbol accuracy

- glyph cross entropy
- foreground-color accuracy
- background-color accuracy

### Temporal stability

Measure unnecessary character/color flicker between frames.

### Motion consistency

Track coarse objects or connected regions across generated sequences.

### Rollout degradation

Evaluate quality as generation horizon increases.

### Compression efficiency

Compare quality against:

```text
training FLOPs
parameter count
tokens/sec
memory use
source information rate
```

### Human evaluation

Because the native output is directly viewable, human judgments of:

- coherent movement
- recognizable objects
- scene stability
- interesting generation
- long-term consistency

are useful.

---

## Key experiment

The strongest empirical test is:

> Given equal parameter count and approximately equal compute, does a model trained on libcaca-derived symbolic video learn more coherent temporal dynamics than a model trained on an equally low-bandwidth pixel representation?

If yes, the project demonstrates something broader than ASCII generation:

> Structured, handcrafted lossy representations can provide useful inductive bias for neural video models.

---

## Risks

### Loss of identity/detail

ASCII conversion may discard information needed to distinguish similar objects.

### Temporal flicker

libcaca may choose different glyphs for tiny pixel changes, creating token noise.

Mitigations:

- deterministic renderer settings
- temporal smoothing before conversion
- hysteresis when selecting glyphs/colors
- train on frame deltas
- palette stabilization

### Sequence length

Even ASCII video creates many tokens.

Mitigations:

- patching
- factorized spatial/temporal attention
- hierarchical tokens
- delta encoding
- state-space models

### Reconstruction ambiguity

Multiple realistic images can map to the same ASCII frame.

The later upscaler therefore cannot literally recover lost information; it must generate a plausible realization conditioned on the symbolic sequence.

This should be treated as conditional synthesis, not lossless decompression.

### Model learns libcaca artifacts

The network may learn quirks of the renderer rather than useful visual dynamics.

This is partly acceptable—the ASCII domain is itself the target medium—but baseline comparisons are necessary before claiming broader significance.

---

## Interesting extensions

### ASCII-native world model

Add actions or control tokens:

```text
state_t + action_t -> state_t+1
```

This turns the project from passive video prediction toward an interactive world model.

### Text-conditioned generation

Condition generation on natural-language prompts.

### Camera-control tokens

Add:

```text
pan
tilt
zoom
forward
rotate
```

### Multiscale ASCII

Generate coarse ASCII first, then refine with denser glyph grids.

### Learned glyph vocabulary

Instead of conventional ASCII, learn a compact character-like basis optimized for video.

This becomes a learned terminal-friendly visual codec.

### Audio

Train a synchronized low-bandwidth audio representation alongside ASCII frames.

### Streaming model

Use recurrent/state-space architecture so arbitrarily long terminal video can be generated without repeatedly attending over the full history.

---

## Philosophy

The intentionally weird part is also the useful part.

ASCII is normally treated as a rendering gimmick: pixels are converted to characters for human display.

Here, the direction is reversed conceptually.

The ASCII grid becomes the model's **native visual universe**.

Characters are not visual decorations around an underlying raster.

They are the actual visual primitives of the learned world.

---

## Project stages

### Phase 0 — format

Define and document the canonical colored-ASCII video representation.

### Phase 1 — corpus pipeline

Build:

```text
video -> deterministic libcaca symbolic frames -> dataset
```

### Phase 2 — tiny predictor

Train a small model on 1-5 hours and demonstrate next-frame prediction.

### Phase 3 — autoregressive video

Generate multi-second terminal-playable sequences.

### Phase 4 — scaling experiment

Train approximately 30-80M models on 10-20+ hours.

### Phase 5 — baselines

Run bandwidth/compute-matched pixel comparisons.

### Phase 6 — longer context

Explore state-space models, delta encoding, and hierarchical tokenization.

### Phase 7 — reconstruction

Only after the ASCII generator itself works, experiment with realistic-video reconstruction.

---

## Success criteria for MVP

The MVP succeeds when:

1. raw video can be deterministically converted to the symbolic dataset format;
2. frames can be reconstructed perfectly back to terminal playback from the stored representation;
3. a small neural model trains on the representation;
4. generated rollouts contain recognizable temporally coherent motion;
5. generation can stream directly to a headless terminal;
6. results are demonstrably better than trivial copying/interpolation.

Photorealistic output is **not** required for MVP success.

---

## License

TBD.
