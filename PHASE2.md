# Phase 2 — deferred after UCF101-Videos Apply* POC / full-repo smoke

Do not start these until Phase 1 (Apply* few-clip POC + full Apply*/repo smoke) passes.

## Scale-up corpora

1. **True-full UCF101** — [`bitmind/UCF101Fullvideo`](https://huggingface.co/datasets/bitmind/UCF101Fullvideo) or ~7GB ZIP (~13k clips).
2. **RedLetterMedia** — 8 long-form videos under `~/Movies/redlettermedia/` (Pi download); domain transfer / long-horizon stress. Push **code only** to GitHub; never push raw video.
3. **Something-Something V2** — only with explicit disk/compute budget.

## Research baselines / ablations

- Equal-bandwidth downsampled RGB / quantized-pixel baselines
- ASCII without color; bg-only ablation
- Patch size 2×2 vs 4×4; delta-frame prediction; longer context
- Streaming SSM / Mamba for live terminal generation
- Stage-2 photoreal reconstruction (separate model; out of MVP)

## Hugging Face model publish

After a trained checkpoint is worth sharing: upload weights + model card to Hugging Face **separately** from the GitHub code repo. Do not put multi-GB checkpoints on GitHub.
