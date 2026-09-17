# Quickstart (implemented MVP)

Native next-frame model on colored ASCII grids (libcaca → symbolic `.avm.npz`).

## Setup

```bash
cd ~/neurascii
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

Requires system `libcaca`, `ffmpeg`, and a CUDA Torch install (or CPU).

## Data: Apply* makeup clips

Download ApplyEyeMakeup + ApplyLipstick from Hugging Face (normalizes Windows-style paths):

```bash
python scripts/download_ucf101_videos.py --apply-only
```

**POC — keep classes separate:**

```bash
python -m neurascii.preprocess data/raw/UCF101-Videos/test/ApplyEyeMakeup \
  --out-dir data/processed/poc_eye --class-prefix ''
python -m neurascii.baselines --config configs/train_poc_eye.yaml
python -m neurascii.train --config configs/train_poc_eye.yaml

python -m neurascii.preprocess data/raw/UCF101-Videos/test/ApplyLipstick \
  --out-dir data/processed/poc_lipstick --class-prefix ''
python -m neurascii.train --config configs/train_poc_lipstick.yaml
```

**Initial smoke — merge Apply* only:**

```bash
python -m neurascii.preprocess data/raw/UCF101-Videos \
  --out-dir data/processed/smoke_apply --class-prefix Apply
python -m neurascii.train --config configs/train_smoke_apply.yaml
```

## Play / generate

```bash
python -m neurascii.play data/processed/poc_eye/*.avm.npz --max-frames 30
python -m neurascii.generate \
  --checkpoint runs/poc_eye/ckpt_last.pt \
  --seed-avm data/processed/poc_eye/SOME.avm.npz \
  --out samples/rollouts/out.avm.npz --steps 32
python -m neurascii.play samples/rollouts/out.avm.npz
```

## What is not in git

`.gitignore` excludes `data/`, `runs/`, `*.pt`, videos, and generated rollouts. Push **code only** to GitHub. Upload trained weights to Hugging Face later (see `PHASE2.md`).
