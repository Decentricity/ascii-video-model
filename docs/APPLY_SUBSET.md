# POC note: ApplyEyeMakeup vs ApplyLipstick

UCF101 makeup classes in [`bitmind/UCF101-Videos`](https://huggingface.co/datasets/bitmind/UCF101-Videos):

- `ApplyEyeMakeup`
- `ApplyLipstick`

## POC (keep separate)

Train and evaluate **one class at a time**. Do not mix eye-makeup and lipstick in the same POC run.

```bash
# Eye makeup POC
python -m neurascii.preprocess data/raw/UCF101-Videos/test/ApplyEyeMakeup \
  --out-dir data/processed/poc_eye --class-prefix '' --config configs/preprocess.yaml
python -m neurascii.baselines --config configs/train_poc_eye.yaml
python -m neurascii.train --config configs/train_poc_eye.yaml

# Lipstick POC (separate run / separate processed dir)
python -m neurascii.preprocess data/raw/UCF101-Videos/test/ApplyLipstick \
  --out-dir data/processed/poc_lipstick --class-prefix '' --config configs/preprocess.yaml
python -m neurascii.train --config configs/train_poc_lipstick.yaml
```

## Initial smoke test (merge Apply*)

Only at smoke time, combine both classes:

```bash
python -m neurascii.preprocess data/raw/UCF101-Videos \
  --out-dir data/processed/smoke_apply --class-prefix Apply --config configs/preprocess.yaml
python -m neurascii.train --config configs/train_smoke_apply.yaml
```

The HF dump sometimes stores `test\ApplyEyeMakeup\file.avi` as a **literal filename**; `scripts/download_ucf101_videos.py` normalizes those into real directories.
