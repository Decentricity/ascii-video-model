"""Trivial baselines for next-frame ASCII prediction."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from .dataset import NextFrameDataset, discover_avm
from .train import split_by_video


def ce(pred: np.ndarray, target: np.ndarray, vocab: int) -> float:
    """Mean per-cell cross-entropy for hard predictions (0 or log(vocab) wrong)."""
    # report accuracy-based pseudo-CE: -log(acc+eps) rough; also exact match rate
    acc = (pred == target).mean()
    return float(-np.log(max(acc, 1e-8)))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--max-windows", type=int, default=512)
    args = p.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    processed = Path(cfg["data"]["processed_dir"])
    paths = discover_avm(processed)
    _, val_paths = split_by_video(paths, seed=int(cfg["train"].get("seed", 42)))
    ph, pw = int(cfg["data"]["patch_h"]), int(cfg["data"]["patch_w"])
    ctx = int(cfg["data"]["context_frames"])
    ds = NextFrameDataset(val_paths, context_frames=ctx, patch_h=ph, patch_w=pw)

    # most-common token from training set
    train_paths, _ = split_by_video(paths, seed=int(cfg["train"].get("seed", 42)))
    train_ds = NextFrameDataset(train_paths, context_frames=ctx, patch_h=ph, patch_w=pw)
    cg: Counter = Counter()
    cf: Counter = Counter()
    cb: Counter = Counter()
    for i in range(min(len(train_ds), 256)):
        s = train_ds[i]
        cg.update(s["tgt_g"].numpy().ravel().tolist())
        cf.update(s["tgt_f"].numpy().ravel().tolist())
        cb.update(s["tgt_b"].numpy().ravel().tolist())
    mc_g = cg.most_common(1)[0][0] if cg else 0
    mc_f = cf.most_common(1)[0][0] if cf else 0
    mc_b = cb.most_common(1)[0][0] if cb else 0

    stats = {
        "prev_frame": {"acc_g": [], "acc_f": [], "acc_b": []},
        "most_common": {"acc_g": [], "acc_f": [], "acc_b": []},
        "markov1": {"acc_g": [], "acc_f": [], "acc_b": []},
    }

    n = min(len(ds), args.max_windows)
    for i in range(n):
        s = ds[i]
        # previous frame = last context frame
        prev_g = s["ctx_g"][-1].numpy()
        prev_f = s["ctx_f"][-1].numpy()
        prev_b = s["ctx_b"][-1].numpy()
        tgt_g = s["tgt_g"].numpy()
        tgt_f = s["tgt_f"].numpy()
        tgt_b = s["tgt_b"].numpy()

        stats["prev_frame"]["acc_g"].append(float((prev_g == tgt_g).mean()))
        stats["prev_frame"]["acc_f"].append(float((prev_f == tgt_f).mean()))
        stats["prev_frame"]["acc_b"].append(float((prev_b == tgt_b).mean()))

        stats["most_common"]["acc_g"].append(float((np.full_like(tgt_g, mc_g) == tgt_g).mean()))
        stats["most_common"]["acc_f"].append(float((np.full_like(tgt_f, mc_f) == tgt_f).mean()))
        stats["most_common"]["acc_b"].append(float((np.full_like(tgt_b, mc_b) == tgt_b).mean()))

        # trivial per-cell markov: predict last frame (same as prev for order-1)
        stats["markov1"]["acc_g"].append(float((prev_g == tgt_g).mean()))
        stats["markov1"]["acc_f"].append(float((prev_f == tgt_f).mean()))
        stats["markov1"]["acc_b"].append(float((prev_b == tgt_b).mean()))

    summary = {
        k: {m: float(np.mean(v)) for m, v in mets.items()} for k, mets in stats.items()
    }
    summary["n_windows"] = n
    summary["val_videos"] = [p.name for p in val_paths]
    out = Path(cfg["paths"]["run_dir"]) / "baselines.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
