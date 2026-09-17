#!/usr/bin/env python3
"""Phase C Fullvideo stability diagnostics (shaped by Phase B rollout findings).

For a few Fullvideo seeds:
  - free-running rollouts at short/medium/long horizons (default 16/48/96)
  - teacher-forced next-frame metrics on the same seeds
  - frame-to-frame token-change stats (via generate sidecar)

Outputs under samples/rollouts/ (gitignored). Sampler defaults to greedy (temp=0).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from neurascii.dataset import NextFrameDataset, frames_to_patches
from neurascii.format import load_avm
from neurascii.generate import generate_rollout
from neurascii.model import build_model_from_config

HORIZONS = (16, 48, 96)
WANT_SEEDS = 4
SAMPLER_TEMP = 0.0  # greedy; Phase C cap is 0.7


def pick_ckpt(*candidates: Path) -> Path | None:
    for p in candidates:
        if p.is_file():
            return p
    return None


def parse_seed(path: Path) -> tuple[str, str]:
    stem = path.name
    if stem.endswith(".avm.npz"):
        stem = stem[: -len(".avm.npz")]
    if "__" in stem:
        cls, name = stem.split("__", 1)
    else:
        cls, name = "unknown", stem
    return cls, name


def pick_diverse_seeds(processed: Path, want: int = WANT_SEEDS) -> list[tuple[str, str, Path]]:
    by_class: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(processed.glob("*.avm.npz")):
        cls, _ = parse_seed(p)
        by_class[cls].append(p)
    if not by_class:
        raise SystemExit(f"no seeds in {processed}")
    classes = sorted(by_class.keys())
    if len(classes) >= want:
        step = len(classes) / want
        chosen = [classes[int(i * step)] for i in range(want)]
    else:
        chosen = classes
    out: list[tuple[str, str, Path]] = []
    for c in chosen[:want]:
        # Prefer a clip long enough for context+96 if possible
        clips = by_class[c]
        best = clips[0]
        for p in clips:
            try:
                v = load_avm(p)
            except Exception:
                continue
            if v.T >= 8 + 16:
                best = p
                if v.T >= 8 + 96:
                    break
        name = parse_seed(best)[1]
        out.append((c, name, best))
    return out


@torch.no_grad()
def teacher_forced_metrics(
    ckpt_path: Path,
    seed_paths: list[Path],
    *,
    device: torch.device,
    max_windows: int = 64,
) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    ph, pw = int(cfg["data"]["patch_h"]), int(cfg["data"]["patch_w"])
    ctx = int(cfg["data"]["context_frames"])

    # Probe n_patches from first seed
    v0 = load_avm(seed_paths[0])
    g0, _, _ = frames_to_patches(v0.glyphs, v0.fg, v0.bg, ph, pw)
    n_patches = g0.shape[1]
    model = build_model_from_config(cfg, n_patches=n_patches).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = NextFrameDataset(seed_paths, context_frames=ctx, patch_h=ph, patch_w=pw)
    if len(ds) == 0:
        return {"error": "no teacher-forced windows (clips too short?)", "n_windows": 0}

    losses, acc_g, acc_f, acc_b = [], [], [], []
    n = min(len(ds), max_windows)
    for i in range(n):
        batch = ds[i]
        batch_t = {k: v.unsqueeze(0).to(device) for k, v in batch.items()}
        out = model.loss(**batch_t)
        losses.append(float(out["loss"]))
        acc_g.append(float(out["acc_g"]))
        acc_f.append(float(out["acc_f"]))
        acc_b.append(float(out["acc_b"]))

    return {
        "n_windows": n,
        "loss_mean": float(np.mean(losses)),
        "acc_g_mean": float(np.mean(acc_g)),
        "acc_f_mean": float(np.mean(acc_f)),
        "acc_b_mean": float(np.mean(acc_b)),
        "sampler_note": "teacher_forced (argmax heads; not free rollout)",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed/fullvideo"))
    p.add_argument("--out-dir", type=Path, default=Path("samples/rollouts"))
    p.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    p.add_argument("--n-seeds", type=int, default=WANT_SEEDS)
    p.add_argument("--temperature", type=float, default=SAMPLER_TEMP)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    ckpt = args.checkpoint or pick_ckpt(
        _ROOT / "runs/fullvideo/ckpt_best.pt",
        _ROOT / "runs/fullvideo/ckpt_last.pt",
    )
    if not ckpt:
        raise SystemExit("no fullvideo checkpoint")
    processed = args.processed_dir
    if not processed.is_dir():
        raise SystemExit(f"missing {processed}")

    device = torch.device(args.device)
    top_k = args.top_k if args.top_k and args.top_k > 0 else None
    seeds = pick_diverse_seeds(processed, want=args.n_seeds)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "checkpoint": str(ckpt),
        "sampler": {
            "mode": "greedy" if args.temperature <= 0 else "temperature",
            "temperature": float(args.temperature),
            "top_k": top_k,
        },
        "horizons": list(args.horizons),
        "seeds": [],
        "rollouts": [],
        "teacher_forced": None,
    }

    print(f"CKPT={ckpt}", flush=True)
    print(f"SEEDS={[f'{c}/{n}' for c, n, _ in seeds]}", flush=True)
    print(f"SAMPLER temp={args.temperature} top_k={top_k}", flush=True)

    for i, (cls, name, path) in enumerate(seeds, 1):
        summary["seeds"].append({"class": cls, "name": name, "path": str(path)})
        for h in args.horizons:
            out = out_dir / f"fullvideo_diag_{i:02d}_{cls}_h{h:03d}.avm.npz"
            print(f"--- free rollout {cls}/{name} horizon={h} -> {out.name}", flush=True)
            avm_path, diag = generate_rollout(
                ckpt,
                path,
                out,
                steps=h,
                device=device,
                temperature=args.temperature,
                top_k=top_k,
            )
            row = {
                "seed_class": cls,
                "seed_name": name,
                "horizon": h,
                "avm": str(avm_path),
                "sidecar": diag["sidecar"],
                "sampler": diag["sampler"],
                "token_change": diag["token_change_summary"],
            }
            summary["rollouts"].append(row)
            tc = diag["token_change_summary"]
            print(
                f"    mean_any_change={tc['mean_any_change']:.4f} "
                f"glyph={tc['mean_glyph_change']:.4f}",
                flush=True,
            )

    print("--- teacher-forced metrics ---", flush=True)
    tf = teacher_forced_metrics(ckpt, [s[2] for s in seeds], device=device)
    summary["teacher_forced"] = tf
    print(json.dumps(tf, indent=2), flush=True)

    summary_path = out_dir / "fullvideo_stability_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"SUMMARY={summary_path}", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
