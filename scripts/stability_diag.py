#!/usr/bin/env python3
"""General stability diagnostics for any processed_dir + checkpoint.

Horizons 16/48/96 by default. Reports blob curves, persistence, time-to-attractor,
changed/unchanged TF loss, and a prev-frame-copy free-rollout baseline.
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
from neurascii.generate import frame_token_change_stats, generate_rollout
from neurascii.model import build_model_from_config
from neurascii.rollout_diag import (
    changed_vs_unchanged_ce,
    prev_frame_copy_rollout,
    summarize_rollout_stability,
)

HORIZONS = (16, 48, 96)
WANT_SEEDS = 4


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
        # fewer classes than want: cycle clips
        chosen = classes
    out: list[tuple[str, str, Path]] = []
    clips_flat = [p for ps in by_class.values() for p in ps]
    for i, c in enumerate(chosen[:want]):
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
    # pad with more clips if single class
    while len(out) < want and len(clips_flat) > len(out):
        p = clips_flat[len(out)]
        cls, name = parse_seed(p)
        out.append((cls, name, p))
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

    v0 = load_avm(seed_paths[0])
    g0, _, _ = frames_to_patches(v0.glyphs, v0.fg, v0.bg, ph, pw)
    n_patches = g0.shape[1]
    model = build_model_from_config(cfg, n_patches=n_patches).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    ds = NextFrameDataset(seed_paths, context_frames=ctx, patch_h=ph, patch_w=pw)
    if len(ds) == 0:
        return {"error": "no teacher-forced windows (clips too short?)", "n_windows": 0}

    losses, acc_g, acc_f, acc_b = [], [], [], []
    loss_ch, loss_un, ch_frac = [], [], []
    n = min(len(ds), max_windows)
    for i in range(n):
        batch = ds[i]
        batch_t = {k: v.unsqueeze(0).to(device) for k, v in batch.items()}
        tg = batch_t["tgt_g"]
        if tg.ndim == 4:
            tg = tg[:, 0]
            tf = batch_t["tgt_f"][:, 0]
            tb = batch_t["tgt_b"][:, 0]
        else:
            tf = batch_t["tgt_f"]
            tb = batch_t["tgt_b"]
        out = model.loss(
            batch_t["ctx_g"],
            batch_t["ctx_f"],
            batch_t["ctx_b"],
            tg,
            tf,
            tb,
            prev_g=batch_t.get("prev_g"),
            prev_f=batch_t.get("prev_f"),
            prev_b=batch_t.get("prev_b"),
        )
        losses.append(float(out["loss"]))
        acc_g.append(float(out["acc_g"]))
        acc_f.append(float(out["acc_f"]))
        acc_b.append(float(out["acc_b"]))
        cu = changed_vs_unchanged_ce(
            model,
            batch_t["ctx_g"],
            batch_t["ctx_f"],
            batch_t["ctx_b"],
            tg,
            tf,
            tb,
            prev_g=batch_t.get("prev_g"),
            prev_f=batch_t.get("prev_f"),
            prev_b=batch_t.get("prev_b"),
        )
        if not np.isnan(cu["loss_changed"]):
            loss_ch.append(cu["loss_changed"])
        if not np.isnan(cu["loss_unchanged"]):
            loss_un.append(cu["loss_unchanged"])
        ch_frac.append(cu["change_frac"])

    return {
        "n_windows": n,
        "loss_mean": float(np.mean(losses)),
        "acc_g_mean": float(np.mean(acc_g)),
        "acc_f_mean": float(np.mean(acc_f)),
        "acc_b_mean": float(np.mean(acc_b)),
        "loss_changed_mean": float(np.nanmean(loss_ch)) if loss_ch else None,
        "loss_unchanged_mean": float(np.nanmean(loss_un)) if loss_un else None,
        "change_frac_mean": float(np.mean(ch_frac)) if ch_frac else None,
        "sampler_note": "teacher_forced (argmax heads; not free rollout)",
    }


def baseline_metrics(
    seed_path: Path,
    *,
    steps: int,
    context_frames: int,
    patch_h: int,
    patch_w: int,
) -> dict:
    video = load_avm(seed_path)
    ctx_n = context_frames
    full_g, full_f, full_b = prev_frame_copy_rollout(
        video.glyphs[:ctx_n],
        video.fg[:ctx_n],
        video.bg[:ctx_n],
        steps=steps,
        context_frames=ctx_n,
    )
    gen_g = full_g[ctx_n:]
    gen_f = full_f[ctx_n:]
    gen_b = full_b[ctx_n:]
    tc = frame_token_change_stats(gen_g, gen_f, gen_b)
    stab = summarize_rollout_stability(
        gen_g,
        gen_f,
        gen_b,
        patch_h=patch_h,
        patch_w=patch_w,
        token_change_summary={
            "mean_any_change": tc["mean_any_change"],
            "per_step_any_change": tc["per_step_any_change"],
        },
    )
    return {
        "kind": "prev_frame_copy",
        "horizon": steps,
        "token_change": {
            "mean_any_change": tc["mean_any_change"],
            "mean_glyph_change": tc["mean_glyph_change"],
        },
        "blob": {
            "growth_slope": stab["blob"]["growth_slope"],
            "largest_region_final": stab["blob"]["largest_region_final"],
            "final_frac": stab["blob"]["final_frac"],
        },
        "attractor_onset": stab["attractor"]["onset_frame"],
        "bug_blob_signature": stab["bug_blob_signature"],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--tag", type=str, default="diag")
    p.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    p.add_argument("--n-seeds", type=int, default=WANT_SEEDS)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=0.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    ckpt = args.checkpoint
    if not ckpt.is_file():
        raise SystemExit(f"missing checkpoint {ckpt}")
    processed = args.processed_dir
    if not processed.is_dir():
        raise SystemExit(f"missing {processed}")

    out_dir = args.out_dir or (_ROOT / "samples" / "rollouts" / args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    top_k = args.top_k if args.top_k and args.top_k > 0 else None
    top_p = args.top_p if args.top_p and 0 < args.top_p < 1 else None
    seeds = pick_diverse_seeds(processed, want=args.n_seeds)

    ckpt_obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt_obj["config"]
    ph, pw = int(cfg["data"]["patch_h"]), int(cfg["data"]["patch_w"])
    ctx_n = int(cfg["data"]["context_frames"])

    summary: dict = {
        "tag": args.tag,
        "checkpoint": str(ckpt),
        "processed_dir": str(processed),
        "sampler": {
            "mode": "greedy" if args.temperature <= 0 else "temperature",
            "temperature": float(args.temperature),
            "top_k": top_k,
            "top_p": top_p,
        },
        "horizons": list(args.horizons),
        "seeds": [],
        "rollouts": [],
        "baselines": [],
        "teacher_forced": None,
        "aggregate": {},
    }

    print(f"CKPT={ckpt}", flush=True)
    print(f"SEEDS={[f'{c}/{n}' for c, n, _ in seeds]}", flush=True)

    h96_rows = []
    for i, (cls, name, path) in enumerate(seeds, 1):
        summary["seeds"].append({"class": cls, "name": name, "path": str(path)})
        for h in args.horizons:
            out = out_dir / f"{args.tag}_{i:02d}_{cls}_h{h:03d}.avm.npz"
            print(f"--- free rollout {cls}/{name} horizon={h}", flush=True)
            avm_path, diag = generate_rollout(
                ckpt,
                path,
                out,
                steps=h,
                device=device,
                temperature=args.temperature,
                top_k=top_k,
                top_p=top_p,
            )
            blob = diag["blob"]
            onset = diag["stability"]["attractor"]["onset_frame"]
            row = {
                "seed_class": cls,
                "seed_name": name,
                "horizon": h,
                "avm": str(avm_path),
                "sidecar": diag["sidecar"],
                "sampler": diag["sampler"],
                "token_change": diag["token_change_summary"],
                "blob_growth_slope": blob["growth_slope"],
                "largest_region_final": blob["largest_region_final"],
                "blob_final_frac": blob["final_frac"],
                "attractor_onset": onset,
                "bug_blob_signature": diag["stability"]["bug_blob_signature"],
                "frac_frozen": diag["stability"]["persistence"]["frac_frozen"],
            }
            summary["rollouts"].append(row)
            if h == 96 or (96 not in args.horizons and h == max(args.horizons)):
                h96_rows.append(row)
            print(
                f"    mean_any={row['token_change']['mean_any_change']:.4f} "
                f"blob_slope={row['blob_growth_slope']:.2f} "
                f"final_frac={row['blob_final_frac']:.3f}",
                flush=True,
            )

            base = baseline_metrics(
                path, steps=h, context_frames=ctx_n, patch_h=ph, patch_w=pw
            )
            base["seed_class"] = cls
            base["seed_name"] = name
            summary["baselines"].append(base)

    print("--- teacher-forced metrics ---", flush=True)
    tf = teacher_forced_metrics(ckpt, [s[2] for s in seeds], device=device)
    summary["teacher_forced"] = tf
    print(json.dumps(tf, indent=2), flush=True)

    if h96_rows:
        summary["aggregate"] = {
            "horizon": h96_rows[0]["horizon"],
            "mean_any_change": float(
                np.mean([r["token_change"]["mean_any_change"] for r in h96_rows])
            ),
            "mean_blob_growth_slope": float(
                np.mean([r["blob_growth_slope"] for r in h96_rows])
            ),
            "mean_blob_final_frac": float(
                np.mean([r["blob_final_frac"] for r in h96_rows])
            ),
            "frac_bug_blob_signature": float(
                np.mean([1.0 if r["bug_blob_signature"] else 0.0 for r in h96_rows])
            ),
            "best_val_note": "see run metrics.jsonl / ckpt best_val",
        }

    summary_path = out_dir / f"{args.tag}_stability_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"SUMMARY={summary_path}", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
