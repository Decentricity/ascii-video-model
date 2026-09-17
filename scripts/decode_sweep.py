#!/usr/bin/env python3
"""Decode sweep: greedy / temp / top-k / top-p on a fixed seed.

Writes samples/rollouts/decode_sweep_*.json summarizing whether sampling
only delays the blob attractor vs changing the failure mode.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from neurascii.format import load_avm
from neurascii.generate import generate_rollout

SWEEPS = [
    {"name": "greedy", "temperature": 0.0, "top_k": None, "top_p": None},
    {"name": "temp_0.3", "temperature": 0.3, "top_k": None, "top_p": None},
    {"name": "temp_0.7", "temperature": 0.7, "top_k": None, "top_p": None},
    {"name": "topk_8_t0.7", "temperature": 0.7, "top_k": 8, "top_p": None},
    {"name": "topk_32_t0.7", "temperature": 0.7, "top_k": 32, "top_p": None},
    {"name": "topp_0.9_t0.7", "temperature": 0.7, "top_k": None, "top_p": 0.9},
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--seed-avm", type=Path, default=None)
    p.add_argument("--processed-dir", type=Path, default=_ROOT / "data/processed/poc_eye")
    p.add_argument("--steps", type=int, default=96)
    p.add_argument("--out-dir", type=Path, default=_ROOT / "samples/rollouts/decode_sweep")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    seed = args.seed_avm
    if seed is None:
        cands = sorted(args.processed_dir.glob("*.avm.npz"))
        if not cands:
            raise SystemExit(f"no seeds in {args.processed_dir}")
        # prefer longest
        best, best_t = cands[0], -1
        for c in cands:
            try:
                t = load_avm(c).T
            except Exception:
                continue
            if t > best_t:
                best, best_t = c, t
        seed = best

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    rows = []
    greedy_slope = None
    greedy_change = None
    for s in SWEEPS:
        name = s["name"]
        out = out_dir / f"decode_sweep_{name}_h{args.steps:03d}.avm.npz"
        print(f"--- {name} temp={s['temperature']} top_k={s['top_k']} top_p={s['top_p']}", flush=True)
        _, diag = generate_rollout(
            args.checkpoint,
            seed,
            out,
            steps=args.steps,
            device=device,
            temperature=float(s["temperature"]),
            top_k=s["top_k"],
            top_p=s["top_p"],
            max_temperature=1.0,
        )
        slope = float(diag["blob"]["growth_slope"])
        mean_any = float(diag["token_change_summary"]["mean_any_change"])
        onset = diag["stability"]["attractor"]["onset_frame"]
        row = {
            "name": name,
            "sampler": diag["sampler"],
            "mean_any_change": mean_any,
            "blob_growth_slope": slope,
            "blob_final_frac": float(diag["blob"]["final_frac"]),
            "attractor_onset": onset,
            "bug_blob_signature": bool(diag["stability"]["bug_blob_signature"]),
            "avm": diag["avm"],
        }
        rows.append(row)
        if name == "greedy":
            greedy_slope = slope
            greedy_change = mean_any
        print(
            f"    mean_any={mean_any:.4f} slope={slope:.2f} onset={onset}",
            flush=True,
        )

    # Heuristic: sampling only delays blob if all still have positive growth
    # and mean_any stays low, but onset is later / slope slightly lower.
    all_grow = all(r["blob_growth_slope"] > 0 for r in rows)
    all_low_motion = all(r["mean_any_change"] < 0.05 for r in rows)
    onsets = [r["attractor_onset"] for r in rows if r["attractor_onset"] is not None]
    delayed = False
    if greedy_slope is not None and all_grow:
        non_greedy = [r for r in rows if r["name"] != "greedy"]
        if non_greedy:
            later = [
                r
                for r in non_greedy
                if (r["attractor_onset"] or 0) > (rows[0]["attractor_onset"] or 0)
            ]
            delayed = len(later) >= max(1, len(non_greedy) // 2) or all_low_motion

    verdict = {
        "only_delays_blob": bool(all_grow and (delayed or all_low_motion)),
        "all_positive_growth": all_grow,
        "all_low_motion": all_low_motion,
        "note": (
            "Sampling variants still show blob growth; stochasticity may delay onset "
            "but does not eliminate the attractor."
            if all_grow
            else "At least one decode setting reduced/eliminated positive blob growth."
        ),
    }

    summary = {
        "checkpoint": str(args.checkpoint),
        "seed_avm": str(seed),
        "steps": args.steps,
        "sweeps": rows,
        "verdict": verdict,
        "greedy_ref": {"blob_growth_slope": greedy_slope, "mean_any_change": greedy_change},
    }
    out_json = out_dir / "decode_sweep_summary.json"
    # also write top-level alias requested by plan
    alias = _ROOT / "samples" / "rollouts" / "decode_sweep_summary.json"
    alias.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(summary, indent=2) + "\n"
    out_json.write_text(text, encoding="utf-8")
    alias.write_text(text, encoding="utf-8")
    print(f"SUMMARY={out_json}", flush=True)
    print(f"VERDICT only_delays_blob={verdict['only_delays_blob']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
