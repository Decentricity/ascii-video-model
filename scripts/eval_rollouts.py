#!/usr/bin/env python3
"""Generate fixed rollouts from ckpt_best (or ckpt_last) into samples/rollouts/.

Local-only artefacts; samples/rollouts/ is gitignored.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Allow running as scripts/eval_rollouts.py without install
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from neurascii.dataset import discover_avm, frames_to_patches, patches_to_frames
from neurascii.format import AsciiVideo, load_avm, save_avm
from neurascii.model import build_model_from_config

DEFAULT_RUNS = {
    "poc_eye": Path("runs/poc_eye"),
    "poc_lipstick": Path("runs/poc_lipstick"),
    "smoke_apply": Path("runs/smoke_apply"),
}


def _resolve_run_dir(name: str) -> Path:
    """Prefer early-stop best dir for poc_eye when present."""
    if name == "poc_eye":
        best = Path("runs/poc_eye_best")
        if best.is_dir() and (
            (best / "ckpt_best.pt").is_file() or (best / "ckpt_last.pt").is_file()
        ):
            return best
    return DEFAULT_RUNS.get(name, Path("runs") / name)


def _pick_ckpt(run_dir: Path) -> Path:
    best = run_dir / "ckpt_best.pt"
    last = run_dir / "ckpt_last.pt"
    if best.is_file():
        return best
    if last.is_file():
        return last
    raise FileNotFoundError(f"no ckpt_best.pt or ckpt_last.pt in {run_dir}")


def _pick_seed(run_dir: Path, processed_hint: Path | None) -> Path:
    split = run_dir / "split.json"
    if processed_hint and processed_hint.is_dir():
        paths = discover_avm(processed_hint)
        if paths:
            return paths[0]
    # fall back: any avm under data/processed matching run name
    for cand in (
        Path("data/processed") / run_dir.name,
        Path("data/processed/smoke_apply"),
        Path("data/processed"),
    ):
        if cand.is_dir():
            paths = discover_avm(cand)
            if paths:
                return paths[0]
    raise FileNotFoundError(f"no seed .avm.npz found for {run_dir}")


@torch.no_grad()
def generate_rollout(
    ckpt_path: Path,
    seed_avm: Path,
    out_path: Path,
    *,
    steps: int,
    device: torch.device,
    start: int = 0,
) -> Path:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    ph, pw = int(cfg["data"]["patch_h"]), int(cfg["data"]["patch_w"])
    ctx_n = int(cfg["data"]["context_frames"])

    video = load_avm(seed_avm)
    if start + ctx_n >= video.T:
        raise SystemExit(f"seed clip too short for context: {seed_avm}")

    g, f, b = frames_to_patches(video.glyphs, video.fg, video.bg, ph, pw)
    n_patches = g.shape[1]
    model = build_model_from_config(cfg, n_patches=n_patches).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    t0 = start
    ctx_g = torch.from_numpy(g[t0 : t0 + ctx_n].astype(np.int64)).unsqueeze(0).to(device)
    ctx_f = torch.from_numpy(f[t0 : t0 + ctx_n].astype(np.int64)).unsqueeze(0).to(device)
    ctx_b = torch.from_numpy(b[t0 : t0 + ctx_n].astype(np.int64)).unsqueeze(0).to(device)

    out_g, out_f, out_b = [], [], []
    for _ in range(steps):
        lg, lf, lb = model(ctx_g, ctx_f, ctx_b)
        pg, pf, pb = lg.argmax(-1), lf.argmax(-1), lb.argmax(-1)
        out_g.append(pg[0].cpu().numpy())
        out_f.append(pf[0].cpu().numpy())
        out_b.append(pb[0].cpu().numpy())
        ctx_g = torch.cat([ctx_g[:, 1:], pg.unsqueeze(1)], dim=1)
        ctx_f = torch.cat([ctx_f[:, 1:], pf.unsqueeze(1)], dim=1)
        ctx_b = torch.cat([ctx_b[:, 1:], pb.unsqueeze(1)], dim=1)

    gf, ff, bf = patches_to_frames(
        np.stack(out_g), np.stack(out_f), np.stack(out_b), video.H, video.W, ph, pw
    )
    seed_g = video.glyphs[t0 : t0 + ctx_n]
    seed_f = video.fg[t0 : t0 + ctx_n]
    seed_b = video.bg[t0 : t0 + ctx_n]
    out = AsciiVideo(
        glyphs=np.concatenate([seed_g, gf.astype(np.uint16)], axis=0),
        fg=np.concatenate([seed_f, ff.astype(np.uint16)], axis=0),
        bg=np.concatenate([seed_b, bf.astype(np.uint16)], axis=0),
        renderer=video.renderer,
        source_path=str(seed_avm),
        extra={
            "generated_steps": steps,
            "checkpoint": str(ckpt_path),
            "kind": "eval_rollout",
        },
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return save_avm(out_path, out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs",
        nargs="*",
        default=list(DEFAULT_RUNS.keys()),
        help="Run names: poc_eye poc_lipstick smoke_apply",
    )
    p.add_argument("--out-dir", type=Path, default=Path("samples/rollouts"))
    p.add_argument("--steps", type=int, default=48)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    device = torch.device(args.device)
    for name in args.runs:
        run_dir = _resolve_run_dir(name)
        if not run_dir.is_dir():
            print(f"skip {name}: missing {run_dir}")
            continue
        try:
            ckpt = _pick_ckpt(run_dir)
        except FileNotFoundError as e:
            print(f"skip {name}: {e}")
            continue
        processed = Path(f"data/processed/{name}")
        try:
            seed = _pick_seed(run_dir, processed if processed.is_dir() else None)
        except FileNotFoundError as e:
            print(f"skip {name}: {e}")
            continue
        out = args.out_dir / f"{name}_rollout.avm.npz"
        path = generate_rollout(ckpt, seed, out, steps=args.steps, device=device)
        print(f"{name}: {ckpt.name} ({run_dir}) -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
