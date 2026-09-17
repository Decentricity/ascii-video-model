"""Generate autoregressive ASCII rollouts from a checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from .dataset import NextFrameDataset, frames_to_patches
from .format import AsciiVideo, RendererConfig, load_avm, save_avm
from .model import build_model_from_config
from .dataset import patches_to_frames


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--seed-avm", type=Path, required=True, help="Source .avm.npz for context")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=48)
    p.add_argument("--start", type=int, default=0, help="Start frame index for context window")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    ph, pw = int(cfg["data"]["patch_h"]), int(cfg["data"]["patch_w"])
    ctx_n = int(cfg["data"]["context_frames"])

    video = load_avm(args.seed_avm)
    if args.start + ctx_n >= video.T:
        raise SystemExit("seed clip too short for context window")

    g, f, b = frames_to_patches(video.glyphs, video.fg, video.bg, ph, pw)
    n_patches = g.shape[1]
    model = build_model_from_config(cfg, n_patches=n_patches).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    t0 = args.start
    ctx_g = torch.from_numpy(g[t0 : t0 + ctx_n].astype(np.int64)).unsqueeze(0).to(device)
    ctx_f = torch.from_numpy(f[t0 : t0 + ctx_n].astype(np.int64)).unsqueeze(0).to(device)
    ctx_b = torch.from_numpy(b[t0 : t0 + ctx_n].astype(np.int64)).unsqueeze(0).to(device)

    out_g, out_f, out_b = [], [], []
    with torch.no_grad():
        for _ in range(args.steps):
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
    # prepend seed context frames for playable continuity
    seed_g = video.glyphs[t0 : t0 + ctx_n]
    seed_f = video.fg[t0 : t0 + ctx_n]
    seed_b = video.bg[t0 : t0 + ctx_n]
    out = AsciiVideo(
        glyphs=np.concatenate([seed_g, gf.astype(np.uint16)], axis=0),
        fg=np.concatenate([seed_f, ff.astype(np.uint16)], axis=0),
        bg=np.concatenate([seed_b, bf.astype(np.uint16)], axis=0),
        renderer=video.renderer,
        source_path=str(args.seed_avm),
        extra={"generated_steps": args.steps, "checkpoint": str(args.checkpoint)},
    )
    path = save_avm(args.out, out)
    print(f"wrote {path}  frames={out.T}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
