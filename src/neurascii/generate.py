"""Generate autoregressive ASCII rollouts from a checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import frames_to_patches, patches_to_frames
from .format import AsciiVideo, load_avm, save_avm
from .model import build_model_from_config
from .rollout_diag import summarize_rollout_stability


def sample_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """Sample class indices from logits. temperature<=0 => greedy (argmax)."""
    if temperature is None or temperature <= 0:
        return logits.argmax(-1)
    scaled = logits / max(float(temperature), 1e-6)
    # top-k filter
    if top_k is not None and top_k > 0:
        k = min(int(top_k), scaled.shape[-1])
        vals, idx = torch.topk(scaled, k, dim=-1)
        masked = torch.full_like(scaled, float("-inf"))
        masked.scatter_(-1, idx, vals)
        scaled = masked
    # nucleus / top-p filter
    if top_p is not None and 0 < float(top_p) < 1.0:
        sorted_logits, sorted_idx = torch.sort(scaled, dim=-1, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        # remove tokens with cumulative prob above top_p (keep first above)
        mask = cum > float(top_p)
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        # scatter back
        scaled = torch.full_like(scaled, float("-inf"))
        scaled.scatter_(-1, sorted_idx, sorted_logits)
    probs = F.softmax(scaled, dim=-1)
    flat = probs.reshape(-1, probs.shape[-1])
    choice = torch.multinomial(flat, 1).view(*probs.shape[:-1])
    return choice


def frame_token_change_stats(
    glyphs: np.ndarray,
    fg: np.ndarray,
    bg: np.ndarray,
) -> dict[str, Any]:
    """Per-step and aggregate frame-to-frame token change fractions."""
    t = int(glyphs.shape[0])
    if t < 2:
        return {
            "n_frames": t,
            "mean_glyph_change": 0.0,
            "mean_fg_change": 0.0,
            "mean_bg_change": 0.0,
            "mean_any_change": 0.0,
            "per_step": [],
            "per_step_any_change": [],
        }
    per_step: list[dict[str, float]] = []
    for i in range(1, t):
        g_ch = float((glyphs[i] != glyphs[i - 1]).mean())
        f_ch = float((fg[i] != fg[i - 1]).mean())
        b_ch = float((bg[i] != bg[i - 1]).mean())
        any_ch = float(
            ((glyphs[i] != glyphs[i - 1]) | (fg[i] != fg[i - 1]) | (bg[i] != bg[i - 1])).mean()
        )
        per_step.append(
            {
                "step": i,
                "glyph_change": g_ch,
                "fg_change": f_ch,
                "bg_change": b_ch,
                "any_change": any_ch,
            }
        )
    return {
        "n_frames": t,
        "mean_glyph_change": float(np.mean([s["glyph_change"] for s in per_step])),
        "mean_fg_change": float(np.mean([s["fg_change"] for s in per_step])),
        "mean_bg_change": float(np.mean([s["bg_change"] for s in per_step])),
        "mean_any_change": float(np.mean([s["any_change"] for s in per_step])),
        "per_step": per_step,
        "per_step_any_change": [s["any_change"] for s in per_step],
    }


def sampler_settings(
    *,
    temperature: float,
    top_k: int | None,
    top_p: float | None = None,
) -> dict[str, Any]:
    mode = "greedy" if temperature is None or temperature <= 0 else "temperature"
    return {
        "mode": mode,
        "temperature": float(temperature) if temperature is not None else 0.0,
        "top_k": int(top_k) if top_k else None,
        "top_p": float(top_p) if top_p is not None else None,
    }


def write_sampler_sidecar(avm_path: Path, payload: dict[str, Any]) -> Path:
    side = Path(str(avm_path).removesuffix(".avm.npz") + ".sampler.json")
    if side == avm_path:
        side = avm_path.with_suffix(avm_path.suffix + ".sampler.json")
    side.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return side


@torch.no_grad()
def generate_rollout(
    ckpt_path: Path,
    seed_avm: Path,
    out_path: Path,
    *,
    steps: int,
    device: torch.device,
    start: int = 0,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    max_temperature: float = 1.0,
) -> tuple[Path, dict[str, Any]]:
    """Free-running rollout; returns (avm_path, diagnostics dict)."""
    if temperature > max_temperature:
        raise SystemExit(
            f"temperature={temperature} exceeds cap of {max_temperature}"
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    ph, pw = int(cfg["data"]["patch_h"]), int(cfg["data"]["patch_w"])
    ctx_n = int(cfg["data"]["context_frames"])

    video = load_avm(seed_avm)
    if start + ctx_n >= video.T:
        raise SystemExit(f"seed clip too short for context window: {seed_avm}")

    g, f, b = frames_to_patches(video.glyphs, video.fg, video.bg, ph, pw)
    n_patches = g.shape[1]
    model = build_model_from_config(cfg, n_patches=n_patches).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        # non-fatal for head_change warm-starts
        pass
    model.eval()

    t0 = start
    ctx_g = torch.from_numpy(g[t0 : t0 + ctx_n].astype(np.int64)).unsqueeze(0).to(device)
    ctx_f = torch.from_numpy(f[t0 : t0 + ctx_n].astype(np.int64)).unsqueeze(0).to(device)
    ctx_b = torch.from_numpy(b[t0 : t0 + ctx_n].astype(np.int64)).unsqueeze(0).to(device)

    settings = sampler_settings(temperature=temperature, top_k=top_k, top_p=top_p)
    out_g, out_f, out_b = [], [], []
    for _ in range(steps):
        if getattr(model, "use_delta", False) and (temperature is None or temperature <= 0):
            pg, pf, pb = model.predict_delta(ctx_g, ctx_f, ctx_b)
        else:
            fwd = model(ctx_g, ctx_f, ctx_b)
            lg, lf, lb = fwd[0], fwd[1], fwd[2]
            bsz, n_p, k_cells, _ = lg.shape
            pg = sample_logits(
                lg.reshape(bsz, n_p * k_cells, -1),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            ).view(bsz, n_p, k_cells)
            pf = sample_logits(
                lf.reshape(bsz, n_p * k_cells, -1),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            ).view(bsz, n_p, k_cells)
            pb = sample_logits(
                lb.reshape(bsz, n_p * k_cells, -1),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            ).view(bsz, n_p, k_cells)
            if getattr(model, "use_delta", False) and len(fwd) > 3:
                # Still apply copy-unchanged for stochastic value heads
                change = fwd[3].argmax(-1).bool()
                prev_g, prev_f, prev_b = ctx_g[:, -1], ctx_f[:, -1], ctx_b[:, -1]
                pg = torch.where(change, pg, prev_g)
                pf = torch.where(change, pf, prev_f)
                pb = torch.where(change, pb, prev_b)
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
    gen_g = gf.astype(np.uint16)
    gen_f = ff.astype(np.uint16)
    gen_b = bf.astype(np.uint16)
    change = frame_token_change_stats(gen_g, gen_f, gen_b)
    stab = summarize_rollout_stability(
        gen_g,
        gen_f,
        gen_b,
        patch_h=ph,
        patch_w=pw,
        token_change_summary={
            "mean_any_change": change["mean_any_change"],
            "per_step_any_change": change["per_step_any_change"],
        },
    )

    extra = {
        "generated_steps": steps,
        "checkpoint": str(ckpt_path),
        "seed_avm": str(seed_avm),
        "start": start,
        "kind": "free_rollout",
        "sampler": settings,
        "token_change": {
            "mean_glyph_change": change["mean_glyph_change"],
            "mean_fg_change": change["mean_fg_change"],
            "mean_bg_change": change["mean_bg_change"],
            "mean_any_change": change["mean_any_change"],
            "n_gen_frames": change["n_frames"],
        },
        "blob": stab["blob"],
        "stability": {
            "bug_blob_signature": stab["bug_blob_signature"],
            "attractor": stab["attractor"],
            "persistence": {
                "frac_frozen": stab["persistence"]["frac_frozen"],
                "frac_keep_changing": stab["persistence"]["frac_keep_changing"],
            },
        },
    }
    out = AsciiVideo(
        glyphs=np.concatenate([seed_g, gen_g], axis=0),
        fg=np.concatenate([seed_f, gen_f], axis=0),
        bg=np.concatenate([seed_b, gen_b], axis=0),
        renderer=video.renderer,
        source_path=str(seed_avm),
        extra=extra,
    )
    path = save_avm(out_path, out)
    sidecar = {
        "avm": str(path),
        "sampler": settings,
        "generated_steps": steps,
        "checkpoint": str(ckpt_path),
        "seed_avm": str(seed_avm),
        "start": start,
        "token_change": change,
        "blob": {
            "growth_slope": stab["blob"]["growth_slope"],
            "largest_region_final": stab["blob"]["largest_region_final"],
            "final_frac": stab["blob"]["final_frac"],
        },
        "stability": extra["stability"],
    }
    side_path = write_sampler_sidecar(path, sidecar)
    diag = {
        "avm": str(path),
        "sidecar": str(side_path),
        "sampler": settings,
        "token_change_summary": {
            k: change[k]
            for k in (
                "mean_glyph_change",
                "mean_fg_change",
                "mean_bg_change",
                "mean_any_change",
                "n_frames",
            )
        },
        "blob": stab["blob"],
        "stability": stab,
    }
    return path, diag


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--seed-avm", type=Path, required=True, help="Source .avm.npz for context")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=48)
    p.add_argument("--start", type=int, default=0, help="Start frame index for context window")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 => greedy; decode sweeps may use up to 1.0",
    )
    p.add_argument("--top-k", type=int, default=0, help="0 disables top-k; only used if temperature>0")
    p.add_argument("--top-p", type=float, default=0.0, help="0 disables; nucleus sampling if (0,1)")
    args = p.parse_args(argv)

    device = torch.device(args.device)
    top_k = args.top_k if args.top_k and args.top_k > 0 else None
    top_p = args.top_p if args.top_p and 0 < args.top_p < 1 else None
    path, diag = generate_rollout(
        args.checkpoint,
        args.seed_avm,
        args.out,
        steps=args.steps,
        device=device,
        start=args.start,
        temperature=args.temperature,
        top_k=top_k,
        top_p=top_p,
    )
    tc = diag["token_change_summary"]
    print(
        f"wrote {path}  frames={load_avm(path).T}  "
        f"sampler={diag['sampler']}  "
        f"mean_any_change={tc['mean_any_change']:.4f}  "
        f"blob_slope={diag['blob']['growth_slope']:.2f}  "
        f"sidecar={diag['sidecar']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
