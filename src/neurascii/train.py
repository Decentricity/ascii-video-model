"""Training loop for next-frame ASCII model."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .dataset import NextFrameDataset, discover_avm, patches_to_frames
from .format import AsciiVideo, RendererConfig, save_avm
from .generate import frame_token_change_stats
from .model import build_model_from_config
from .rollout_diag import blob_growth_curve, summarize_rollout_stability


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_by_video(paths: list[Path], seed: int = 42) -> tuple[list[Path], list[Path]]:
    paths = list(paths)
    rng = random.Random(seed)
    rng.shuffle(paths)
    if len(paths) == 1:
        return paths, paths
    n_val = max(1, len(paths) // 8)
    return paths[n_val:], paths[:n_val]


def _scheduled_sampling_p(step: int, p_max: float, warmup: int) -> float:
    if p_max <= 0:
        return 0.0
    if warmup <= 0:
        return float(p_max)
    return float(p_max) * min(1.0, step / float(warmup))


def _rollout_steps_curriculum(
    step: int,
    max_steps: int,
    steps_min: int,
    steps_max: int,
) -> int:
    steps_min = max(1, int(steps_min))
    steps_max = max(steps_min, int(steps_max))
    if steps_max == steps_min or max_steps <= 1:
        return steps_min
    frac = min(1.0, (step - 1) / float(max_steps - 1))
    return int(round(steps_min + frac * (steps_max - steps_min)))


def _slice_target(batch: dict, t: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get target frame t from batch (supports R=1 flat or R>1 sequenced)."""
    tg, tf, tb = batch["tgt_g"], batch["tgt_f"], batch["tgt_b"]
    if tg.ndim == 4:  # B, R, N, K
        return tg[:, t], tf[:, t], tb[:, t]
    if t != 0:
        raise IndexError("single-target batch but t>0")
    return tg, tf, tb


def _n_future(batch: dict) -> int:
    tg = batch["tgt_g"]
    return int(tg.shape[1]) if tg.ndim == 4 else 1


@torch.no_grad()
def _argmax_pred(model, ctx_g, ctx_f, ctx_b):
    if getattr(model, "use_delta", False):
        return model.predict_delta(ctx_g, ctx_f, ctx_b)
    out = model(ctx_g, ctx_f, ctx_b)
    return out[0].argmax(-1), out[1].argmax(-1), out[2].argmax(-1)


def compute_train_loss(
    model,
    batch: dict,
    *,
    step: int,
    max_steps: int,
    change_weight: float,
    use_delta: bool,
    ss_p_max: float,
    ss_warmup: int,
    rollout_steps_min: int,
    rollout_steps_max: int,
) -> dict[str, torch.Tensor]:
    """Teacher-forced / scheduled-sampling / multi-step rollout loss."""
    n_fut = _n_future(batch)
    r_steps = _rollout_steps_curriculum(step, max_steps, rollout_steps_min, rollout_steps_max)
    r_steps = min(r_steps, n_fut)
    p = _scheduled_sampling_p(step, ss_p_max, ss_warmup)

    ctx_g = batch["ctx_g"]
    ctx_f = batch["ctx_f"]
    ctx_b = batch["ctx_b"]
    prev_g = batch.get("prev_g", ctx_g[:, -1])
    prev_f = batch.get("prev_f", ctx_f[:, -1])
    prev_b = batch.get("prev_b", ctx_b[:, -1])

    total = None
    last_out: dict[str, torch.Tensor] | None = None
    B = ctx_g.shape[0]

    for t in range(r_steps):
        tgt_g, tgt_f, tgt_b = _slice_target(batch, t)
        # Optional mix: replace last context frame with stop-grad pred before loss
        if p > 0 and t == 0:
            # First do a detached pred on true context, then mix into last frame
            with torch.no_grad():
                pg, pf, pb = _argmax_pred(model, ctx_g, ctx_f, ctx_b)
            mix = (torch.rand(B, device=ctx_g.device) < p).view(B, 1, 1)
            ctx_g = ctx_g.clone()
            ctx_f = ctx_f.clone()
            ctx_b = ctx_b.clone()
            ctx_g[:, -1] = torch.where(mix, pg, ctx_g[:, -1])
            ctx_f[:, -1] = torch.where(mix, pf, ctx_f[:, -1])
            ctx_b[:, -1] = torch.where(mix, pb, ctx_b[:, -1])
            prev_g = ctx_g[:, -1]
            prev_f = ctx_f[:, -1]
            prev_b = ctx_b[:, -1]

        out = model.loss(
            ctx_g,
            ctx_f,
            ctx_b,
            tgt_g,
            tgt_f,
            tgt_b,
            change_weight=change_weight,
            use_delta=use_delta,
            prev_g=prev_g,
            prev_f=prev_f,
            prev_b=prev_b,
        )
        last_out = out
        total = out["loss"] if total is None else total + out["loss"]

        if t + 1 < r_steps:
            # Push into context: mix pred vs GT by scheduled sampling p
            with torch.no_grad():
                pg, pf, pb = _argmax_pred(model, ctx_g, ctx_f, ctx_b)
            mix = (torch.rand(B, device=ctx_g.device) < p).view(B, 1, 1) if p > 0 else None
            if mix is not None:
                nxt_g = torch.where(mix, pg, tgt_g)
                nxt_f = torch.where(mix, pf, tgt_f)
                nxt_b = torch.where(mix, pb, tgt_b)
            else:
                nxt_g, nxt_f, nxt_b = tgt_g, tgt_f, tgt_b
            ctx_g = torch.cat([ctx_g[:, 1:], nxt_g.unsqueeze(1)], dim=1)
            ctx_f = torch.cat([ctx_f[:, 1:], nxt_f.unsqueeze(1)], dim=1)
            ctx_b = torch.cat([ctx_b[:, 1:], nxt_b.unsqueeze(1)], dim=1)
            prev_g, prev_f, prev_b = nxt_g, nxt_f, nxt_b

    assert last_out is not None and total is not None
    result = dict(last_out)
    result["loss"] = total / float(r_steps)
    result["rollout_train_steps"] = torch.tensor(float(r_steps), device=total.device)
    result["ss_p"] = torch.tensor(float(p), device=total.device)
    return result


@torch.no_grad()
def sample_rollout(
    model,
    batch: dict,
    *,
    steps: int,
    H: int,
    W: int,
    patch_h: int,
    patch_w: int,
    device: torch.device,
) -> AsciiVideo:
    model.eval()
    ctx_g = batch["ctx_g"][:1].to(device)
    ctx_f = batch["ctx_f"][:1].to(device)
    ctx_b = batch["ctx_b"][:1].to(device)
    out_g, out_f, out_b = [], [], []
    for _ in range(steps):
        if getattr(model, "use_delta", False):
            pg, pf, pb = model.predict_delta(ctx_g, ctx_f, ctx_b)
        else:
            lg, lf, lb = model(ctx_g, ctx_f, ctx_b)[:3]
            pg, pf, pb = lg.argmax(-1), lf.argmax(-1), lb.argmax(-1)
        out_g.append(pg[0].cpu().numpy())
        out_f.append(pf[0].cpu().numpy())
        out_b.append(pb[0].cpu().numpy())
        ctx_g = torch.cat([ctx_g[:, 1:], pg.unsqueeze(1)], dim=1)
        ctx_f = torch.cat([ctx_f[:, 1:], pf.unsqueeze(1)], dim=1)
        ctx_b = torch.cat([ctx_b[:, 1:], pb.unsqueeze(1)], dim=1)
    g = np.stack(out_g)
    f = np.stack(out_f)
    b = np.stack(out_b)
    gf, ff, bf = patches_to_frames(g, f, b, H, W, patch_h, patch_w)
    gf_u = gf.astype(np.uint16)
    ff_u = ff.astype(np.uint16)
    bf_u = bf.astype(np.uint16)
    tc = frame_token_change_stats(gf_u, ff_u, bf_u)
    stab = summarize_rollout_stability(
        gf_u,
        ff_u,
        bf_u,
        patch_h=patch_h,
        patch_w=patch_w,
        token_change_summary={
            "mean_any_change": tc["mean_any_change"],
            "per_step_any_change": [s["any_change"] for s in tc["per_step"]],
        },
    )
    blob = blob_growth_curve(gf_u, ff_u, bf_u)
    return AsciiVideo(
        glyphs=gf_u,
        fg=ff_u,
        bg=bf_u,
        renderer=RendererConfig(width_chars=W, height_chars=H),
        extra={
            "kind": "rollout",
            "token_change": {
                "mean_any_change": tc["mean_any_change"],
                "mean_glyph_change": tc["mean_glyph_change"],
                "mean_fg_change": tc["mean_fg_change"],
                "mean_bg_change": tc["mean_bg_change"],
            },
            "blob": {
                "growth_slope": blob["growth_slope"],
                "largest_region_final": blob["largest_region_final"],
                "final_frac": blob["final_frac"],
            },
            "stability": {
                "bug_blob_signature": stab["bug_blob_signature"],
                "attractor_onset": stab["attractor"]["onset_frame"],
                "frac_frozen": stab["persistence"]["frac_frozen"],
            },
        },
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Load model weights only (state_dict); do not restore optimizer/step",
    )
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    set_seed(int(cfg["train"].get("seed", 42)))
    device = torch.device(args.device)
    run_dir = Path(cfg["paths"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

    processed = Path(cfg["data"]["processed_dir"])
    all_paths = discover_avm(processed)
    if not all_paths:
        raise SystemExit(f"no .avm.npz in {processed}")
    train_paths, val_paths = split_by_video(all_paths, seed=int(cfg["train"].get("seed", 42)))
    (run_dir / "split.json").write_text(
        json.dumps(
            {
                "train": [p.name for p in train_paths],
                "val": [p.name for p in val_paths],
            },
            indent=2,
        )
    )

    ph, pw = int(cfg["data"]["patch_h"]), int(cfg["data"]["patch_w"])
    ctx = int(cfg["data"]["context_frames"])
    future_frames = int(cfg["data"].get("future_frames", 1))
    # Ensure dataset covers curriculum max
    r_max = int(cfg["train"].get("rollout_train_steps_max", future_frames))
    future_frames = max(future_frames, r_max)

    train_ds = NextFrameDataset(
        train_paths,
        context_frames=ctx,
        patch_h=ph,
        patch_w=pw,
        future_frames=future_frames,
    )
    val_ds = NextFrameDataset(
        val_paths,
        context_frames=ctx,
        patch_h=ph,
        patch_w=pw,
        future_frames=future_frames,
    )
    if len(train_ds) == 0:
        raise SystemExit("train dataset empty — need longer clips or fewer context frames")

    sample0 = train_ds[0]
    n_patches = sample0["tgt_g"].shape[-2] if sample0["tgt_g"].ndim >= 2 else sample0["tgt_g"].shape[0]
    # tgt shape: (N,K) or (R,N,K)
    if sample0["tgt_g"].ndim == 3:
        n_patches = sample0["tgt_g"].shape[1]
    else:
        n_patches = sample0["tgt_g"].shape[0]
    H = train_ds.videos[0]["H"]
    W = train_ds.videos[0]["W"]

    model = build_model_from_config(cfg, n_patches=n_patches).to(device)
    n_params = model.count_parameters()
    print(
        f"parameters: {n_params/1e6:.2f}M  n_patches={n_patches}  "
        f"train_windows={len(train_ds)} future_frames={future_frames} "
        f"use_delta={model.use_delta}",
        flush=True,
    )

    if args.init_checkpoint is not None:
        init_path = args.init_checkpoint
        if not init_path.is_file():
            raise SystemExit(f"init checkpoint not found: {init_path}")
        init_ckpt = torch.load(init_path, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(init_ckpt["model"], strict=False)
        print(
            f"init from {init_path} (weights only; missing={len(missing)} unexpected={len(unexpected)})",
            flush=True,
        )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.01)),
    )
    use_bf16 = cfg["train"].get("precision") == "bf16" and device.type == "cuda"
    loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    max_steps = args.max_steps or int(cfg["train"]["max_steps"])
    log_every = int(cfg["train"].get("log_every", 50))
    sample_every = int(cfg["train"].get("sample_every", 500))
    ckpt_every = int(cfg["train"].get("checkpoint_every", 1000))
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))
    early_stop_patience = int(cfg["train"].get("early_stop_patience", 0))
    min_steps = int(cfg["train"].get("min_steps", 0))
    change_weight = float(cfg["train"].get("change_weight", 1.0))
    use_delta = bool(cfg["train"].get("use_delta", False))
    ss_p_max = float(cfg["train"].get("scheduled_sampling_p_max", 0.0))
    ss_warmup = int(cfg["train"].get("scheduled_sampling_warmup_steps", 500))
    r_min = int(cfg["train"].get("rollout_train_steps", 1))
    r_max = int(cfg["train"].get("rollout_train_steps_max", max(r_min, future_frames)))
    early_stop_on = str(cfg["train"].get("early_stop_on", "val_loss"))

    step = 0
    model.train()
    t0 = time.time()
    data_iter = iter(loader)
    metrics_path = run_dir / "metrics.jsonl"
    best_val = float("inf")
    best_metric = float("inf")  # lower is better for both val_loss and -stability
    bad_checks = 0
    best_step = 0

    def _save_ckpt(tag: str) -> None:
        ckpt = {
            "step": step,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "config": cfg,
            "n_patches": n_patches,
            "H": H,
            "W": W,
            "n_params": n_params,
            "best_val": best_val,
            "best_step": best_step,
        }
        torch.save(ckpt, run_dir / f"{tag}.pt")

    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        batch = {k: v.to(device) for k, v in batch.items()}
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            out = compute_train_loss(
                model,
                batch,
                step=step + 1,
                max_steps=max_steps,
                change_weight=change_weight,
                use_delta=use_delta,
                ss_p_max=ss_p_max,
                ss_warmup=ss_warmup,
                rollout_steps_min=r_min,
                rollout_steps_max=r_max,
            )
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        step += 1

        if step % log_every == 0 or step == 1:
            elapsed = time.time() - t0
            row = {
                "step": step,
                "loss": float(out["loss"].detach()),
                "loss_g": float(out["loss_g"]),
                "loss_f": float(out["loss_f"]),
                "loss_b": float(out["loss_b"]),
                "loss_changed": float(out.get("loss_changed", 0.0)),
                "loss_unchanged": float(out.get("loss_unchanged", 0.0)),
                "change_frac": float(out.get("change_frac", 0.0)),
                "acc_g": float(out["acc_g"]),
                "acc_f": float(out["acc_f"]),
                "acc_b": float(out["acc_b"]),
                "ss_p": float(out.get("ss_p", 0.0)),
                "rollout_train_steps": float(out.get("rollout_train_steps", 1.0)),
                "sec": elapsed,
                "vram_mb": (
                    torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0
                ),
            }
            print(
                f"step {step}/{max_steps} loss={row['loss']:.3f} "
                f"acc_g={row['acc_g']:.3f} chg={row['change_frac']:.3f} "
                f"ss_p={row['ss_p']:.2f} r={row['rollout_train_steps']:.0f} "
                f"vram={row['vram_mb']:.0f}MB",
                flush=True,
            )
            with metrics_path.open("a") as f:
                f.write(json.dumps(row) + "\n")

        if step % sample_every == 0:
            model.eval()
            losses = []
            with torch.no_grad():
                for i, vb in enumerate(val_loader):
                    if i >= 32:
                        break
                    vb = {k: v.to(device) for k, v in vb.items()}
                    with torch.autocast(
                        device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
                    ):
                        # val: single-step TF on first target
                        tg, tf, tb = _slice_target(vb, 0)
                        vo = model.loss(
                            vb["ctx_g"],
                            vb["ctx_f"],
                            vb["ctx_b"],
                            tg,
                            tf,
                            tb,
                            change_weight=change_weight,
                            use_delta=use_delta,
                            prev_g=vb.get("prev_g"),
                            prev_f=vb.get("prev_f"),
                            prev_b=vb.get("prev_b"),
                        )
                    losses.append(float(vo["loss"]))
            val_loss = sum(losses) / max(len(losses), 1)

            # rollout stability probe
            vb0 = next(iter(val_loader))
            roll = sample_rollout(
                model,
                vb0,
                steps=min(48, 96),
                H=H,
                W=W,
                patch_h=ph,
                patch_w=pw,
                device=device,
            )
            mean_any = float(roll.extra.get("token_change", {}).get("mean_any_change", 0.0))
            growth = float(roll.extra.get("blob", {}).get("growth_slope", 0.0))
            # rollout_stability: higher motion + lower blob growth is better
            # store as lower-is-better metric: -mean_any + normalized growth penalty
            rollout_stability_score = -mean_any + 0.001 * max(0.0, growth)
            print(
                f"  val_loss≈{val_loss:.3f} rollout_mean_any={mean_any:.4f} "
                f"blob_slope={growth:.2f}",
                flush=True,
            )
            with metrics_path.open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "step": step,
                            "val_loss": val_loss,
                            "rollout_mean_any_change": mean_any,
                            "rollout_blob_growth_slope": growth,
                            "rollout_stability": -rollout_stability_score,
                        }
                    )
                    + "\n"
                )

            if early_stop_on == "rollout_stability":
                metric = rollout_stability_score
            else:
                metric = val_loss

            if metric < best_metric - 1e-4:
                best_metric = metric
                best_val = val_loss
                best_step = step
                bad_checks = 0
                _save_ckpt("ckpt_best")
                print(f"  saved ckpt_best.pt (val={best_val:.3f} metric={metric:.4f})", flush=True)
            else:
                bad_checks += 1
                print(
                    f"  no improve ({bad_checks}/{early_stop_patience or 'off'})",
                    flush=True,
                )

            save_avm(run_dir / f"rollout_step{step:06d}.avm.npz", roll)
            model.train()

            if (
                early_stop_patience > 0
                and step >= min_steps
                and bad_checks >= early_stop_patience
            ):
                print(
                    f"early stop at step {step} (best_step={best_step} val={best_val:.3f})",
                    flush=True,
                )
                _save_ckpt("ckpt_last")
                break

        if step % ckpt_every == 0 or step == max_steps:
            _save_ckpt(f"ckpt_step{step:06d}")
            _save_ckpt("ckpt_last")

    if not (run_dir / "ckpt_best.pt").is_file() and (run_dir / "ckpt_last.pt").is_file():
        import shutil

        shutil.copy(run_dir / "ckpt_last.pt", run_dir / "ckpt_best.pt")
        print("copied ckpt_last.pt -> ckpt_best.pt (no val improve recorded)")

    print(f"done. checkpoints in {run_dir}  best_step={best_step} best_val={best_val}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
