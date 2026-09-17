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
from .model import build_model_from_config


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
        lg, lf, lb = model(ctx_g, ctx_f, ctx_b)
        pg = lg.argmax(-1)
        pf = lf.argmax(-1)
        pb = lb.argmax(-1)
        out_g.append(pg[0].cpu().numpy())
        out_f.append(pf[0].cpu().numpy())
        out_b.append(pb[0].cpu().numpy())
        ctx_g = torch.cat([ctx_g[:, 1:], pg.unsqueeze(1)], dim=1)
        ctx_f = torch.cat([ctx_f[:, 1:], pf.unsqueeze(1)], dim=1)
        ctx_b = torch.cat([ctx_b[:, 1:], pb.unsqueeze(1)], dim=1)
    g = np.stack(out_g)
    f = np.stack(out_f)
    b = np.stack(out_b)
    # convert patches → frames
    gf, ff, bf = patches_to_frames(g, f, b, H, W, patch_h, patch_w)
    return AsciiVideo(
        glyphs=gf.astype(np.uint16),
        fg=ff.astype(np.uint16),
        bg=bf.astype(np.uint16),
        renderer=RendererConfig(width_chars=W, height_chars=H),
        extra={"kind": "rollout"},
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-steps", type=int, default=None)
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
    train_ds = NextFrameDataset(train_paths, context_frames=ctx, patch_h=ph, patch_w=pw)
    val_ds = NextFrameDataset(val_paths, context_frames=ctx, patch_h=ph, patch_w=pw)
    if len(train_ds) == 0:
        raise SystemExit("train dataset empty — need longer clips or fewer context frames")

    # infer n_patches from first sample
    sample0 = train_ds[0]
    n_patches = sample0["tgt_g"].shape[0]
    H = train_ds.videos[0]["H"]
    W = train_ds.videos[0]["W"]

    model = build_model_from_config(cfg, n_patches=n_patches).to(device)
    n_params = model.count_parameters()
    print(f"parameters: {n_params/1e6:.2f}M  n_patches={n_patches}  train_windows={len(train_ds)}")

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

    step = 0
    model.train()
    t0 = time.time()
    data_iter = iter(loader)
    metrics_path = run_dir / "metrics.jsonl"
    best_val = float("inf")
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
            out = model.loss(**batch)
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
                "acc_g": float(out["acc_g"]),
                "acc_f": float(out["acc_f"]),
                "acc_b": float(out["acc_b"]),
                "sec": elapsed,
                "vram_mb": (
                    torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0
                ),
            }
            print(
                f"step {step}/{max_steps} loss={row['loss']:.3f} "
                f"acc_g={row['acc_g']:.3f} acc_f={row['acc_f']:.3f} "
                f"vram={row['vram_mb']:.0f}MB",
                flush=True,
            )
            with metrics_path.open("a") as f:
                f.write(json.dumps(row) + "\n")

        if step % sample_every == 0:
            # val loss
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
                        vo = model.loss(**vb)
                    losses.append(float(vo["loss"]))
            val_loss = sum(losses) / max(len(losses), 1)
            print(f"  val_loss≈{val_loss:.3f}", flush=True)
            with metrics_path.open("a") as f:
                f.write(json.dumps({"step": step, "val_loss": val_loss}) + "\n")

            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_step = step
                bad_checks = 0
                _save_ckpt("ckpt_best")
                print(f"  saved ckpt_best.pt (val={best_val:.3f})", flush=True)
            else:
                bad_checks += 1
                print(
                    f"  no val improve ({bad_checks}/{early_stop_patience or 'off'})",
                    flush=True,
                )

            # rollout sample
            vb0 = next(iter(val_loader))
            roll = sample_rollout(
                model,
                vb0,
                steps=min(24, 48),
                H=H,
                W=W,
                patch_h=ph,
                patch_w=pw,
                device=device,
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
        # ensure a best pointer exists for downstream scripts
        import shutil

        shutil.copy(run_dir / "ckpt_last.pt", run_dir / "ckpt_best.pt")
        print("copied ckpt_last.pt -> ckpt_best.pt (no val improve recorded)")

    print(f"done. checkpoints in {run_dir}  best_step={best_step} best_val={best_val}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
