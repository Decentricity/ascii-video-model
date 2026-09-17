#!/usr/bin/env python3
"""Generate 10-GIF Merged Apply* smoke carousel for neurascii.github.io."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/decentricity/neurascii")
PAGES = Path("/home/decentricity/neurascii.github.io")
VENV_PY = ROOT / ".venv" / "bin" / "python"
LOG = Path("/tmp/carousel_smoke_apply.log")


def log(msg: str) -> None:
    line = msg.rstrip() + "\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def pick_ckpt(*candidates: str) -> Path | None:
    for c in candidates:
        p = ROOT / c
        if p.is_file():
            return p
    return None


def even(xs: list[Path], n: int) -> list[Path]:
    if len(xs) <= n:
        return xs
    step = len(xs) / n
    seen: set[int] = set()
    out: list[Path] = []
    for i in range(n):
        idx = int(i * step)
        while idx in seen:
            idx = (idx + 1) % len(xs)
        seen.add(idx)
        out.append(xs[idx])
    return out


def parse_seed(path: Path) -> tuple[str, str]:
    stem = path.name
    if stem.endswith(".avm.npz"):
        stem = stem[: -len(".avm.npz")]
    if "__" in stem:
        cls, name = stem.split("__", 1)
    else:
        cls, name = "unknown", stem
    if name.endswith(".avm"):
        name = name[: -len(".avm")]
    return cls, name


def pick_seeds(processed: Path) -> list[tuple[str, str, Path]]:
    eye = sorted(processed.glob("ApplyEyeMakeup__*.avm.npz"))
    lip = sorted(processed.glob("ApplyLipstick__*.avm.npz"))
    if not eye and not lip:
        # fallback: mix from poc_eye + poc_lipstick
        eye = sorted((ROOT / "data/processed/poc_eye").glob("*.avm.npz"))
        lip = sorted((ROOT / "data/processed/poc_lipstick").glob("*.avm.npz"))
        log(f"fallback to poc_eye ({len(eye)}) + poc_lipstick ({len(lip)})")
    picked = even(eye, 5) + even(lip, 5)
    if len(picked) < 10:
        # fill from remaining
        all_seeds = sorted(processed.glob("*.avm.npz")) if processed.is_dir() else []
        have = {p.resolve() for p in picked}
        for p in all_seeds:
            if p.resolve() in have:
                continue
            picked.append(p)
            if len(picked) >= 10:
                break
    out: list[tuple[str, str, Path]] = []
    for p in picked[:10]:
        cls, name = parse_seed(p)
        out.append((cls, name, p))
    return out


def run(cmd: list[str]) -> None:
    log("+ " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    LOG.write_text("", encoding="utf-8")
    ckpt = pick_ckpt(
        "runs/smoke_apply/ckpt_best.pt",
        "runs/smoke_apply/ckpt_last.pt",
    )
    if not ckpt:
        raise SystemExit("no smoke_apply checkpoint")
    log(f"CKPT={ckpt}")

    processed = ROOT / "data/processed/smoke_apply"
    seeds = pick_seeds(processed)
    if len(seeds) < 10:
        raise SystemExit(f"only {len(seeds)} seeds available")
    log(f"SEEDS={[f'{c}/{n}' for c, n, _ in seeds]}")

    (ROOT / "samples" / "rollouts").mkdir(parents=True, exist_ok=True)
    (PAGES / "assets" / "gifs").mkdir(parents=True, exist_ok=True)
    caps_path = PAGES / "assets" / "gifs" / "smoke_captions.txt"
    caps_path.write_text("", encoding="utf-8")

    captions: list[str] = []
    for i, (cls, name, path) in enumerate(seeds, 1):
        caption = f"{cls} / {name}"
        out = ROOT / "samples" / "rollouts" / f"carousel_smoke_{i:02d}.avm.npz"
        gif = PAGES / "assets" / "gifs" / f"smoke_{i:02d}.gif"
        log(f"--- smoke {i:02d} {caption} ---")
        run(
            [
                str(VENV_PY),
                "-m",
                "neurascii.generate",
                "--checkpoint",
                str(ckpt),
                "--seed-avm",
                str(path),
                "--out",
                str(out),
                "--steps",
                "100",
            ]
        )
        run(
            [
                str(VENV_PY),
                "-m",
                "neurascii.render_gif",
                str(out),
                "-o",
                str(gif),
                "--fps",
                "6",
                "--max-frames",
                "120",
                "--max-width",
                "720",
                "--cell",
                "8",
            ]
        )
        captions.append(caption)
        with caps_path.open("a", encoding="utf-8") as f:
            f.write(f"{i:02d}|{caption}\n")

    log(f"CAPTIONS={captions}")
    log("DONE")


if __name__ == "__main__":
    main()
