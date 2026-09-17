#!/usr/bin/env python3
"""Generate 10-GIF eye + lipstick carousels for neurascii.github.io."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/decentricity/neurascii")
PAGES = Path("/home/decentricity/neurascii.github.io")
VENV_PY = ROOT / ".venv" / "bin" / "python"
LOG = Path("/tmp/carousel_eye_lipstick.log")


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


def even_seeds(processed: Path, want: int = 10) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    manifest = processed / "manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text())
        for e in data:
            p = Path(e["avm"])
            if not p.is_file():
                p = ROOT / e["avm"]
            if not p.is_file():
                p = processed / Path(e["avm"]).name
            if p.is_file():
                stem = e.get("stem") or p.stem
                entries.append((stem, p))
    else:
        for p in sorted(processed.glob("*.avm.npz")):
            entries.append((p.stem, p))
    if not entries:
        raise SystemExit(f"no seeds in {processed}")
    n = len(entries)
    if n <= want:
        return entries
    step = n / want
    idxs: list[int] = []
    seen: set[int] = set()
    for i in range(want):
        idx = int(i * step)
        while idx in seen:
            idx = (idx + 1) % n
        seen.add(idx)
        idxs.append(idx)
    return [entries[i] for i in idxs]


def run(cmd: list[str]) -> None:
    log("+ " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def carousel(label: str, ckpt: Path, processed: Path, out_prefix: str, gif_prefix: str) -> list[str]:
    seeds = even_seeds(processed)
    captions: list[str] = []
    log(f"=== {label}: {len(seeds)} seeds, ckpt={ckpt} ===")
    (ROOT / "samples" / "rollouts").mkdir(parents=True, exist_ok=True)
    (PAGES / "assets" / "gifs").mkdir(parents=True, exist_ok=True)
    for i, (stem, path) in enumerate(seeds, 1):
        out = ROOT / "samples" / "rollouts" / f"{out_prefix}_{i:02d}.avm.npz"
        gif = PAGES / "assets" / "gifs" / f"{gif_prefix}_{i:02d}.gif"
        log(f"--- {label} {i:02d} {stem} ---")
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
        captions.append(stem)
        caps = PAGES / "assets" / "gifs" / f"{gif_prefix}_captions.txt"
        with caps.open("a", encoding="utf-8") as f:
            f.write(f"{i:02d}|{stem}\n")
    return captions


def main() -> None:
    LOG.write_text("", encoding="utf-8")
    eye = pick_ckpt(
        "runs/poc_eye_best/ckpt_best.pt",
        "runs/poc_eye/ckpt_last.pt",
        "runs/poc_eye/ckpt_best.pt",
        "runs/poc_eye_best/ckpt_last.pt",
    )
    lip = pick_ckpt(
        "runs/poc_lipstick/ckpt_best.pt",
        "runs/poc_lipstick/ckpt_last.pt",
    )
    log(f"EYE_CKPT={eye}")
    log(f"LIP_CKPT={lip}")
    if not eye:
        raise SystemExit("Eye carousel stopped: no checkpoint")
    if not lip:
        raise SystemExit("Lipstick carousel stopped: no checkpoint")

    for prefix in ("eye", "lipstick"):
        caps = PAGES / "assets" / "gifs" / f"{prefix}_captions.txt"
        if caps.exists():
            caps.unlink()

    eye_caps = carousel("eye", eye, ROOT / "data/processed/poc_eye", "carousel_eye", "eye")
    lip_caps = carousel(
        "lipstick", lip, ROOT / "data/processed/poc_lipstick", "carousel_lipstick", "lipstick"
    )
    log(f"EYE_SEEDS={eye_caps}")
    log(f"LIP_SEEDS={lip_caps}")
    log("DONE")


if __name__ == "__main__":
    main()
