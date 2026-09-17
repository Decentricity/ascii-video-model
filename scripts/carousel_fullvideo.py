#!/usr/bin/env python3
"""Generate 10-GIF fullvideo carousel for neurascii.github.io (Phase C)."""
from __future__ import annotations

import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/decentricity/neurascii")
PAGES = Path("/home/decentricity/neurascii.github.io")
VENV_PY = ROOT / ".venv" / "bin" / "python"
LOG = Path("/tmp/carousel_fullvideo.log")
WANT = 10


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


def pick_class_seeds(processed: Path, want: int = WANT) -> list[tuple[str, str, Path]]:
    """One video per distinct class when possible; round-robin / even spread across classes."""
    by_class: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(processed.glob("*.avm.npz")):
        cls, _ = parse_seed(p)
        by_class[cls].append(p)
    if not by_class:
        raise SystemExit(f"no seeds in {processed}")

    classes = sorted(by_class.keys())
    log(f"CLASSES_AVAILABLE={len(classes)}")

    # Prefer diverse class coverage: evenly sample classes when many exist.
    if len(classes) >= want:
        step = len(classes) / want
        chosen_classes = [classes[int(i * step)] for i in range(want)]
        # de-dupe while preserving order (edge case on tiny step collisions)
        seen: set[str] = set()
        uniq: list[str] = []
        for c in chosen_classes:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        # fill any shortfall from remaining classes
        for c in classes:
            if len(uniq) >= want:
                break
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        return [(c, parse_seed(by_class[c][0])[1], by_class[c][0]) for c in uniq[:want]]

    # Fewer than want classes: round-robin second/third clips across classes.
    indices = {c: 0 for c in classes}
    out: list[tuple[str, str, Path]] = []
    while len(out) < want:
        progressed = False
        for c in classes:
            idx = indices[c]
            clips = by_class[c]
            if idx >= len(clips):
                continue
            p = clips[idx]
            indices[c] = idx + 1
            name = parse_seed(p)[1]
            out.append((c, name, p))
            progressed = True
            if len(out) >= want:
                break
        if not progressed:
            break
    return out


def run(cmd: list[str]) -> None:
    log("+ " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    LOG.write_text("", encoding="utf-8")
    ckpt = pick_ckpt(
        "runs/fullvideo/ckpt_best.pt",
        "runs/fullvideo/ckpt_last.pt",
    )
    if not ckpt:
        raise SystemExit("no fullvideo checkpoint (runs/fullvideo/ckpt_best.pt or ckpt_last.pt)")
    log(f"CKPT={ckpt}")

    processed = ROOT / "data/processed/fullvideo"
    if not processed.is_dir():
        raise SystemExit(f"missing processed dir: {processed}")
    seeds = pick_class_seeds(processed)
    if len(seeds) < WANT:
        raise SystemExit(f"only {len(seeds)} class seeds available (need {WANT})")
    log(f"SEEDS={[f'{c}/{n}' for c, n, _ in seeds]}")

    (ROOT / "samples" / "rollouts").mkdir(parents=True, exist_ok=True)
    gifs_dir = PAGES / "assets" / "gifs"
    gifs_dir.mkdir(parents=True, exist_ok=True)
    caps_path = gifs_dir / "fullvideo_captions.txt"
    caps_path.write_text("", encoding="utf-8")

    captions: list[str] = []
    for i, (cls, name, path) in enumerate(seeds, 1):
        caption = f"{cls} / {name}"
        out = ROOT / "samples" / "rollouts" / f"carousel_fullvideo_{i:02d}.avm.npz"
        gif = gifs_dir / f"fullvideo_{i:02d}_{cls}.gif"
        log(f"--- fullvideo {i:02d} {caption} ---")
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
                "--max-width",
                "640",
                "--max-frames",
                "100",
                "--cell",
                "7",
                "--fps",
                "6",
            ]
        )
        captions.append(caption)
        with caps_path.open("a", encoding="utf-8") as f:
            f.write(f"{i:02d}|{caption}\n")

    log(f"CAPTIONS={captions}")
    log("DONE")


if __name__ == "__main__":
    main()
