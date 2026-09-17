#!/usr/bin/env python3
"""Short Phase C regression rollouts from eye + lipstick (or smoke) seeds."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/decentricity/neurascii")
VENV_PY = ROOT / ".venv" / "bin" / "python"
STEPS = 48


def pick_ckpt(*candidates: str) -> Path | None:
    for c in candidates:
        p = ROOT / c
        if p.is_file():
            return p
    return None


def first_avm(*dirs: Path) -> Path | None:
    for d in dirs:
        if not d.is_dir():
            continue
        hits = sorted(d.glob("*.avm.npz"))
        if hits:
            return hits[0]
    return None


def run(cmd: list[str]) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    ckpt = pick_ckpt(
        "runs/fullvideo/ckpt_best.pt",
        "runs/fullvideo/ckpt_last.pt",
    )
    if not ckpt:
        raise SystemExit("no fullvideo checkpoint (runs/fullvideo/ckpt_best.pt or ckpt_last.pt)")
    print(f"CKPT={ckpt}", flush=True)

    eye = first_avm(
        ROOT / "data/processed/poc_eye",
        ROOT / "data/processed/smoke_apply",
    )
    lip = first_avm(
        ROOT / "data/processed/poc_lipstick",
        ROOT / "data/processed/smoke_apply",
    )
    if not eye:
        raise SystemExit("no eye/smoke seed avm found")
    if not lip:
        raise SystemExit("no lipstick/smoke seed avm found")
    # Prefer a lipstick-class clip from smoke if eye and lip both fell back to smoke.
    if eye.parent.name == "smoke_apply" and lip.parent.name == "smoke_apply":
        lips = sorted((ROOT / "data/processed/smoke_apply").glob("ApplyLipstick__*.avm.npz"))
        eyes = sorted((ROOT / "data/processed/smoke_apply").glob("ApplyEyeMakeup__*.avm.npz"))
        if eyes:
            eye = eyes[0]
        if lips:
            lip = lips[0]

    out_dir = ROOT / "samples" / "rollouts"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = [
        ("eye", eye, out_dir / "regression_eye.avm.npz"),
        ("lipstick", lip, out_dir / "regression_lipstick.avm.npz"),
    ]
    for label, seed, out in pairs:
        print(f"--- regression {label}: seed={seed} -> {out}", flush=True)
        run(
            [
                str(VENV_PY),
                "-m",
                "neurascii.generate",
                "--checkpoint",
                str(ckpt),
                "--seed-avm",
                str(seed),
                "--out",
                str(out),
                "--steps",
                str(STEPS),
                "--temperature",
                "0",
            ]
        )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
