#!/usr/bin/env python3
"""Copy one long ApplyEyeMakeup avm into data/processed/overfit_eye/."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from neurascii.format import load_avm


def pick_longest(src: Path) -> Path:
    best = None
    best_t = -1
    for p in sorted(src.glob("ApplyEyeMakeup__*.avm.npz")):
        try:
            v = load_avm(p)
        except Exception:
            continue
        if v.T > best_t:
            best_t = v.T
            best = p
    if best is None:
        raise SystemExit(f"no ApplyEyeMakeup clips in {src}")
    return best


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=_ROOT / "data/processed/poc_eye")
    p.add_argument("--dst", type=Path, default=_ROOT / "data/processed/overfit_eye")
    args = p.parse_args(argv)

    src = pick_longest(args.src)
    args.dst.mkdir(parents=True, exist_ok=True)
    # Clear prior copies so dataset is truly single-clip
    for old in args.dst.glob("*.avm.npz"):
        old.unlink()
    dst = args.dst / src.name
    shutil.copy2(src, dst)
    v = load_avm(dst)
    print(f"OVERFIT_SRC={src}")
    print(f"OVERFIT_DST={dst}")
    print(f"T={v.T} H={v.H} W={v.W}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
