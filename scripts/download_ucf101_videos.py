#!/usr/bin/env python3
"""Download bitmind/UCF101-Videos and normalize Windows-style path filenames."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from huggingface_hub import snapshot_download


def normalize_windows_paths(root: Path) -> int:
    """HF repo stores paths like 'test\\ApplyEyeMakeup\\file.avi' as flat names."""
    moved = 0
    for p in list(root.glob("*.avi")):
        name = p.name
        if "\\" not in name and "/" not in name:
            continue
        dest = root / name.replace("\\", "/")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.resolve() != p.resolve():
            shutil.move(str(p), str(dest))
            moved += 1
    return moved


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--local-dir", type=Path, default=Path("data/raw/UCF101-Videos"))
    p.add_argument(
        "--apply-only",
        action="store_true",
        help="Download ApplyEyeMakeup + ApplyLipstick only (and CSVs)",
    )
    p.add_argument("--repo", default="bitmind/UCF101-Videos")
    args = p.parse_args()
    args.local_dir.mkdir(parents=True, exist_ok=True)

    if args.apply_only:
        allow = [
            "**/ApplyEyeMakeup/**",
            "**/ApplyLipstick/**",
            "*ApplyEyeMakeup*",
            "*ApplyLipstick*",
            "*.csv",
            ".gitattributes",
        ]
        print("Downloading ApplyEyeMakeup + ApplyLipstick…")
    else:
        allow = None
        print("Downloading full UCF101-Videos repo…")

    path = snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=str(args.local_dir),
        allow_patterns=allow,
    )
    root = Path(path)
    n = normalize_windows_paths(root)
    if n:
        print(f"normalized {n} Windows-style path filenames")

    avis = sorted(root.rglob("*.avi"))
    counts = Counter(a.parent.name for a in avis)
    manifest = [
        {
            "path": str(a.relative_to(root)),
            "class": a.parent.name,
            "split": a.parent.parent.name if a.parent.parent != root else "",
        }
        for a in avis
    ]
    man = root / "on_disk_manifest.json"
    man.write_text(json.dumps({"counts": dict(counts), "files": manifest}, indent=2))
    print(f"local_dir={root}")
    print(f"avis={len(avis)} counts={dict(counts)}")
    print(f"manifest={man}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
