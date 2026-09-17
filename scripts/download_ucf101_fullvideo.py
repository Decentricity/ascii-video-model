#!/usr/bin/env python3
"""Download bitmind/UCF101Fullvideo, prefer videos, unzip if needed, normalize paths."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


VIDEO_EXTS = {".avi", ".mp4", ".mkv", ".webm", ".mov", ".mpg", ".mpeg", ".m4v"}


def normalize_windows_paths(root: Path) -> int:
    """HF / zip may store paths like 'Videos\\Class\\file.avi' as flat names."""
    moved = 0
    for p in list(root.rglob("*")):
        if not p.is_file():
            continue
        name = p.name
        if "\\" not in name:
            continue
        dest = p.parent / name.replace("\\", "/")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.resolve() != p.resolve():
            shutil.move(str(p), str(dest))
            moved += 1
    # Flatten oddly named "Videos dump" dirs if present
    for dump in list(root.rglob("*Videos dump*")) + list(root.glob("*Videos*dump*")):
        if not dump.is_dir():
            continue
        for child in list(dump.iterdir()):
            dest = (
                root / "Videos" / child.name
                if dump.parent == root
                else dump.parent / child.name
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.move(str(child), str(dest))
                moved += 1
    return moved


def list_videos(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def extract_zips(root: Path) -> list[Path]:
    """Extract top-level / nested zips so video files become visible on disk."""
    extracted: list[Path] = []
    for zpath in sorted(root.rglob("*.zip")):
        print(f"Extracting {zpath.relative_to(root)} …", flush=True)
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(root)
        extracted.append(zpath)
    return extracted


def delete_extracted_zips(zips: list[Path]) -> int:
    """Remove zip archives after successful extract to reclaim disk space."""
    deleted = 0
    for zpath in zips:
        try:
            zpath.unlink(missing_ok=True)
            deleted += 1
            print(f"deleted zip {zpath}", flush=True)
        except OSError as exc:
            print(f"WARNING: could not delete {zpath}: {exc}", flush=True)
    return deleted


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--local-dir",
        type=Path,
        default=Path("data/raw/UCF101Fullvideo"),
    )
    p.add_argument("--repo", default="bitmind/UCF101Fullvideo")
    args = p.parse_args()
    args.local_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    info = api.dataset_info(args.repo)
    siblings = [s.rfilename for s in (info.siblings or [])]
    video_files = [f for f in siblings if Path(f).suffix.lower() in VIDEO_EXTS]
    parquet_files = [f for f in siblings if f.endswith(".parquet")]
    zip_files = [f for f in siblings if f.lower().endswith(".zip")]
    print(
        f"repo={args.repo} siblings={len(siblings)} "
        f"videos={len(video_files)} parquets={len(parquet_files)} zips={len(zip_files)}",
        flush=True,
    )

    if video_files:
        allow = list(video_files) + [
            "*.csv",
            "*.json",
            "*.txt",
            ".gitattributes",
            "README*",
        ]
        print(f"Downloading {len(video_files)} video files (+ metadata)…", flush=True)
        path = snapshot_download(
            repo_id=args.repo,
            repo_type="dataset",
            local_dir=str(args.local_dir),
            allow_patterns=allow,
        )
    else:
        # Full snapshot (e.g. single zip archive) then inspect / extract
        print("No video extensions in file list — downloading full snapshot…", flush=True)
        path = snapshot_download(
            repo_id=args.repo,
            repo_type="dataset",
            local_dir=str(args.local_dir),
        )

    root = Path(path)
    avis = list_videos(root)
    if not avis:
        zips = extract_zips(root)
        if zips:
            print(f"extracted {len(zips)} zip archive(s)", flush=True)
        avis = list_videos(root)
        if avis and zips:
            n_del = delete_extracted_zips(zips)
            print(f"reclaimed space: deleted {n_del} zip archive(s)", flush=True)

    n = normalize_windows_paths(root)
    if n:
        print(f"normalized {n} Windows-style / dump path entries", flush=True)
    avis = list_videos(root)

    if not avis:
        top = sorted([x.name for x in root.iterdir()][:50])
        print(f"WARNING: no video files found. top-level: {top}", flush=True)
        sample = sorted(
            str(x.relative_to(root)) for x in root.rglob("*") if x.is_file()
        )[:40]
        print(f"sample files: {sample}", flush=True)

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
    print(f"local_dir={root}", flush=True)
    print(f"videos={len(avis)} n_classes={len(counts)}", flush=True)
    print(f"manifest={man}", flush=True)
    return 0 if avis else 1


if __name__ == "__main__":
    raise SystemExit(main())
