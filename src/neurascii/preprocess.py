"""CLI: convert source videos → .avm.npz symbolic ASCII video."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from .caca_convert import convert_and_save
from .format import RendererConfig


def _load_cfg(path: Path | None) -> tuple[RendererConfig, dict]:
    raw: dict = {}
    if path and path.is_file():
        raw = yaml.safe_load(path.read_text()) or {}
    r = RendererConfig.from_dict(raw.get("renderer", raw))
    return r, raw


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Preprocess video → symbolic ASCII (.avm.npz)")
    p.add_argument("inputs", nargs="+", help="Video files or directories")
    p.add_argument("--config", type=Path, default=Path("configs/preprocess.yaml"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--limit", type=int, default=0, help="Max videos to process (0=all)")
    p.add_argument(
        "--class-prefix",
        default=None,
        help="Only class folders starting with this prefix (e.g. Apply). "
        "Empty string disables. Default: config paths.class_prefix",
    )
    args = p.parse_args(argv)

    cfg, raw = _load_cfg(args.config if args.config.exists() else None)
    out_dir = args.out_dir or Path(raw.get("paths", {}).get("out_dir", "data/processed"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.class_prefix is None:
        class_prefix = raw.get("paths", {}).get("class_prefix", "")
    else:
        class_prefix = args.class_prefix

    files: list[Path] = []
    for inp in args.inputs:
        path = Path(inp)
        if path.is_dir():
            for ext in ("*.avi", "*.mp4", "*.mkv", "*.webm", "*.mov"):
                files.extend(sorted(path.rglob(ext)))
        elif path.is_file():
            files.append(path)
        else:
            print(f"skip missing: {path}", file=sys.stderr)

    # de-dupe preserving order
    seen: set[Path] = set()
    uniq: list[Path] = []
    for f in files:
        rp = f.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(f)
    files = uniq
    if class_prefix:
        before = len(files)
        files = [f for f in files if f.parent.name.startswith(class_prefix)]
        print(f"class_prefix={class_prefix!r}: {before} -> {len(files)} videos", flush=True)
    if args.limit > 0:
        files = files[: args.limit]

    manifest: list[dict] = []
    for i, f in enumerate(files):
        rel = f.stem
        # keep class folder name if present
        parent = f.parent.name
        out_name = f"{parent}__{rel}.avm.npz" if parent not in (".", "") else f"{rel}.avm.npz"
        out_path = out_dir / out_name
        print(f"[{i+1}/{len(files)}] {f} -> {out_path}", flush=True)
        try:
            convert_and_save(f, out_path, cfg=cfg)
            manifest.append(
                {
                    "source": str(f),
                    "avm": str(out_path),
                    "class": parent,
                    "stem": rel,
                }
            )
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)

    man_path = out_dir / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(manifest)} clips; manifest {man_path}")
    return 0 if manifest else 1


if __name__ == "__main__":
    raise SystemExit(main())
