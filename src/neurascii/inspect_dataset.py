"""Dataset inspection / statistics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .format import load_avm


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("processed_dir", type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    files = sorted(args.processed_dir.glob("*.avm.npz"))
    if not files:
        raise SystemExit(f"no avm in {args.processed_dir}")

    total_frames = 0
    glyph_c: Counter = Counter()
    fg_c: Counter = Counter()
    bg_c: Counter = Counter()
    deltas = []
    durations = []

    for f in files:
        v = load_avm(f)
        total_frames += v.T
        durations.append(v.T / max(v.renderer.fps, 1e-6))
        glyph_c.update(v.glyphs.ravel().tolist())
        fg_c.update(v.fg.ravel().tolist())
        bg_c.update(v.bg.ravel().tolist())
        if v.T > 1:
            changed = (
                (v.glyphs[1:] != v.glyphs[:-1])
                | (v.fg[1:] != v.fg[:-1])
                | (v.bg[1:] != v.bg[:-1])
            )
            deltas.append(float(changed.mean()))

    def entropy(counter: Counter) -> float:
        total = sum(counter.values())
        if total == 0:
            return 0.0
        ent = 0.0
        for c in counter.values():
            p = c / total
            ent -= p * np.log2(p)
        return float(ent)

    report = {
        "n_videos": len(files),
        "total_frames": total_frames,
        "total_seconds": float(sum(durations)),
        "mean_seconds": float(np.mean(durations)),
        "mean_frame_delta": float(np.mean(deltas)) if deltas else 0.0,
        "glyph_entropy_bits": entropy(glyph_c),
        "fg_entropy_bits": entropy(fg_c),
        "bg_entropy_bits": entropy(bg_c),
        "unique_glyphs": len(glyph_c),
        "unique_fg": len(fg_c),
        "unique_bg": len(bg_c),
        "shape_hw": list(load_avm(files[0]).glyphs.shape[1:]),
        "fps": load_avm(files[0]).renderer.fps,
    }
    text = json.dumps(report, indent=2)
    print(text)
    out = args.out or (args.processed_dir / "inspect.json")
    out.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
