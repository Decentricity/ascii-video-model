"""Terminal player for .avm.npz symbolic ASCII video."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .format import frame_to_ansi, load_avm


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Play symbolic ASCII video (.avm.npz)")
    p.add_argument("path", type=Path)
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--loop", action="store_true")
    p.add_argument("--no-color", action="store_true")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)
    p.add_argument("--max-frames", type=int, default=0)
    args = p.parse_args(argv)

    video = load_avm(args.path)
    fps = args.fps or float(video.renderer.fps)
    start = max(0, args.start)
    end = video.T if args.end < 0 else min(args.end, video.T)
    if args.max_frames > 0:
        end = min(end, start + args.max_frames)
    delay = 1.0 / max(fps, 1e-3)

    try:
        while True:
            for t in range(start, end):
                if args.no_color:
                    # glyph-only
                    rows = []
                    from .format import glyph_id_to_char

                    for y in range(video.H):
                        rows.append(
                            "".join(
                                glyph_id_to_char(int(video.glyphs[t, y, x]))
                                for x in range(video.W)
                            )
                        )
                    frame = "\n".join(rows)
                else:
                    frame = frame_to_ansi(video.glyphs[t], video.fg[t], video.bg[t])
                sys.stdout.write("\033[H\033[2J")
                sys.stdout.write(frame)
                sys.stdout.write(f"\n\033[0m[{t+1}/{video.T}] {args.path.name}\n")
                sys.stdout.flush()
                time.sleep(delay)
            if not args.loop:
                break
    except KeyboardInterrupt:
        sys.stdout.write("\033[0m\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
