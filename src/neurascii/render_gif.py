"""Render .avm.npz symbolic ASCII video to animated GIF (deterministic, offline)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .format import glyph_id_to_char, load_avm, xterm256_to_rgb


def _font(cell: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    # Prefer monospace; fall back to default
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/UTF-8/Misc/10x20.pcf.gz",
    ):
        p = Path(path)
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), max(8, cell - 2))
            except OSError:
                continue
    return ImageFont.load_default()


def render_frame_rgb(
    glyphs: np.ndarray,
    fg: np.ndarray,
    bg: np.ndarray,
    *,
    cell: int = 10,
) -> Image.Image:
    h, w = glyphs.shape
    img = Image.new("RGB", (w * cell, h * cell), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = _font(cell)
    for y in range(h):
        for x in range(w):
            br, bg_, bb = xterm256_to_rgb(int(bg[y, x]))
            fr, fg_, fb = xterm256_to_rgb(int(fg[y, x]))
            x0, y0 = x * cell, y * cell
            draw.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=(br, bg_, bb))
            ch = glyph_id_to_char(int(glyphs[y, x]))
            if ch.strip():
                draw.text((x0 + 1, y0), ch, fill=(fr, fg_, fb), font=font)
    return img


def avm_to_gif(
    avm_path: str | Path,
    out_path: str | Path,
    *,
    fps: float = 6.0,
    cell: int = 8,
    max_frames: int = 0,
    max_width: int = 800,
) -> Path:
    video = load_avm(avm_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = video.T if max_frames <= 0 else min(video.T, max_frames)
    frames: list[Image.Image] = []
    for t in range(n):
        im = render_frame_rgb(video.glyphs[t], video.fg[t], video.bg[t], cell=cell)
        if max_width and im.width > max_width:
            ratio = max_width / im.width
            im = im.resize(
                (max_width, max(1, int(im.height * ratio))),
                Image.Resampling.NEAREST,
            )
        frames.append(im.convert("P", palette=Image.Palette.ADAPTIVE, colors=256))
    duration_ms = int(1000 / max(fps, 0.1))
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert .avm.npz to animated GIF")
    p.add_argument("avm", type=Path)
    p.add_argument("-o", "--out", type=Path, required=True)
    p.add_argument("--fps", type=float, default=6.0)
    p.add_argument("--cell", type=int, default=8)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--max-width", type=int, default=800)
    args = p.parse_args(argv)
    path = avm_to_gif(
        args.avm,
        args.out,
        fps=args.fps,
        cell=args.cell,
        max_frames=args.max_frames,
        max_width=args.max_width,
    )
    print(f"wrote {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
