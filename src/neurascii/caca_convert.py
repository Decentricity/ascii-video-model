"""libcaca video→symbolic grid conversion via ctypes (no python-caca package).

Reuses the same libcaca shared library already on this machine. Playback env
policy for live display remains ~/bin/mplay-caca; this module only *extracts*
glyph/color lattices — it never trains on terminal screenshots.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import subprocess
from pathlib import Path
from typing import Iterator

import numpy as np

from .format import (
    REDUCED_ASCII_GLYPHS,
    RendererConfig,
    AsciiVideo,
    glyph_char_to_id,
    rgb_to_xterm256,
    save_avm,
)


class LibCaca:
    def __init__(self) -> None:
        path = ctypes.util.find_library("caca")
        if not path:
            raise RuntimeError("libcaca not found (install libcaca0)")
        self.lib = ctypes.CDLL(path)
        L = self.lib
        L.caca_create_canvas.restype = ctypes.c_void_p
        L.caca_create_canvas.argtypes = [ctypes.c_int, ctypes.c_int]
        L.caca_free_canvas.argtypes = [ctypes.c_void_p]
        L.caca_create_dither.restype = ctypes.c_void_p
        L.caca_create_dither.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
        ]
        L.caca_free_dither.argtypes = [ctypes.c_void_p]
        L.caca_dither_bitmap.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        L.caca_set_dither_charset.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        L.caca_set_dither_color.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        L.caca_get_char.restype = ctypes.c_uint32
        L.caca_get_char.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        L.caca_get_attr.restype = ctypes.c_uint32
        L.caca_get_attr.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        L.caca_attr_to_rgb12_fg.restype = ctypes.c_uint16
        L.caca_attr_to_rgb12_fg.argtypes = [ctypes.c_uint32]
        L.caca_attr_to_rgb12_bg.restype = ctypes.c_uint16
        L.caca_attr_to_rgb12_bg.argtypes = [ctypes.c_uint32]


_CACA: LibCaca | None = None


def _caca() -> LibCaca:
    global _CACA
    if _CACA is None:
        _CACA = LibCaca()
    return _CACA


def _rgb12_to_rgb8(v: int) -> tuple[int, int, int]:
    """caca rgb12: 4 bits per channel packed as 0xRGB."""
    v = int(v) & 0xFFF
    r = ((v >> 8) & 0xF) * 17
    g = ((v >> 4) & 0xF) * 17
    b = (v & 0xF) * 17
    return r, g, b


def dither_rgb_frame(
    rgb: np.ndarray,
    cfg: RendererConfig,
    caca: LibCaca | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """rgb HxWx3 uint8 → glyph/fg/bg grids of shape (height_chars, width_chars)."""
    caca = caca or _caca()
    L = caca.lib
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 uint8, got {rgb.shape} {rgb.dtype}")
    src_h, src_w, _ = rgb.shape
    cw, ch = cfg.width_chars, cfg.height_chars

    # contiguous RGB24 buffer; masks match little-endian channel order in memory
    flat = np.ascontiguousarray(rgb)
    buf = (ctypes.c_uint8 * flat.size).from_buffer_copy(flat.tobytes())

    cv = L.caca_create_canvas(cw, ch)
    if not cv:
        raise RuntimeError("caca_create_canvas failed")
    try:
        dith = L.caca_create_dither(
            24,
            src_w,
            src_h,
            src_w * 3,
            0x0000FF,
            0x00FF00,
            0xFF0000,
            0,
        )
        if not dith:
            raise RuntimeError("caca_create_dither failed")
        try:
            L.caca_set_dither_charset(dith, cfg.dither_charset.encode())
            L.caca_set_dither_color(dith, cfg.dither_color.encode())
            L.caca_dither_bitmap(cv, 0, 0, cw, ch, dith, buf)
        finally:
            L.caca_free_dither(dith)

        glyphs = np.zeros((ch, cw), dtype=np.uint16)
        fg = np.zeros((ch, cw), dtype=np.uint16)
        bg = np.zeros((ch, cw), dtype=np.uint16)
        for y in range(ch):
            for x in range(cw):
                code = int(L.caca_get_char(cv, x, y))
                glyphs[y, x] = glyph_char_to_id(code)
                attr = int(L.caca_get_attr(cv, x, y))
                fr, fg_, fb = _rgb12_to_rgb8(L.caca_attr_to_rgb12_fg(attr))
                br, bg_, bb = _rgb12_to_rgb8(L.caca_attr_to_rgb12_bg(attr))
                fg[y, x] = rgb_to_xterm256(fr, fg_, fb)
                bg[y, x] = rgb_to_xterm256(br, bg_, bb)
        return glyphs, fg, bg
    finally:
        L.caca_free_canvas(cv)


def iter_ffmpeg_frames(
    path: str | Path,
    *,
    fps: float,
    width: int,
    height: int,
) -> Iterator[np.ndarray]:
    """Yield RGB frames via ffmpeg (no OpenCV dependency)."""
    path = Path(path)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-vf",
        f"fps={fps},scale={width}:{height}:flags=bicubic",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    frame_bytes = width * height * 3
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
    finally:
        proc.kill()
        proc.wait()


def convert_video(path: str | Path, cfg: RendererConfig | None = None) -> AsciiVideo:
    cfg = cfg or RendererConfig()
    path = Path(path)
    glyphs_l: list[np.ndarray] = []
    fg_l: list[np.ndarray] = []
    bg_l: list[np.ndarray] = []
    caca = _caca()
    for frame in iter_ffmpeg_frames(
        path,
        fps=cfg.fps,
        width=cfg.source_width,
        height=cfg.source_height,
    ):
        g, f, b = dither_rgb_frame(frame, cfg, caca=caca)
        glyphs_l.append(g)
        fg_l.append(f)
        bg_l.append(b)
    if not glyphs_l:
        raise RuntimeError(f"no frames decoded from {path}")
    return AsciiVideo(
        glyphs=np.stack(glyphs_l, axis=0),
        fg=np.stack(fg_l, axis=0),
        bg=np.stack(bg_l, axis=0),
        renderer=cfg,
        source_path=str(path),
        extra={"glyph_vocab_size": len(REDUCED_ASCII_GLYPHS), "color_levels": 256},
    )


def convert_and_save(
    path: str | Path,
    out_path: str | Path,
    cfg: RendererConfig | None = None,
) -> Path:
    video = convert_video(path, cfg=cfg)
    return save_avm(out_path, video)
