"""Canonical symbolic ASCII-video format (.avm.npz).

Each archive stores:
  glyphs: uint16 [T, H, W]  — indices into glyph_vocab
  fg:     uint16 [T, H, W]  — xterm-256 foreground IDs
  bg:     uint16 [T, H, W]  — xterm-256 background IDs
  meta JSON sidecar keys in the npz under 'meta_json'
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FORMAT_VERSION = 1

# Reduced printable ASCII (space .. ~) — deterministic, versioned.
REDUCED_ASCII_GLYPHS: str = "".join(chr(c) for c in range(32, 127))


@dataclass
class RendererConfig:
    version: str = "libcaca-ctypes-v1"
    width_chars: int = 80
    height_chars: int = 48
    fps: float = 10.0
    glyph_vocab: str = "reduced_ascii"
    color_palette: str = "xterm256"
    dither_charset: str = "ascii"
    dither_color: str = "full"
    source_width: int = 320
    source_height: int = 240

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RendererConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class AsciiVideo:
    glyphs: np.ndarray  # T,H,W uint16
    fg: np.ndarray
    bg: np.ndarray
    renderer: RendererConfig = field(default_factory=RendererConfig)
    source_path: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, arr in (("glyphs", self.glyphs), ("fg", self.fg), ("bg", self.bg)):
            if arr.ndim != 3:
                raise ValueError(f"{name} must be T,H,W got {arr.shape}")
        if self.glyphs.shape != self.fg.shape or self.glyphs.shape != self.bg.shape:
            raise ValueError("glyphs/fg/bg shape mismatch")
        self.glyphs = np.asarray(self.glyphs, dtype=np.uint16)
        self.fg = np.asarray(self.fg, dtype=np.uint16)
        self.bg = np.asarray(self.bg, dtype=np.uint16)

    @property
    def T(self) -> int:
        return int(self.glyphs.shape[0])

    @property
    def H(self) -> int:
        return int(self.glyphs.shape[1])

    @property
    def W(self) -> int:
        return int(self.glyphs.shape[2])

    def meta(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "renderer": self.renderer.to_dict(),
            "shape": list(self.glyphs.shape),
            "source_path": self.source_path,
            "glyph_chars": REDUCED_ASCII_GLYPHS,
            "extra": self.extra,
        }


def glyph_char_to_id(ch: int | str) -> int:
    if isinstance(ch, str):
        ch = ord(ch) if ch else 32
    if 32 <= ch <= 126:
        return ch - 32
    return 0  # space


def glyph_id_to_char(gid: int) -> str:
    gid = int(gid)
    if 0 <= gid < len(REDUCED_ASCII_GLYPHS):
        return REDUCED_ASCII_GLYPHS[gid]
    return " "


def save_avm(path: str | Path, video: AsciiVideo) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix != ".npz" and not str(path).endswith(".avm.npz"):
        path = path.with_suffix(".avm.npz")
    meta_json = json.dumps(video.meta(), separators=(",", ":"))
    np.savez_compressed(
        path,
        glyphs=video.glyphs,
        fg=video.fg,
        bg=video.bg,
        meta_json=np.array(meta_json),
    )
    return path


def load_avm(path: str | Path) -> AsciiVideo:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        glyphs = z["glyphs"]
        fg = z["fg"]
        bg = z["bg"]
        meta = json.loads(str(z["meta_json"]))
    renderer = RendererConfig.from_dict(meta.get("renderer", {}))
    return AsciiVideo(
        glyphs=glyphs,
        fg=fg,
        bg=bg,
        renderer=renderer,
        source_path=meta.get("source_path", ""),
        extra=meta.get("extra", {}),
    )


def rgb_to_xterm256(r: int, g: int, b: int) -> int:
    """Map 0–255 RGB to xterm-256 color index."""
    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))
    # grayscale cube shortcut
    if r == g == b:
        if r < 8:
            return 16
        if r > 248:
            return 231
        return 232 + int(round((r - 8) / 247 * 23))
    ri = int(round(r / 255 * 5))
    gi = int(round(g / 255 * 5))
    bi = int(round(b / 255 * 5))
    return 16 + 36 * ri + 6 * gi + bi


def xterm256_to_rgb(idx: int) -> tuple[int, int, int]:
    idx = int(idx) & 0xFF
    if idx < 16:
        table = [
            (0, 0, 0),
            (128, 0, 0),
            (0, 128, 0),
            (128, 128, 0),
            (0, 0, 128),
            (128, 0, 128),
            (0, 128, 128),
            (192, 192, 192),
            (128, 128, 128),
            (255, 0, 0),
            (0, 255, 0),
            (255, 255, 0),
            (0, 0, 255),
            (255, 0, 255),
            (0, 255, 255),
            (255, 255, 255),
        ]
        return table[idx]
    if 16 <= idx <= 231:
        i = idx - 16
        ri, gi, bi = i // 36, (i // 6) % 6, i % 6
        levels = [0, 95, 135, 175, 215, 255]
        return levels[ri], levels[gi], levels[bi]
    gray = 8 + (idx - 232) * 10
    return gray, gray, gray


def frame_to_ansi(glyphs: np.ndarray, fg: np.ndarray, bg: np.ndarray) -> str:
    """Render one HxW symbolic frame to ANSI (truecolor via xterm palette)."""
    h, w = glyphs.shape
    lines: list[str] = []
    for y in range(h):
        parts: list[str] = []
        for x in range(w):
            ch = glyph_id_to_char(int(glyphs[y, x]))
            fr, fg_, fb = xterm256_to_rgb(int(fg[y, x]))
            br, bg_, bb = xterm256_to_rgb(int(bg[y, x]))
            parts.append(
                f"\033[38;2;{fr};{fg_};{fb}m\033[48;2;{br};{bg_};{bb}m{ch}"
            )
        parts.append("\033[0m")
        lines.append("".join(parts))
    return "\n".join(lines)
