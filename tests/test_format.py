"""Unit tests for format roundtrip and patch reshape."""

from __future__ import annotations

import numpy as np

from neurascii.dataset import frames_to_patches, patches_to_frames
from neurascii.format import (
    AsciiVideo,
    RendererConfig,
    glyph_char_to_id,
    glyph_id_to_char,
    load_avm,
    save_avm,
)


def test_glyph_roundtrip():
    for ch in " Aa~@#":
        assert glyph_id_to_char(glyph_char_to_id(ch)) == ch


def test_avm_roundtrip(tmp_path):
    T, H, W = 3, 8, 8
    video = AsciiVideo(
        glyphs=np.random.randint(0, 95, (T, H, W), dtype=np.uint16),
        fg=np.random.randint(0, 256, (T, H, W), dtype=np.uint16),
        bg=np.random.randint(0, 256, (T, H, W), dtype=np.uint16),
        renderer=RendererConfig(width_chars=W, height_chars=H, fps=10),
        source_path="synth",
    )
    path = save_avm(tmp_path / "t.avm.npz", video)
    loaded = load_avm(path)
    assert np.array_equal(loaded.glyphs, video.glyphs)
    assert np.array_equal(loaded.fg, video.fg)
    assert np.array_equal(loaded.bg, video.bg)
    assert loaded.renderer.fps == 10


def test_patch_roundtrip():
    T, H, W = 2, 48, 80
    g = np.random.randint(0, 95, (T, H, W), dtype=np.uint16)
    f = np.random.randint(0, 256, (T, H, W), dtype=np.uint16)
    b = np.random.randint(0, 256, (T, H, W), dtype=np.uint16)
    gp, fp, bp = frames_to_patches(g, f, b, 4, 4)
    assert gp.shape == (2, 240, 16)
    g2, f2, b2 = patches_to_frames(gp, fp, bp, H, W, 4, 4)
    assert np.array_equal(g, g2)
    assert np.array_equal(f, f2)
    assert np.array_equal(b, b2)
