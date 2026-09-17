"""Patch tokenization and next-frame datasets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .format import REDUCED_ASCII_GLYPHS, load_avm


GLYPH_VOCAB = len(REDUCED_ASCII_GLYPHS)
COLOR_VOCAB = 256


def frames_to_patches(
    glyphs: np.ndarray,
    fg: np.ndarray,
    bg: np.ndarray,
    patch_h: int,
    patch_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """T,H,W → T,Ph,Pw,ph,pw (keep cell attrs inside patches)."""
    t, h, w = glyphs.shape
    assert h % patch_h == 0 and w % patch_w == 0
    ph, pw = h // patch_h, w // patch_w
    def _reshape(a: np.ndarray) -> np.ndarray:
        return (
            a.reshape(t, ph, patch_h, pw, patch_w)
            .transpose(0, 1, 3, 2, 4)
            .reshape(t, ph * pw, patch_h * patch_w)
        )

    return _reshape(glyphs), _reshape(fg), _reshape(bg)


def patches_to_frames(
    g_pat: np.ndarray,
    f_pat: np.ndarray,
    b_pat: np.ndarray,
    h: int,
    w: int,
    patch_h: int,
    patch_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of frames_to_patches for a single timestep or T."""
    single = g_pat.ndim == 2
    if single:
        g_pat = g_pat[None]
        f_pat = f_pat[None]
        b_pat = b_pat[None]
    t, n, cells = g_pat.shape
    ph, pw = h // patch_h, w // patch_w
    assert n == ph * pw and cells == patch_h * patch_w

    def _inv(a: np.ndarray) -> np.ndarray:
        return (
            a.reshape(t, ph, pw, patch_h, patch_w)
            .transpose(0, 1, 3, 2, 4)
            .reshape(t, h, w)
        )

    out = _inv(g_pat), _inv(f_pat), _inv(b_pat)
    if single:
        return out[0][0], out[1][0], out[2][0]
    return out


class NextFrameDataset(Dataset):
    """Windows of context_frames → next frame (as flat patch cell sequences)."""

    def __init__(
        self,
        avm_paths: list[Path],
        *,
        context_frames: int = 8,
        patch_h: int = 4,
        patch_w: int = 4,
    ) -> None:
        self.context = context_frames
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.samples: list[tuple[int, int]] = []  # (video_idx, start_t)
        self.videos: list[dict] = []
        for path in avm_paths:
            v = load_avm(path)
            g, f, b = frames_to_patches(
                v.glyphs, v.fg, v.bg, patch_h, patch_w
            )
            self.videos.append(
                {
                    "path": str(path),
                    "g": g,
                    "f": f,
                    "b": b,
                    "H": v.H,
                    "W": v.W,
                    "T": v.T,
                }
            )
            # need context + 1 target
            max_start = v.T - (context_frames + 1)
            for t0 in range(max(0, max_start + 1)):
                self.samples.append((len(self.videos) - 1, t0))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        vi, t0 = self.samples[idx]
        v = self.videos[vi]
        t1 = t0 + self.context
        ctx_g = v["g"][t0:t1]  # C, N, cells
        ctx_f = v["f"][t0:t1]
        ctx_b = v["b"][t0:t1]
        tgt_g = v["g"][t1]
        tgt_f = v["f"][t1]
        tgt_b = v["b"][t1]
        return {
            "ctx_g": torch.from_numpy(ctx_g.astype(np.int64)),
            "ctx_f": torch.from_numpy(ctx_f.astype(np.int64)),
            "ctx_b": torch.from_numpy(ctx_b.astype(np.int64)),
            "tgt_g": torch.from_numpy(tgt_g.astype(np.int64)),
            "tgt_f": torch.from_numpy(tgt_f.astype(np.int64)),
            "tgt_b": torch.from_numpy(tgt_b.astype(np.int64)),
        }


def discover_avm(processed_dir: str | Path, split_file: Path | None = None) -> list[Path]:
    processed_dir = Path(processed_dir)
    if split_file and split_file.is_file():
        names = json.loads(split_file.read_text())
        return [processed_dir / n for n in names if (processed_dir / n).is_file()]
    return sorted(processed_dir.glob("*.avm.npz"))
