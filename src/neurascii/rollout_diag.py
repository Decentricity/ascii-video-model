"""Rollout diagnostics for bug-blob / attractor analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _label_combined(glyphs: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """Combine glyph+fg into a single label map for connected components."""
    return (fg.astype(np.int32) << 8) | glyphs.astype(np.int32)


def largest_connected_region_size(label: np.ndarray) -> int:
    """Largest 4-connected region size for a single HxW label map."""
    return int(_largest_cc_size(np.asarray(label)))


def largest_connected_region_sizes(fg_or_glyph_frame: np.ndarray) -> list[int]:
    """Per-frame largest 4-connected region size.

    Accepts HxW or TxHxW integer frames (fg alone, or precombined (fg<<8)|glyph).
    """
    arr = np.asarray(fg_or_glyph_frame)
    if arr.ndim == 2:
        arr = arr[None]
    out: list[int] = []
    for t in range(arr.shape[0]):
        out.append(int(_largest_cc_size(arr[t])))
    return out


def _largest_cc_size(label: np.ndarray, *, ignore_label: int | None = 0) -> int:
    h, w = label.shape
    visited = np.zeros((h, w), dtype=bool)
    best = 0
    for y in range(h):
        for x in range(w):
            if visited[y, x]:
                continue
            val = int(label[y, x])
            if ignore_label is not None and val == ignore_label:
                visited[y, x] = True
                continue
            stack = [(y, x)]
            visited[y, x] = True
            size = 0
            while stack:
                cy, cx = stack.pop()
                size += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if ny < 0 or nx < 0 or ny >= h or nx >= w:
                        continue
                    if visited[ny, nx]:
                        continue
                    if int(label[ny, nx]) != val:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))
            if size > best:
                best = size
    return best


def _all_cc_sizes(label: np.ndarray, *, ignore_label: int | None = 0) -> list[int]:
    h, w = label.shape
    visited = np.zeros((h, w), dtype=bool)
    sizes: list[int] = []
    for y in range(h):
        for x in range(w):
            if visited[y, x]:
                continue
            val = int(label[y, x])
            if ignore_label is not None and val == ignore_label:
                visited[y, x] = True
                # flood-fill mark background so we don't revisit
                stack = [(y, x)]
                while stack:
                    cy, cx = stack.pop()
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if ny < 0 or nx < 0 or ny >= h or nx >= w:
                            continue
                        if visited[ny, nx]:
                            continue
                        if int(label[ny, nx]) != val:
                            continue
                        visited[ny, nx] = True
                        stack.append((ny, nx))
                continue
            stack = [(y, x)]
            visited[y, x] = True
            size = 0
            while stack:
                cy, cx = stack.pop()
                size += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if ny < 0 or nx < 0 or ny >= h or nx >= w:
                        continue
                    if visited[ny, nx]:
                        continue
                    if int(label[ny, nx]) != val:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))
            sizes.append(size)
    return sizes


def blob_growth_curve(
    glyphs: np.ndarray,
    fg: np.ndarray,
    bg: np.ndarray | None = None,
) -> dict[str, Any]:
    """Per-frame blob stats + linear growth slope of largest region."""
    del bg
    glyphs = np.asarray(glyphs)
    fg = np.asarray(fg)
    if glyphs.ndim == 2:
        glyphs = glyphs[None]
        fg = fg[None]
    t = int(glyphs.shape[0])
    hw = int(glyphs.shape[1] * glyphs.shape[2])
    largest: list[int] = []
    mean_region: list[float] = []
    for i in range(t):
        lab = _label_combined(glyphs[i], fg[i])
        sizes = _all_cc_sizes(lab)
        largest.append(int(max(sizes)) if sizes else 0)
        mean_region.append(float(np.mean(sizes)) if sizes else 0.0)
    xs = np.arange(t, dtype=np.float64)
    ys = np.asarray(largest, dtype=np.float64)
    slope = float(np.polyfit(xs, ys, 1)[0]) if t >= 2 else 0.0
    return {
        "per_frame_largest_region": largest,
        "per_frame_mean_region": mean_region,
        "mean_largest_region": float(ys.mean()) if t else 0.0,
        "max_largest_region": int(ys.max()) if t else 0,
        "largest_region_final": int(largest[-1]) if largest else 0,
        "largest_region_initial": int(largest[0]) if largest else 0,
        "growth_slope": slope,
        "final_frac": float(largest[-1] / hw) if t and hw else 0.0,
        "n_cells": hw,
        "n_frames": t,
    }


def time_to_attractor(
    token_change_per_step: list[float] | np.ndarray,
    blob_sizes: list[int] | np.ndarray,
    *,
    change_thresh: float = 0.01,
    blob_frac: float = 0.25,
    hw: int | None = None,
    n_cells: int | None = None,
) -> int | None:
    """First frame where change is low AND blob is large. Returns onset index or None."""
    n_cells = int(n_cells if n_cells is not None else (hw if hw is not None else 1))
    changes = np.asarray(token_change_per_step, dtype=np.float64)
    blobs = np.asarray(blob_sizes, dtype=np.float64)
    t_frames = len(blobs)
    if t_frames == 0:
        return None
    blob_thresh = blob_frac * float(max(n_cells, 1))
    for i in range(t_frames):
        if len(changes) == t_frames:
            ch = float(changes[i])
        elif len(changes) == t_frames - 1:
            if i == 0:
                continue
            ch = float(changes[i - 1])
        elif len(changes) == 0:
            ch = 1.0
        else:
            ch = float(changes[min(i, len(changes) - 1)])
        if ch <= change_thresh and float(blobs[i]) >= blob_thresh:
            return int(i)
    return None


def changed_patch_persistence(
    glyphs: np.ndarray,
    fg: np.ndarray,
    bg: np.ndarray,
    patch_h: int,
    patch_w: int,
) -> dict[str, Any]:
    """Fraction of patches that keep changing vs freeze over the rollout."""
    g = np.asarray(glyphs)
    f = np.asarray(fg)
    b = np.asarray(bg)
    t, h, w = g.shape
    assert h % patch_h == 0 and w % patch_w == 0
    ph, pw = h // patch_h, w // patch_w
    n_patches = ph * pw
    if t < 2:
        return {
            "n_patches": n_patches,
            "frac_keep_changing": 0.0,
            "frac_frozen": 1.0,
            "frac_never_changed": 1.0,
            "mean_active_frac": 0.0,
            "final_active_frac": 0.0,
            "per_step_active_frac": [],
            "mean_patch_change_rate": 0.0,
        }

    any_ch = (g[1:] != g[:-1]) | (f[1:] != f[:-1]) | (b[1:] != b[:-1])
    patch_ch = (
        any_ch.reshape(t - 1, ph, patch_h, pw, patch_w)
        .any(axis=(2, 4))
        .reshape(t - 1, n_patches)
    )
    rates = patch_ch.mean(axis=0)
    keep = rates > 0.5
    never = rates == 0.0
    half = max(1, (t - 1) // 2)
    late_frozen = patch_ch[-half:].sum(axis=0) == 0
    frozen = late_frozen | never
    per_step_active = patch_ch.mean(axis=1).astype(float).tolist()
    return {
        "n_patches": n_patches,
        "frac_keep_changing": float(keep.mean()),
        "frac_frozen": float(frozen.mean()),
        "frac_never_changed": float(never.mean()),
        "mean_active_frac": float(np.mean(per_step_active)) if per_step_active else 0.0,
        "final_active_frac": float(per_step_active[-1]) if per_step_active else 0.0,
        "per_step_active_frac": per_step_active,
        "mean_patch_change_rate": float(rates.mean()),
    }


def changed_vs_unchanged_ce(
    model,
    ctx_g: torch.Tensor,
    ctx_f: torch.Tensor,
    ctx_b: torch.Tensor,
    tgt_g: torch.Tensor,
    tgt_f: torch.Tensor,
    tgt_b: torch.Tensor,
    *,
    prev_g: torch.Tensor | None = None,
    prev_f: torch.Tensor | None = None,
    prev_b: torch.Tensor | None = None,
) -> dict[str, float]:
    """Teacher-forced CE on changed vs unchanged cells (prev = last ctx frame)."""
    if prev_g is None:
        prev_g = ctx_g[:, -1]
        prev_f = ctx_f[:, -1]
        prev_b = ctx_b[:, -1]
    assert prev_f is not None and prev_b is not None

    out = model(ctx_g, ctx_f, ctx_b)
    lg, lf, lb = out[0], out[1], out[2]
    change = (tgt_g != prev_g) | (tgt_f != prev_f) | (tgt_b != prev_b)
    unchanged = ~change

    def _ce(logits: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> float:
        if int(mask.sum().item()) == 0:
            return float("nan")
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            tgt.reshape(-1),
            reduction="none",
        )
        return float(loss[mask.reshape(-1)].mean().item())

    lc = _ce(lg, tgt_g, change) + _ce(lf, tgt_f, change) + _ce(lb, tgt_b, change)
    lu = _ce(lg, tgt_g, unchanged) + _ce(lf, tgt_f, unchanged) + _ce(lb, tgt_b, unchanged)
    return {
        "loss_changed": lc,
        "loss_unchanged": lu,
        "change_frac": float(change.float().mean().item()),
        "n_changed": int(change.sum().item()),
        "n_unchanged": int(unchanged.sum().item()),
    }


def prev_frame_copy_rollout(
    seed_glyphs: np.ndarray,
    seed_fg: np.ndarray,
    seed_bg: np.ndarray,
    *,
    steps: int,
    context_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Baseline: copy last seed/context frame for ``steps`` generated frames.

    If ``context_frames`` is set, returns seed[:context]+gen; else returns gen only.
    """
    if context_frames is not None:
        ctx_g = seed_glyphs[:context_frames]
        ctx_f = seed_fg[:context_frames]
        ctx_b = seed_bg[:context_frames]
        last_g, last_f, last_b = ctx_g[-1], ctx_f[-1], ctx_b[-1]
        gen_g = np.stack([last_g] * steps, axis=0)
        gen_f = np.stack([last_f] * steps, axis=0)
        gen_b = np.stack([last_b] * steps, axis=0)
        return (
            np.concatenate([ctx_g, gen_g], axis=0),
            np.concatenate([ctx_f, gen_f], axis=0),
            np.concatenate([ctx_b, gen_b], axis=0),
        )
    last_g, last_f, last_b = seed_glyphs[-1], seed_fg[-1], seed_bg[-1]
    return (
        np.stack([last_g] * steps, axis=0),
        np.stack([last_f] * steps, axis=0),
        np.stack([last_b] * steps, axis=0),
    )


def summarize_rollout_stability(
    glyphs: np.ndarray,
    fg: np.ndarray,
    bg: np.ndarray,
    *,
    patch_h: int = 4,
    patch_w: int = 4,
    token_change_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine blob, persistence, and attractor timing into one dict."""
    from .generate import frame_token_change_stats

    blob = blob_growth_curve(glyphs, fg, bg)
    persist = changed_patch_persistence(glyphs, fg, bg, patch_h, patch_w)
    if token_change_summary is None:
        tc = frame_token_change_stats(glyphs, fg, bg)
        token_change_summary = {
            "mean_any_change": tc["mean_any_change"],
            "per_step_any_change": [s["any_change"] for s in tc["per_step"]],
        }
    per_step = list(token_change_summary.get("per_step_any_change") or [])
    blob_sizes = blob["per_frame_largest_region"]
    onset = time_to_attractor(
        per_step,
        blob_sizes,
        n_cells=int(blob["n_cells"]),
    )
    mean_any = float(token_change_summary.get("mean_any_change") or 0.0)
    return {
        "blob": blob,
        "persistence": persist,
        "attractor": {
            "onset_frame": onset,
            "reached": onset is not None,
        },
        "token_change": {"mean_any_change": mean_any},
        "bug_blob_signature": bool(blob["growth_slope"] > 0 and mean_any < 0.02),
    }
