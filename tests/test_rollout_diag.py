"""Tests for rollout_diag helpers."""

from __future__ import annotations

import numpy as np

from neurascii.rollout_diag import (
    blob_growth_curve,
    changed_patch_persistence,
    largest_connected_region_sizes,
    prev_frame_copy_rollout,
    time_to_attractor,
)


def test_largest_connected_region_sizes_shape():
    frames = np.zeros((3, 8, 8), dtype=np.int32)
    frames[0, 0:4, 0:4] = 1
    frames[1, 0:6, 0:6] = 1
    frames[2, :, :] = 1
    sizes = largest_connected_region_sizes(frames)
    assert len(sizes) == 3
    assert sizes[0] == 16
    assert sizes[1] == 36
    assert sizes[2] == 64


def test_blob_growth_curve_keys():
    T, H, W = 5, 12, 16
    g = np.random.randint(0, 10, (T, H, W), dtype=np.int32)
    f = np.random.randint(0, 10, (T, H, W), dtype=np.int32)
    b = np.zeros((T, H, W), dtype=np.int32)
    curve = blob_growth_curve(g, f, b)
    assert "per_frame_largest_region" in curve
    assert "per_frame_mean_region" in curve
    assert "growth_slope" in curve
    assert len(curve["per_frame_largest_region"]) == T


def test_time_to_attractor():
    hw = 100
    changes = [0.2, 0.15, 0.02, 0.005, 0.001]
    blobs = [10, 20, 30, 40, 50]  # frame-aligned with changes (len equal)
    onset = time_to_attractor(changes, blobs, change_thresh=0.01, blob_frac=0.25, hw=hw)
    # frame 3: change 0.005 <= 0.01 and blob 40 >= 25
    assert onset == 3
    assert time_to_attractor([0.5] * 5, [10] * 5, hw=hw) is None


def test_changed_patch_persistence():
    T, H, W = 6, 8, 8
    g = np.zeros((T, H, W), dtype=np.int32)
    f = np.zeros((T, H, W), dtype=np.int32)
    b = np.zeros((T, H, W), dtype=np.int32)
    # animate one patch corner every frame
    for t in range(T):
        g[t, 0, 0] = t
    out = changed_patch_persistence(g, f, b, 4, 4)
    assert out["n_patches"] == 4
    assert 0.0 <= out["frac_keep_changing"] <= 1.0
    assert 0.0 <= out["frac_frozen"] <= 1.0


def test_prev_frame_copy_zero_change():
    seed_g = np.arange(2 * 4 * 4, dtype=np.int32).reshape(2, 4, 4)
    seed_f = np.zeros((2, 4, 4), dtype=np.int32)
    seed_b = np.zeros((2, 4, 4), dtype=np.int32)
    gg, ff, bb = prev_frame_copy_rollout(seed_g, seed_f, seed_b, steps=5)
    assert gg.shape == (5, 4, 4)
    assert np.all(gg == seed_g[-1])
