"""Tests for delta/change-weighted loss shapes."""

from __future__ import annotations

import torch

from neurascii.model import AsciiNextFrameModel


def _batch(B=2, C=4, N=6, K=4, Vg=95, Vc=256):
    ctx_g = torch.randint(0, Vg, (B, C, N, K))
    ctx_f = torch.randint(0, Vc, (B, C, N, K))
    ctx_b = torch.randint(0, Vc, (B, C, N, K))
    tgt_g = torch.randint(0, Vg, (B, N, K))
    tgt_f = torch.randint(0, Vc, (B, N, K))
    tgt_b = torch.randint(0, Vc, (B, N, K))
    # ensure some unchanged cells
    tgt_g[:, 0, 0] = ctx_g[:, -1, 0, 0]
    tgt_f[:, 0, 0] = ctx_f[:, -1, 0, 0]
    tgt_b[:, 0, 0] = ctx_b[:, -1, 0, 0]
    return ctx_g, ctx_f, ctx_b, tgt_g, tgt_f, tgt_b


def test_change_weighted_loss_shapes():
    B, C, N, K = 2, 4, 6, 4
    model = AsciiNextFrameModel(
        n_patches=N,
        cells_per_patch=K,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.0,
        context_frames=C,
        use_delta=False,
    )
    ctx_g, ctx_f, ctx_b, tgt_g, tgt_f, tgt_b = _batch(B, C, N, K)
    out = model.loss(
        ctx_g,
        ctx_f,
        ctx_b,
        tgt_g,
        tgt_f,
        tgt_b,
        change_weight=5.0,
    )
    assert out["loss"].ndim == 0
    assert "loss_changed" in out
    assert "loss_unchanged" in out
    assert "change_frac" in out
    assert 0.0 <= float(out["change_frac"]) <= 1.0
    out["loss"].backward()


def test_delta_loss_and_predict_shapes():
    B, C, N, K = 2, 4, 6, 4
    model = AsciiNextFrameModel(
        n_patches=N,
        cells_per_patch=K,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.0,
        context_frames=C,
        use_delta=True,
    )
    ctx_g, ctx_f, ctx_b, tgt_g, tgt_f, tgt_b = _batch(B, C, N, K)
    fwd = model(ctx_g, ctx_f, ctx_b)
    assert len(fwd) == 4
    assert fwd[3].shape == (B, N, K, 2)
    out = model.loss(
        ctx_g,
        ctx_f,
        ctx_b,
        tgt_g,
        tgt_f,
        tgt_b,
        change_weight=5.0,
        use_delta=True,
    )
    assert out["loss"].ndim == 0
    out["loss"].backward()
    pg, pf, pb = model.predict_delta(ctx_g, ctx_f, ctx_b)
    assert pg.shape == (B, N, K)
    assert pf.shape == (B, N, K)
    assert pb.shape == (B, N, K)


def test_prev_override_change_mask():
    B, C, N, K = 1, 3, 4, 4
    model = AsciiNextFrameModel(
        n_patches=N,
        cells_per_patch=K,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
        context_frames=C,
    )
    ctx_g, ctx_f, ctx_b, tgt_g, tgt_f, tgt_b = _batch(B, C, N, K)
    # Force all-unchanged via prev == tgt
    out = model.loss(
        ctx_g,
        ctx_f,
        ctx_b,
        tgt_g,
        tgt_f,
        tgt_b,
        prev_g=tgt_g.clone(),
        prev_f=tgt_f.clone(),
        prev_b=tgt_b.clone(),
        change_weight=5.0,
    )
    assert float(out["change_frac"]) == 0.0
