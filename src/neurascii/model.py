"""Small decoder-only Transformer for next-frame ASCII prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import COLOR_VOCAB, GLYPH_VOCAB


class AsciiNextFrameModel(nn.Module):
    """Context of C frames (each N patches × K cells) → predict next frame cells."""

    def __init__(
        self,
        *,
        n_patches: int = 240,
        cells_per_patch: int = 16,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        context_frames: int = 8,
        glyph_vocab: int = GLYPH_VOCAB,
        color_vocab: int = COLOR_VOCAB,
        use_delta: bool = False,
    ) -> None:
        super().__init__()
        self.n_patches = n_patches
        self.cells_per_patch = cells_per_patch
        self.context_frames = context_frames
        self.d_model = d_model
        self.use_delta = bool(use_delta)

        self.glyph_emb = nn.Embedding(glyph_vocab, d_model)
        self.fg_emb = nn.Embedding(color_vocab, d_model)
        self.bg_emb = nn.Embedding(color_vocab, d_model)
        self.cell_proj = nn.Linear(d_model, d_model)

        max_tokens = context_frames * n_patches + n_patches
        self.pos_emb = nn.Embedding(max_tokens, d_model)
        self.frame_emb = nn.Embedding(context_frames + 1, d_model)
        self.query_tokens = nn.Parameter(torch.randn(n_patches, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        self.head_g = nn.Linear(d_model, cells_per_patch * glyph_vocab)
        self.head_f = nn.Linear(d_model, cells_per_patch * color_vocab)
        self.head_b = nn.Linear(d_model, cells_per_patch * color_vocab)
        # Always allocate change head so warm-starts with strict=False stay simple;
        # only used when use_delta=True.
        self.head_change = nn.Linear(d_model, cells_per_patch * 2)

        self.glyph_vocab = glyph_vocab
        self.color_vocab = color_vocab

    def encode_frame_patches(
        self, g: torch.Tensor, f: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        """g,f,b: B,C,N,K or B,N,K → B,*,N,D patch tokens."""
        e = self.glyph_emb(g) + self.fg_emb(f) + self.bg_emb(b)
        e = self.cell_proj(e)
        return e.mean(dim=-2)  # ... N D

    def forward(
        self,
        ctx_g: torch.Tensor,
        ctx_f: torch.Tensor,
        ctx_b: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Return logits: B,N,K,Vg / Vc / Vc [, change B,N,K,2 if use_delta]."""
        B, C, N, K = ctx_g.shape
        assert N == self.n_patches and K == self.cells_per_patch

        patch_tok = self.encode_frame_patches(ctx_g, ctx_f, ctx_b)  # B,C,N,D
        x = patch_tok.reshape(B, C * N, self.d_model)
        frame_ids = torch.arange(C, device=x.device).view(1, C, 1).expand(B, C, N)
        frame_ids = frame_ids.reshape(B, C * N)
        pos_ids = torch.arange(C * N, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(pos_ids) + self.frame_emb(frame_ids)

        q = self.query_tokens.unsqueeze(0).expand(B, -1, -1)
        q_pos = torch.arange(C * N, C * N + N, device=x.device).unsqueeze(0).expand(B, -1)
        q = q + self.pos_emb(q_pos) + self.frame_emb.weight[C].view(1, 1, -1)

        tokens = torch.cat([x, q], dim=1)
        h = self.encoder(tokens)
        h = self.norm(h[:, -N:, :])  # B,N,D

        lg = self.head_g(h).view(B, N, K, self.glyph_vocab)
        lf = self.head_f(h).view(B, N, K, self.color_vocab)
        lb = self.head_b(h).view(B, N, K, self.color_vocab)
        if self.use_delta:
            lc = self.head_change(h).view(B, N, K, 2)
            return lg, lf, lb, lc
        return lg, lf, lb

    @staticmethod
    def _weighted_ce(
        logits: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """logits (..., V), target (...), weights (...) → scalar."""
        v = logits.shape[-1]
        loss = F.cross_entropy(
            logits.reshape(-1, v),
            target.reshape(-1),
            reduction="none",
        )
        w = weights.reshape(-1).to(loss.dtype)
        return (loss * w).sum() / w.sum().clamp_min(1.0)

    def loss(
        self,
        ctx_g: torch.Tensor,
        ctx_f: torch.Tensor,
        ctx_b: torch.Tensor,
        tgt_g: torch.Tensor,
        tgt_f: torch.Tensor,
        tgt_b: torch.Tensor,
        *,
        change_weight: float = 1.0,
        use_delta: bool | None = None,
        prev_g: torch.Tensor | None = None,
        prev_f: torch.Tensor | None = None,
        prev_b: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        use_delta = self.use_delta if use_delta is None else bool(use_delta)
        if prev_g is None:
            prev_g = ctx_g[:, -1]
            prev_f = ctx_f[:, -1]
            prev_b = ctx_b[:, -1]
        assert prev_f is not None and prev_b is not None

        change_mask = (tgt_g != prev_g) | (tgt_f != prev_f) | (tgt_b != prev_b)
        w = torch.where(
            change_mask,
            torch.full((), float(change_weight), device=tgt_g.device, dtype=torch.float32),
            torch.ones((), device=tgt_g.device, dtype=torch.float32),
        )

        out = self.forward(ctx_g, ctx_f, ctx_b)
        lg, lf, lb = out[0], out[1], out[2]

        if use_delta:
            if len(out) < 4:
                raise RuntimeError("use_delta=True but forward did not return change logits")
            lc = out[3]
            change_tgt = change_mask.long()
            loss_change = F.cross_entropy(lc.reshape(-1, 2), change_tgt.reshape(-1))
            w_delta = change_mask.float()
            if float(w_delta.sum()) < 1.0:
                w_delta = torch.ones_like(w_delta) * 1e-6
            loss_g = self._weighted_ce(lg, tgt_g, w_delta)
            loss_f = self._weighted_ce(lf, tgt_f, w_delta)
            loss_b = self._weighted_ce(lb, tgt_b, w_delta)
            total = loss_change + loss_g + loss_f + loss_b
            loss_changed = (loss_g + loss_f + loss_b).detach()
            loss_unchanged = torch.zeros((), device=tgt_g.device)
        else:
            loss_g = self._weighted_ce(lg, tgt_g, w)
            loss_f = self._weighted_ce(lf, tgt_f, w)
            loss_b = self._weighted_ce(lb, tgt_b, w)
            total = loss_g + loss_f + loss_b
            loss_change = torch.zeros((), device=tgt_g.device)
            with torch.no_grad():
                ce_g = F.cross_entropy(
                    lg.reshape(-1, self.glyph_vocab), tgt_g.reshape(-1), reduction="none"
                ).view_as(tgt_g)
                ce_f = F.cross_entropy(
                    lf.reshape(-1, self.color_vocab), tgt_f.reshape(-1), reduction="none"
                ).view_as(tgt_f)
                ce_b = F.cross_entropy(
                    lb.reshape(-1, self.color_vocab), tgt_b.reshape(-1), reduction="none"
                ).view_as(tgt_b)
                cell_ce = ce_g + ce_f + ce_b
                if change_mask.any():
                    loss_changed = cell_ce[change_mask].mean()
                else:
                    loss_changed = torch.zeros((), device=tgt_g.device)
                if (~change_mask).any():
                    loss_unchanged = cell_ce[~change_mask].mean()
                else:
                    loss_unchanged = torch.zeros((), device=tgt_g.device)

        with torch.no_grad():
            acc_g = (lg.argmax(-1) == tgt_g).float().mean()
            acc_f = (lf.argmax(-1) == tgt_f).float().mean()
            acc_b = (lb.argmax(-1) == tgt_b).float().mean()
            change_frac = change_mask.float().mean()

        return {
            "loss": total,
            "loss_g": loss_g.detach(),
            "loss_f": loss_f.detach(),
            "loss_b": loss_b.detach(),
            "loss_change": loss_change.detach() if torch.is_tensor(loss_change) else loss_change,
            "loss_changed": loss_changed.detach() if torch.is_tensor(loss_changed) else loss_changed,
            "loss_unchanged": (
                loss_unchanged.detach() if torch.is_tensor(loss_unchanged) else loss_unchanged
            ),
            "change_frac": change_frac,
            "acc_g": acc_g,
            "acc_f": acc_f,
            "acc_b": acc_b,
        }

    @torch.no_grad()
    def predict_delta(
        self,
        ctx_g: torch.Tensor,
        ctx_f: torch.Tensor,
        ctx_b: torch.Tensor,
        *,
        prev_g: torch.Tensor | None = None,
        prev_f: torch.Tensor | None = None,
        prev_b: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict next frame; copy prev where change-head says unchanged."""
        if prev_g is None:
            prev_g = ctx_g[:, -1]
            prev_f = ctx_f[:, -1]
            prev_b = ctx_b[:, -1]
        assert prev_f is not None and prev_b is not None
        out = self.forward(ctx_g, ctx_f, ctx_b)
        lg, lf, lb = out[0], out[1], out[2]
        pg = lg.argmax(-1)
        pf = lf.argmax(-1)
        pb = lb.argmax(-1)
        if self.use_delta and len(out) > 3:
            change = out[3].argmax(-1).bool()  # 1 = change
            pg = torch.where(change, pg, prev_g)
            pf = torch.where(change, pf, prev_f)
            pb = torch.where(change, pb, prev_b)
        return pg, pf, pb

    @torch.no_grad()
    def predict_cells(
        self,
        ctx_g: torch.Tensor,
        ctx_f: torch.Tensor,
        ctx_b: torch.Tensor,
        *,
        prev_g: torch.Tensor | None = None,
        prev_f: torch.Tensor | None = None,
        prev_b: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Alias for predict_delta (cell-level argmax / copy-on-no-change)."""
        return self.predict_delta(
            ctx_g, ctx_f, ctx_b, prev_g=prev_g, prev_f=prev_f, prev_b=prev_b
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model_from_config(cfg: dict, *, n_patches: int = 240) -> AsciiNextFrameModel:
    m = cfg.get("model", cfg)
    d = cfg.get("data", {})
    t = cfg.get("train", {})
    patch_h = int(d.get("patch_h", 4))
    patch_w = int(d.get("patch_w", 4))
    cells = patch_h * patch_w
    use_delta = bool(t.get("use_delta", m.get("use_delta", False)))
    return AsciiNextFrameModel(
        n_patches=n_patches,
        cells_per_patch=cells,
        d_model=int(m["d_model"]),
        n_heads=int(m["n_heads"]),
        n_layers=int(m["n_layers"]),
        d_ff=int(m["d_ff"]),
        dropout=float(m.get("dropout", 0.1)),
        context_frames=int(d.get("context_frames", 8)),
        use_delta=use_delta,
    )
