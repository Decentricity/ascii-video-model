"""Small decoder-only Transformer for next-frame ASCII prediction."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import COLOR_VOCAB, GLYPH_VOCAB


class AsciiNextFrameModel(nn.Module):
    """Context of C frames (each N patches × K cells) → predict next frame cells.

    Sequence layout per batch item:
      for each of C frames:
        for each of N patches:
          embed mean of K cell (glyph+fg+bg) embeddings  → token
      + 1 learned [PRED] token whose output feeds per-patch prediction heads
        actually: we predict all N patches of the next frame from the C*N tokens
        via a cross-attn-free approach: append N query tokens and read them.
    """

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
    ) -> None:
        super().__init__()
        self.n_patches = n_patches
        self.cells_per_patch = cells_per_patch
        self.context_frames = context_frames
        self.d_model = d_model

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

        self.glyph_vocab = glyph_vocab
        self.color_vocab = color_vocab

    def encode_frame_patches(
        self, g: torch.Tensor, f: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        """g,f,b: B,C,N,K or B,N,K → B,*,N,D patch tokens."""
        # mean pool cell embeddings inside each patch
        e = self.glyph_emb(g) + self.fg_emb(f) + self.bg_emb(b)
        e = self.cell_proj(e)
        return e.mean(dim=-2)  # ... N D

    def forward(
        self,
        ctx_g: torch.Tensor,
        ctx_f: torch.Tensor,
        ctx_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return logits: B,N,K,Vg / Vc / Vc."""
        B, C, N, K = ctx_g.shape
        assert N == self.n_patches and K == self.cells_per_patch

        patch_tok = self.encode_frame_patches(ctx_g, ctx_f, ctx_b)  # B,C,N,D
        # flatten frames
        x = patch_tok.reshape(B, C * N, self.d_model)
        # frame ids
        frame_ids = torch.arange(C, device=x.device).view(1, C, 1).expand(B, C, N)
        frame_ids = frame_ids.reshape(B, C * N)
        pos_ids = torch.arange(C * N, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(pos_ids) + self.frame_emb(frame_ids)

        q = self.query_tokens.unsqueeze(0).expand(B, -1, -1)
        q_pos = torch.arange(C * N, C * N + N, device=x.device).unsqueeze(0).expand(B, -1)
        q = q + self.pos_emb(q_pos) + self.frame_emb.weight[C].view(1, 1, -1)

        tokens = torch.cat([x, q], dim=1)
        # causal mask over context; queries may attend to all context + themselves
        # Use full attention for POC simplicity (context is short).
        h = self.encoder(tokens)
        h = self.norm(h[:, -N:, :])  # B,N,D

        lg = self.head_g(h).view(B, N, K, self.glyph_vocab)
        lf = self.head_f(h).view(B, N, K, self.color_vocab)
        lb = self.head_b(h).view(B, N, K, self.color_vocab)
        return lg, lf, lb

    def loss(
        self,
        ctx_g: torch.Tensor,
        ctx_f: torch.Tensor,
        ctx_b: torch.Tensor,
        tgt_g: torch.Tensor,
        tgt_f: torch.Tensor,
        tgt_b: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        lg, lf, lb = self.forward(ctx_g, ctx_f, ctx_b)
        loss_g = F.cross_entropy(lg.reshape(-1, self.glyph_vocab), tgt_g.reshape(-1))
        loss_f = F.cross_entropy(lf.reshape(-1, self.color_vocab), tgt_f.reshape(-1))
        loss_b = F.cross_entropy(lb.reshape(-1, self.color_vocab), tgt_b.reshape(-1))
        total = loss_g + loss_f + loss_b
        with torch.no_grad():
            acc_g = (lg.argmax(-1) == tgt_g).float().mean()
            acc_f = (lf.argmax(-1) == tgt_f).float().mean()
            acc_b = (lb.argmax(-1) == tgt_b).float().mean()
        return {
            "loss": total,
            "loss_g": loss_g.detach(),
            "loss_f": loss_f.detach(),
            "loss_b": loss_b.detach(),
            "acc_g": acc_g,
            "acc_f": acc_f,
            "acc_b": acc_b,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model_from_config(cfg: dict, *, n_patches: int = 240) -> AsciiNextFrameModel:
    m = cfg.get("model", cfg)
    d = cfg.get("data", {})
    patch_h = int(d.get("patch_h", 4))
    patch_w = int(d.get("patch_w", 4))
    cells = patch_h * patch_w
    return AsciiNextFrameModel(
        n_patches=n_patches,
        cells_per_patch=cells,
        d_model=int(m["d_model"]),
        n_heads=int(m["n_heads"]),
        n_layers=int(m["n_layers"]),
        d_ff=int(m["d_ff"]),
        dropout=float(m.get("dropout", 0.1)),
        context_frames=int(d.get("context_frames", 8)),
    )
