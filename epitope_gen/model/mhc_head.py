"""MHC-I binding guidance head.

Cross-attention transformer: peptide ESM-2 latents are queries, HLA
pseudo-sequence ESM-2 latents are keys/values. Predicts two scalars per
peptide: `binder_logit` (BCE target) and `pct_rank_pred` (regression on
log10(1 + %rank), decoded elsewhere).
"""
from __future__ import annotations

import torch
from torch import nn


class _Block(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(model_dim)
        self.self_attn = nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_cross = nn.LayerNorm(model_dim)
        self.cross_attn = nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, model_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor,
        ctx: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm_self(x)
        attn_out, _ = self.self_attn(h, h, h, key_padding_mask=key_padding_mask)
        x = x + attn_out
        h = self.norm_cross(x)
        attn_out, _ = self.cross_attn(h, ctx, ctx)
        x = x + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x


class MhcHead(nn.Module):
    def __init__(
        self,
        esm_dim: int,
        model_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pep_proj = nn.Linear(esm_dim, model_dim)
        self.hla_proj = nn.Linear(esm_dim, model_dim)
        self.blocks = nn.ModuleList(
            [_Block(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.norm_out = nn.LayerNorm(model_dim)
        self.binder_head = nn.Linear(model_dim, 1)
        self.rank_head = nn.Linear(model_dim, 1)

    def forward(
        self,
        peptide_latents: torch.Tensor,
        peptide_mask: torch.Tensor,
        hla_latents: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x = self.pep_proj(peptide_latents)
        ctx = self.hla_proj(hla_latents)
        key_padding = ~peptide_mask
        for blk in self.blocks:
            x = blk(x, key_padding, ctx)
        x = self.norm_out(x)
        w = peptide_mask.float().unsqueeze(-1)
        pooled = (x * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
        return {
            "binder_logit": self.binder_head(pooled).squeeze(-1),
            "pct_rank_pred": self.rank_head(pooled).squeeze(-1),
        }
