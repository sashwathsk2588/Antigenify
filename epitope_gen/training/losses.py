"""Losses and eval metrics for the MHC-I head."""
from __future__ import annotations

import torch
from torch import nn


def pct_rank_target_transform(pct_rank: torch.Tensor) -> torch.Tensor:
    """Map %rank in [0, 100] to log10(1 + pct_rank). Monotone, dampens tail."""
    return torch.log10(1.0 + pct_rank.clamp_min(0.0))


def mhc_head_loss(
    binder_logit: torch.Tensor,
    pct_rank_pred: torch.Tensor,
    binder_target: torch.Tensor,
    pct_rank_target: torch.Tensor,
    pct_rank_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    bce = nn.functional.binary_cross_entropy_with_logits(binder_logit, binder_target)
    target_transformed = pct_rank_target_transform(pct_rank_target)
    mse = nn.functional.mse_loss(pct_rank_pred, target_transformed)
    total = bce + pct_rank_weight * mse
    return {"bce": bce, "mse": mse, "total": total}


def auroc_score(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Rank-based AUROC."""
    labels = labels.float()
    n_pos = labels.sum().item()
    n_neg = labels.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=scores.dtype)
    sum_ranks_pos = (ranks * labels).sum().item()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def auprc_score(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Average precision via trapezoidal precision-recall integration."""
    labels = labels.float()
    if labels.sum().item() == 0:
        return float("nan")
    order = torch.argsort(scores, descending=True)
    y = labels[order]
    tp = torch.cumsum(y, dim=0)
    precision = tp / torch.arange(1, y.numel() + 1, dtype=y.dtype)
    recall = tp / y.sum()
    recall = torch.cat([torch.zeros(1), recall])
    precision = torch.cat([torch.ones(1), precision])
    return torch.trapz(precision, recall).item()
