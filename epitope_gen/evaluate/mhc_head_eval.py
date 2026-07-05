"""Held-out evaluation of the MHC-I head."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from epitope_gen.data.hla_cache import HlaEmbeddingCache
from epitope_gen.data.labels import LabelledPeptideDataset
from epitope_gen.model.mhc_head import MhcHead
from epitope_gen.training.losses import (
    auprc_score,
    auroc_score,
    mhc_head_loss,
)


def _collate(batch):
    return {
        "peptides": [r.peptide for r in batch],
        "alleles": [r.hla_allele for r in batch],
        "binder": torch.tensor([r.labels["binder"] for r in batch], dtype=torch.float32),
        "pct_rank": torch.tensor(
            [r.labels["pct_rank_el"] for r in batch], dtype=torch.float32
        ),
    }


def evaluate_mhc_head(
    *,
    dataset: LabelledPeptideDataset,
    model: MhcHead,
    esm_encoder,
    hla_cache: HlaEmbeddingCache,
    batch_size: int = 32,
    device: str = "cpu",
) -> dict[str, float]:
    dev = torch.device(device)
    model.to(dev)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=_collate, shuffle=False)
    total_loss = 0.0
    n_batches = 0
    all_scores: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            pep_out = esm_encoder.encode(batch["peptides"])
            pep = pep_out["embeddings"].to(dev)
            mask = pep_out["mask"].to(dev)
            hla = hla_cache.batch(batch["alleles"]).to(dev)
            out = model(pep, mask, hla)
            loss = mhc_head_loss(
                binder_logit=out["binder_logit"],
                pct_rank_pred=out["pct_rank_pred"],
                binder_target=batch["binder"].to(dev),
                pct_rank_target=batch["pct_rank"].to(dev),
            )
            total_loss += loss["total"].item()
            n_batches += 1
            all_scores.append(torch.sigmoid(out["binder_logit"]).cpu())
            all_labels.append(batch["binder"])
    scores = torch.cat(all_scores) if all_scores else torch.empty(0)
    labels = torch.cat(all_labels) if all_labels else torch.empty(0)
    return {
        "loss": total_loss / max(n_batches, 1),
        "auroc": auroc_score(scores, labels),
        "auprc": auprc_score(scores, labels),
    }
