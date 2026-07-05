"""MHC-I head training loop."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from epitope_gen.data.hla_cache import HlaEmbeddingCache
from epitope_gen.data.labels import LabelledPeptideDataset
from epitope_gen.model.mhc_head import MhcHead
from epitope_gen.training.args import MhcHeadTrainArgs
from epitope_gen.training.losses import (
    auprc_score,
    auroc_score,
    mhc_head_loss,
)
from epitope_gen.training.utils import (
    MovingAverage,
    save_checkpoint,
    seed_everything,
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


def _forward_batch(
    batch: dict[str, Any],
    *,
    model: MhcHead,
    esm_encoder,
    hla_cache: HlaEmbeddingCache,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    pep_out = esm_encoder.encode(batch["peptides"])
    pep = pep_out["embeddings"].to(device)
    mask = pep_out["mask"].to(device)
    hla = hla_cache.batch(batch["alleles"]).to(device)
    return model(pep, mask, hla)


def _evaluate(
    dataset: LabelledPeptideDataset,
    *,
    args: MhcHeadTrainArgs,
    model: MhcHead,
    esm_encoder,
    hla_cache: HlaEmbeddingCache,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=_collate, shuffle=False
    )
    all_binder_score: list[torch.Tensor] = []
    all_binder_label: list[torch.Tensor] = []
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            out = _forward_batch(
                batch, model=model, esm_encoder=esm_encoder, hla_cache=hla_cache, device=device
            )
            loss = mhc_head_loss(
                binder_logit=out["binder_logit"],
                pct_rank_pred=out["pct_rank_pred"],
                binder_target=batch["binder"].to(device),
                pct_rank_target=batch["pct_rank"].to(device),
                pct_rank_weight=args.pct_rank_weight,
            )
            total_loss += loss["total"].item()
            n_batches += 1
            all_binder_score.append(torch.sigmoid(out["binder_logit"]).cpu())
            all_binder_label.append(batch["binder"])
    scores = torch.cat(all_binder_score) if all_binder_score else torch.empty(0)
    labels = torch.cat(all_binder_label) if all_binder_label else torch.empty(0)
    model.train()
    return {
        "loss": total_loss / max(n_batches, 1),
        "auroc": auroc_score(scores, labels),
        "auprc": auprc_score(scores, labels),
    }


def train_mhc_head(
    *,
    args: MhcHeadTrainArgs,
    train_dataset: LabelledPeptideDataset,
    val_dataset: LabelledPeptideDataset,
    model: MhcHead,
    esm_encoder,
    hla_cache: HlaEmbeddingCache,
) -> dict[str, Any]:
    seed_everything(args.seed)
    device = torch.device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=_collate,
        shuffle=True,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loss_mv = MovingAverage(window=args.log_interval)
    step = 0
    last_train_loss = float("nan")

    for epoch in range(args.epochs):
        for batch in train_loader:
            out = _forward_batch(
                batch, model=model, esm_encoder=esm_encoder, hla_cache=hla_cache, device=device
            )
            loss = mhc_head_loss(
                binder_logit=out["binder_logit"],
                pct_rank_pred=out["pct_rank_pred"],
                binder_target=batch["binder"].to(device),
                pct_rank_target=batch["pct_rank"].to(device),
                pct_rank_weight=args.pct_rank_weight,
            )
            optimizer.zero_grad()
            loss["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            loss_mv.update(loss["total"].item())
            last_train_loss = loss["total"].item()
            step += 1
            if step % args.save_interval == 0:
                save_checkpoint(
                    output_dir / f"step_{step}.pt",
                    model=model, optimizer=optimizer, step=step,
                )

    save_checkpoint(
        output_dir / "final.pt",
        model=model, optimizer=optimizer, step=step,
    )
    val_metrics = _evaluate(
        val_dataset, args=args, model=model, esm_encoder=esm_encoder,
        hla_cache=hla_cache, device=device,
    )
    return {
        "final_train_loss": last_train_loss,
        "final_val_metrics": val_metrics,
        "steps": step,
    }
