"""CLI: evaluate an MHC-I head checkpoint on a held-out labels CSV."""
from __future__ import annotations

import argparse
import sys

import torch

from epitope_gen.data import AlleleEncoder, LabelledPeptideDataset
from epitope_gen.data.hla_cache import HlaEmbeddingCache
from epitope_gen.evaluate.mhc_head_eval import evaluate_mhc_head
from epitope_gen.model import Esm2Encoder, MhcHead
from epitope_gen.training import load_train_args
from epitope_gen.training.utils import load_checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--labels-csv", required=True)
    ns = parser.parse_args(argv)

    args = load_train_args(ns.config)
    esm = Esm2Encoder(model_name=args.esm2_model, device=args.device)
    if args.hla_cache_path:
        cache = HlaEmbeddingCache.load(args.hla_cache_path)
    else:
        cache = HlaEmbeddingCache.build(AlleleEncoder(), esm)
    model = MhcHead(
        esm_dim=args.esm2_dim,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
    )
    load_checkpoint(ns.checkpoint, model=model, optimizer=torch.optim.AdamW(model.parameters()))
    ds = LabelledPeptideDataset(ns.labels_csv)
    metrics = evaluate_mhc_head(
        dataset=ds, model=model, esm_encoder=esm, hla_cache=cache,
        batch_size=args.batch_size, device=args.device,
    )
    for k, v in metrics.items():
        print(f"{k}={v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
