"""CLI: train the MHC-I guidance head."""
from __future__ import annotations

import argparse
import sys

from epitope_gen.data import AlleleEncoder, LabelledPeptideDataset
from epitope_gen.data.hla_cache import HlaEmbeddingCache
from epitope_gen.model import Esm2Encoder, MhcHead
from epitope_gen.training import MhcHeadTrainArgs, load_train_args, train_mhc_head


def _build_esm_encoder(args: MhcHeadTrainArgs) -> Esm2Encoder:
    return Esm2Encoder(model_name=args.esm2_model, device=args.device)


def _build_hla_encoder(args: MhcHeadTrainArgs) -> Esm2Encoder:
    return Esm2Encoder(model_name=args.esm2_model, device=args.device)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    ns = parser.parse_args(argv)

    args = load_train_args(ns.config)

    esm = _build_esm_encoder(args)
    train_ds = LabelledPeptideDataset(args.labels_csv)
    val_ds = LabelledPeptideDataset(args.val_labels_csv)

    if args.hla_cache_path:
        hla_cache = HlaEmbeddingCache.load(args.hla_cache_path)
    else:
        hla_cache = HlaEmbeddingCache.build(AlleleEncoder(), _build_hla_encoder(args))

    model = MhcHead(
        esm_dim=args.esm2_dim,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
    )

    result = train_mhc_head(
        args=args,
        train_dataset=train_ds,
        val_dataset=val_ds,
        model=model,
        esm_encoder=esm,
        hla_cache=hla_cache,
    )
    print(f"final_train_loss={result['final_train_loss']:.4f}")
    print(f"final_val_metrics={result['final_val_metrics']}")
    print(f"steps={result['steps']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
