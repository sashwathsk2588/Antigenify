"""Args for MHC-I head training."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from epitope_gen.data import constants


@dataclass
class MhcHeadTrainArgs:
    # Model
    esm2_model: str = constants.DEFAULT_ESM2_MODEL
    esm2_dim: int = constants.DEFAULT_ESM2_DIM
    model_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.1

    # Data
    labels_csv: str = ""
    val_labels_csv: str = ""
    hla_cache_path: str = ""

    # Optim
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 10
    warmup_steps: int = 500
    grad_clip: float = 1.0
    pct_rank_weight: float = 1.0

    # Runtime
    device: str = "cpu"
    seed: int = 42
    log_interval: int = 50
    val_interval: int = 500
    save_interval: int = 500
    output_dir: str = "runs/mhc_head"


def load_train_args(path: str | Path) -> MhcHeadTrainArgs:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    valid = {f.name for f in fields(MhcHeadTrainArgs)}
    unknown = set(cfg) - valid
    if unknown:
        raise TypeError(f"Unknown config keys: {sorted(unknown)}")
    return MhcHeadTrainArgs(**cfg)
