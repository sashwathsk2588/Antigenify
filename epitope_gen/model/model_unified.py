"""Unified epitope-gen model.

Adapted from ProCyon's `procyon/model/model_unified.py`. The ProCyon original
composes a LLaMA text encoder, ESM-2 protein encoder, protein-structure and
drug embeddings, retrieval / QA / captioning heads, InfoNCE contrastive
losses, LoRA/QLoRA fine-tuning, and DeepSpeed integration into a single
`UnifiedProCyon` module (~1500 LOC). Very little of that maps onto this
research codebase.

This module keeps ProCyon's *pattern* — one config-driven `nn.Module` that
composes sub-modules and exposes a small unified API — but scopes it to the
epitope-gen components that exist right now:

- `Esm2Encoder` (frozen HuggingFace ESM-2, from Phase 1)
- `HlaEmbeddingCache` (frozen pseudo-seq embeddings, from Phase 2a)
- `MhcHead` (cross-attention MHC-I binder / %rank predictor, Phase 2a)

Kept from ProCyon's design:
- Dataclass config separated from training args, so a checkpoint uniquely
  identifies its own architecture.
- `save_pretrained` / `from_pretrained` — JSON config next to a `.pt` weights
  file, mirroring ProCyon's `SAVE_CONFIG_FNAME` + `SAVE_TRAINING_STATE_FNAME`
  split (though we use `weights.pt` instead of `training_state.pt` since we
  do not persist optimizer state here).
- The ESM encoder is *frozen* by default and its weights are excluded from
  saving/loading (ProCyon does the same via `freeze_protein_encoder="all"`).
- Injection: the ESM encoder can be passed in explicitly so tests do not
  need HuggingFace network access.

Dropped from the ProCyon original:
- LLaMA text encoder + tokenizer wiring
- Retrieval / QA / captioning branches and their loss heads
- InfoNCE / MaxMargin contrastive losses
- DeepSpeed / model splitting / all-gather across ranks
- LoRA / QLoRA / prefix tuning
- Protein-structure and drug-structure embeddings
- Task-specific LoRA groups and MoE-LoRA

Extension points marked with `# extend:` — Phase 2b heads (immunogenicity,
cleavage, self-similarity) and the Phase 3 velocity field slot into those
comments without changing the public API.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import nn

from epitope_gen.data import AlleleEncoder, constants
from epitope_gen.data.hla_cache import HlaEmbeddingCache
from epitope_gen.model.esm2 import Esm2Encoder
from epitope_gen.model.mhc_head import MhcHead
from epitope_gen.training.losses import mhc_head_loss


CONFIG_FNAME = "epitope_gen_config.json"
WEIGHTS_FNAME = "weights.pt"


@dataclass
class UnifiedEpitopeGenConfig:
    """Architecture config for `UnifiedEpitopeGen`.

    Separated from `MhcHeadTrainArgs` (which carries optimizer / data / runtime
    fields) so a checkpoint uniquely identifies its own architecture.
    """

    esm2_model: str = constants.DEFAULT_ESM2_MODEL
    esm2_dim: int = constants.DEFAULT_ESM2_DIM

    # MHC head
    mhc_model_dim: int = 256
    mhc_num_heads: int = 4
    mhc_num_layers: int = 4
    mhc_ffn_dim: int = 1024
    mhc_dropout: float = 0.1

    # Data / allele table. Empty string → bundled default in `AlleleEncoder`.
    pseudoseq_table: str = ""

    # Runtime device for freshly built modules. Not persisted / not part of arch.
    device: str = "cpu"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "UnifiedEpitopeGenConfig":
        valid = {f.name for f in fields(cls)}
        unknown = set(cfg) - valid
        if unknown:
            raise TypeError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**cfg)


class UnifiedEpitopeGen(nn.Module):
    """Compose the ESM-2 peptide encoder, the HLA embedding cache, and the
    MHC-I guidance head into a single module with a small unified API.

    `esm_encoder` and `allele_encoder` can be injected (both default to the
    ones described by the config). Injection is what makes the class testable
    without HuggingFace network access.
    """

    def __init__(
        self,
        config: UnifiedEpitopeGenConfig,
        *,
        esm_encoder: Esm2Encoder | None = None,
        allele_encoder: AlleleEncoder | None = None,
        hla_cache: HlaEmbeddingCache | None = None,
    ) -> None:
        super().__init__()
        self.config = config

        self.esm_encoder = esm_encoder if esm_encoder is not None else Esm2Encoder(
            model_name=config.esm2_model,
            device=config.device,
        )
        self.allele_encoder = allele_encoder or AlleleEncoder(
            table_path=config.pseudoseq_table or None
        )
        self.hla_cache = hla_cache or HlaEmbeddingCache.build(
            self.allele_encoder, self.esm_encoder
        )

        self.mhc_head = MhcHead(
            esm_dim=config.esm2_dim,
            model_dim=config.mhc_model_dim,
            num_heads=config.mhc_num_heads,
            num_layers=config.mhc_num_layers,
            ffn_dim=config.mhc_ffn_dim,
            dropout=config.mhc_dropout,
        )
        self.mhc_head.to(config.device)

        # extend: immuno_head / cleavage_head / self_sim_head go here in Phase 2b
        # extend: velocity_field (DiT) goes here in Phase 3

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def score_mhc(
        self, peptides: list[str], alleles: list[str]
    ) -> dict[str, torch.Tensor]:
        """Score a batch of (peptide, allele) pairs with the MHC-I head.

        Returns:
            {"binder_logit": (B,), "pct_rank_pred": (B,)}
        """
        if len(peptides) != len(alleles):
            raise ValueError(
                f"peptides and alleles must have matching length "
                f"(got {len(peptides)} vs {len(alleles)})"
            )
        pep_out = self.esm_encoder.encode(peptides)
        pep = pep_out["embeddings"].to(self.config.device)
        mask = pep_out["mask"].to(self.config.device)
        hla = self.hla_cache.batch(alleles).to(self.config.device)
        return self.mhc_head(pep, mask, hla)

    def compute_mhc_loss(
        self,
        peptides: list[str],
        alleles: list[str],
        binder_target: torch.Tensor,
        pct_rank_target: torch.Tensor,
        pct_rank_weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Full training step for the MHC head: encode → forward → loss.

        Mirrors the loss computation inside `train_mhc_head` but keeps the
        model composition inside this class so a caller can plug it in without
        reaching for the private collate helpers.
        """
        out = self.score_mhc(peptides, alleles)
        loss = mhc_head_loss(
            binder_logit=out["binder_logit"],
            pct_rank_pred=out["pct_rank_pred"],
            binder_target=binder_target.to(self.config.device),
            pct_rank_target=pct_rank_target.to(self.config.device),
            pct_rank_weight=pct_rank_weight,
        )
        return {**loss, **out}

    def forward(
        self,
        peptides: list[str],
        alleles: list[str],
        *,
        task: str = "score_mhc",
        **task_kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Task-dispatched forward.

        `task="score_mhc"` runs `score_mhc(peptides, alleles)`.
        `task="mhc_loss"` runs `compute_mhc_loss(peptides, alleles, **kwargs)`.
        Future phases add new task strings here; existing task strings never
        change semantics.
        """
        if task == "score_mhc":
            return self.score_mhc(peptides, alleles)
        if task == "mhc_loss":
            return self.compute_mhc_loss(peptides, alleles, **task_kwargs)
        raise ValueError(f"Unknown task {task!r}")

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """State dict minus the frozen ESM-2 weights.

        ProCyon similarly excludes frozen backbones from its saved training
        state. We do it manually because the ESM-2 wrapper still lives inside
        this module tree.
        """
        prefix = "esm_encoder."
        return {
            k: v for k, v in self.state_dict().items() if not k.startswith(prefix)
        }

    def save_pretrained(self, path: str | Path) -> None:
        """Persist config + trainable weights under `path/`.

        Layout mirrors ProCyon's `SAVE_CONFIG_FNAME` + weights split:
            <path>/epitope_gen_config.json
            <path>/weights.pt
        """
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / CONFIG_FNAME, "w", encoding="utf-8") as fh:
            json.dump(self.config.to_dict(), fh, indent=2)
        torch.save(self._trainable_state_dict(), directory / WEIGHTS_FNAME)

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        esm_encoder: Esm2Encoder | None = None,
        allele_encoder: AlleleEncoder | None = None,
        hla_cache: HlaEmbeddingCache | None = None,
    ) -> "UnifiedEpitopeGen":
        """Load config + trainable weights from `path/`.

        The frozen ESM encoder must be provided (or is built from the saved
        config), since its weights are not persisted.
        """
        directory = Path(path)
        with open(directory / CONFIG_FNAME, "r", encoding="utf-8") as fh:
            cfg = UnifiedEpitopeGenConfig.from_dict(json.load(fh))
        model = cls(
            cfg,
            esm_encoder=esm_encoder,
            allele_encoder=allele_encoder,
            hla_cache=hla_cache,
        )
        state = torch.load(directory / WEIGHTS_FNAME, map_location=cfg.device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        # Only the frozen ESM keys are permitted to be missing.
        missing = [k for k in missing if not k.startswith("esm_encoder.")]
        if missing:
            raise RuntimeError(f"Missing keys when loading checkpoint: {missing}")
        if unexpected:
            raise RuntimeError(f"Unexpected keys when loading checkpoint: {unexpected}")
        return model
