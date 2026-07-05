"""Peptide encoder utilities.

Adapted from ProCyon's `procyon/model/encoder.py`. Two components:

- `TorchDrugMLP` — the general-purpose MLP block from ProCyon (which itself
  ported it from TorchDrug). Kept verbatim; useful anywhere we need a small
  configurable MLP head (e.g. projection to a shared latent, denoiser blocks).

- `PeptideEncoder` — the epitope-gen adaptation of ProCyon's `EasyESM2`.
  Rather than reloading fair-esm's `ESM2` and its checkpoint conversion, it
  wraps an already-built encoder (defaults to our HuggingFace-backed
  `epitope_gen.model.esm2.Esm2Encoder`) and adds masked mean pooling plus
  an optional MLP projection to a target output dim.

Modifications from the ProCyon source:
- The ESM backbone is injected rather than loaded internally. This lets the
  peptide-length pipeline (short 8-11 aa, no need for split-long-sequence
  handling) share the frozen encoder that the rest of `epitope_gen` uses.
- `readout` reads the mask returned by the encoder's `encode()` contract
  (`{"embeddings", "mask"}`) instead of taking a separately-passed
  residue_mask, so callers don't need to construct the mask themselves.
- Added the optional `output_dim` projection head so this class can be used
  directly as an input encoder to the flow-matching / diffusion model.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class TorchDrugMLP(nn.Module):
    """MLP with optional short-cut residual and batch-norm.

    Ported verbatim (aside from formatting) from ProCyon's
    `procyon.model.encoder.TorchDrugMLP`.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims,
        short_cut: bool = False,
        batch_norm: bool = False,
        activation="relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.dims = [input_dim] + list(hidden_dims)
        self.short_cut = short_cut

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.dropout = nn.Dropout(dropout) if dropout else None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))

        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 2):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))
        else:
            self.batch_norms = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        layer_input = input
        for i, layer in enumerate(self.layers):
            hidden = layer(layer_input)
            if i < len(self.layers) - 1:
                if self.batch_norms:
                    x = hidden.flatten(0, -2)
                    hidden = self.batch_norms[i](x).view_as(hidden)
                hidden = self.activation(hidden)
                if self.dropout:
                    hidden = self.dropout(hidden)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden
        return hidden


class PeptideEncoder(nn.Module):
    """Encode a list of peptide strings into a pooled (and optionally
    projected) vector representation.

    Structure mirrors ProCyon's `EasyESM2` — frozen ESM backbone, masked
    mean readout, optional MLP head — but the ESM backbone is injected so we
    can share the same frozen encoder used elsewhere in `epitope_gen` and
    keep the fair-esm dependency optional.

    Args:
        esm_encoder: an object exposing `.hidden_size: int` and
            `.encode(list[str]) -> {"embeddings": (B, L, H), "mask": (B, L)}`.
            The default expectation is
            `epitope_gen.model.esm2.Esm2Encoder`, but any object with the
            same contract works (e.g. a wrapped `ESM_PLM_basic`).
        output_dim: if set, apply a two-layer `TorchDrugMLP` projecting the
            pooled representation from `esm_encoder.hidden_size` to
            `output_dim`. If `None`, the pooled hidden state is returned
            directly.
        mlp_hidden_dim: hidden width of the projection MLP. Defaults to
            `esm_encoder.hidden_size`.
        activation: activation for the projection MLP.
    """

    def __init__(
        self,
        esm_encoder,
        output_dim: int | None = None,
        mlp_hidden_dim: int | None = None,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.esm_encoder = esm_encoder
        hidden = esm_encoder.hidden_size
        self.output_dim = output_dim if output_dim is not None else hidden

        if output_dim is not None:
            mid = mlp_hidden_dim if mlp_hidden_dim is not None else hidden
            self.projection = TorchDrugMLP(
                input_dim=hidden,
                hidden_dims=[mid, output_dim],
                activation=activation,
            )
        else:
            self.projection = None

    @staticmethod
    def readout(
        residue_feature: torch.Tensor,
        residue_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Masked mean pool over the sequence dimension.

        Adapted from ProCyon's `EasyESM2.readout`: instead of a broadcast
        multiplication using a (B, L) float mask, we take a boolean mask
        with the same shape, weight the summed residue features by it, and
        divide by the mask sum (clamped to at least 1 to avoid divide-by-zero
        on empty rows).
        """
        w = residue_mask.to(residue_feature.dtype).unsqueeze(-1)
        pooled = (residue_feature * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
        return pooled

    def forward(self, peptides: list[str]) -> torch.Tensor:
        out = self.esm_encoder.encode(peptides)
        pooled = self.readout(out["embeddings"], out["mask"])
        if self.projection is not None:
            pooled = self.projection(pooled)
        return pooled
