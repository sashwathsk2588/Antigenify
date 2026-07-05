"""ESM protein-language-model utilities.

Slim port of ProCyon's `procyon/model/esm.py` — kept just the pieces that
are useful for the epitope-generation research codebase:

- `ProteinPooler` — masked mean / max / CLS pooling with optional
  batch-key flattening (mirrors ProCyon's `ProteinPooler` verbatim, minus the
  commented-out scatter alternatives).
- `ESM_PLM_basic` — a thin fair-esm loader that returns per-token or
  mean-pooled embeddings. Requires the optional `fair-esm` package; unlike
  ProCyon we do not depend on it at import time.

Dropped from the ProCyon original:
- `ESMPrefix` and all prefix-tuning wiring — not used here.
- LoRA / QLoRA / MoLoRA / quantization / task-specific adapters —
  the paper doesn't fine-tune ESM, so this scaffolding is dead weight.
- The `EsmSelfOutputQuant / EsmAttentionQuant / EsmLayerQuant / ESMEncoderQuant`
  hierarchy — only exists to support quantized fine-tuning.
- `ESM_PLM` (the LoRA/HF hybrid variant) — the same functionality without
  fine-tuning already exists in `epitope_gen/model/esm2.py` via HuggingFace.

Use this module when you need fair-esm-style batch_converter tokenization or
the pooling utility. For plain frozen encoding of short peptides, prefer
`epitope_gen.model.esm2.Esm2Encoder` (HuggingFace-backed, no extra deps).
"""
from __future__ import annotations

import torch
from torch import nn


class ProteinPooler(nn.Module):
    """Pool residue embeddings into a per-protein representation.

    Ported from ProCyon's `procyon.model.esm.ProteinPooler`. Kept the three
    supported methods (`mean`, `max`, `cls_token`) and the batch-key flattening
    path used for split long sequences.

    Args:
        pooling_method: "mean", "max", or "cls_token".
        protein_pooling_correction_option: If True and pooling_method == "mean",
            strip the first and last token (CLS/EOS) from the sequence before
            averaging. Matches ProCyon's option of the same name.
    """

    def __init__(
        self,
        pooling_method: str = "mean",
        protein_pooling_correction_option: bool = True,
    ) -> None:
        super().__init__()
        self.pooling_method = pooling_method.lower()
        self.protein_pooling_correction_option = protein_pooling_correction_option

        if self.pooling_method == "max":
            self.pooler = lambda x: x.max(dim=-2)[0]
        elif self.pooling_method == "mean":
            if self.protein_pooling_correction_option:
                self.pooler = lambda x: x[1:-1, :].nanmean(dim=-2)
            else:
                self.pooler = lambda x: x.nanmean(dim=-2)
        elif self.pooling_method == "cls_token":
            self.pooler = lambda x: x[:, 0, :]
        else:
            raise NotImplementedError(
                f"Protein pooling method {self.pooling_method!r} is not implemented"
            )

    def forward(
        self,
        protein_embeds: torch.Tensor,
        batch_keys: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (padding_mask is not None) and (self.pooling_method == "max"):
            protein_embeds[padding_mask] = -float("inf")

        if batch_keys is not None:
            max_ind = batch_keys.max().item()
            pooled_reps: list[torch.Tensor] = []
            for i in range(max_ind + 1):
                iship = batch_keys == i
                if iship.sum() == 0:
                    continue
                common_prot = protein_embeds[batch_keys == i, :, :]
                common_prot = common_prot.reshape(1, -1, protein_embeds.shape[-1]).squeeze(0)
                if self.pooling_method == "mean" and padding_mask is not None:
                    pad_whole_seq = padding_mask[batch_keys == i].reshape(1, -1).squeeze(0)
                    common_prot = common_prot[~pad_whole_seq]
                pooled_reps.append(self.pooler(common_prot))
            return torch.stack(pooled_reps)

        return self.pooler(protein_embeds)


class ESM_PLM_basic(nn.Module):
    """Thin fair-esm loader that returns per-token or mean-pooled embeddings.

    Mirrors ProCyon's `ESM_PLM_basic` API. The `fair-esm` package is imported
    lazily so this file can be imported even if fair-esm is not installed.

    Args:
        num_params: "35m", "650m", or "3b" — the size of the ESM-2 model.
    """

    _SPECS = {
        "35m": ("esm2_t12_35M_UR50D", 12, 480),
        "650m": ("esm2_t33_650M_UR50D", 33, 1280),
        "3b": ("esm2_t36_3B_UR50D", 36, 2560),
    }

    def __init__(self, num_params: str = "35m") -> None:
        super().__init__()
        num_params = num_params.lower()
        if num_params not in self._SPECS:
            raise ValueError(
                f"Unsupported num_params {num_params!r}; expected one of "
                f"{list(self._SPECS)}"
            )
        try:
            import esm  # fair-esm
        except ImportError as exc:
            raise ImportError(
                "ESM_PLM_basic requires fair-esm. Install with `pip install fair-esm`."
            ) from exc

        loader_name, repr_layers, embedding_size = self._SPECS[num_params]
        loader = getattr(esm.pretrained, loader_name)
        self.model, self.alphabet = loader()
        self.repr_layers = repr_layers
        self.embedding_size = embedding_size
        self.batch_converter = self.alphabet.get_batch_converter()
        self.num_params = num_params

    def forward(self, batch, aggregate: bool = True) -> torch.Tensor:
        """Forward pass matching ProCyon's `ESM_PLM_basic.forward`.

        Args:
            batch: list of (label, sequence) tuples, as expected by
                fair-esm's `BatchConverter`.
            aggregate: if True, return (B, E) mean over non-pad tokens;
                otherwise (B, L, E).
        """
        _, _, batch_tokens = self.batch_converter(batch)
        batch_tokens = batch_tokens.to(next(self.model.parameters()).device)
        results = self.model(
            batch_tokens, repr_layers=[self.repr_layers], return_contacts=False
        )
        z = results["representations"][self.repr_layers]

        if aggregate:
            pad_mask = batch_tokens == self.alphabet.padding_idx
            z = z.clone()
            z[pad_mask] = torch.nan
            z = z.nanmean(dim=1)

        return z
