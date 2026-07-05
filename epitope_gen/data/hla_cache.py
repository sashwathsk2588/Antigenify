"""Precomputed frozen ESM-2 embeddings for HLA pseudo-sequences."""
from __future__ import annotations

from pathlib import Path

import torch

from epitope_gen.data.alleles import AlleleEncoder


class HlaEmbeddingCache:
    """Dict-like store mapping normalized allele string → (L, H) tensor."""

    def __init__(self, table: dict[str, torch.Tensor]) -> None:
        self._table = table

    @classmethod
    def build(cls, allele_enc: AlleleEncoder, esm_encoder) -> "HlaEmbeddingCache":
        alleles = allele_enc.alleles
        seqs = [allele_enc.pseudoseq(a) for a in alleles]
        out = esm_encoder.encode(seqs)
        emb = out["embeddings"]
        table = {alleles[i]: emb[i].clone() for i in range(len(alleles))}
        return cls(table)

    def save(self, path: str | Path) -> None:
        torch.save(self._table, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "HlaEmbeddingCache":
        table = torch.load(Path(path), map_location="cpu")
        return cls(table)

    @property
    def alleles(self) -> list[str]:
        return list(self._table.keys())

    def __getitem__(self, allele: str) -> torch.Tensor:
        return self._table[allele.strip().upper()]

    def batch(self, alleles: list[str]) -> torch.Tensor:
        return torch.stack([self[a] for a in alleles], dim=0)
