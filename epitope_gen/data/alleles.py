"""HLA allele → NetMHCpan-style 34-residue pseudo-sequence lookup."""
from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from epitope_gen.data import constants


class UnknownAlleleError(KeyError):
    """Raised when an HLA allele is not present in the loaded table."""


def _normalize(allele: str) -> str:
    return allele.strip().upper()


class AlleleEncoder:
    """Loads a TSV of `allele<TAB>pseudoseq` rows and exposes lookup."""

    def __init__(self, table_path: str | Path | None = None) -> None:
        if table_path is None:
            table_path = files("epitope_gen.data").joinpath("_pseudoseq.tsv")
        self._table: dict[str, str] = {}
        with open(table_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                allele, seq = line.split("\t")
                if len(seq) != constants.HLA_PSEUDOSEQ_LEN:
                    raise ValueError(
                        f"Pseudo-sequence for {allele} has length {len(seq)}, "
                        f"expected {constants.HLA_PSEUDOSEQ_LEN}"
                    )
                self._table[_normalize(allele)] = seq

    @property
    def alleles(self) -> list[str]:
        return list(self._table.keys())

    def pseudoseq(self, allele: str) -> str:
        key = _normalize(allele)
        try:
            return self._table[key]
        except KeyError as e:
            raise UnknownAlleleError(allele) from e
