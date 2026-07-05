"""Dataset loaders that produce `PeptideRecord` items."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from Bio import SeqIO

from torch.utils.data import Dataset

from epitope_gen.data import constants


@dataclass(frozen=True)
class PeptideRecord:
    peptide: str
    hla_allele: str
    labels: dict[str, float] = field(default_factory=dict)
    source: str = ""

    def __post_init__(self) -> None:
        if not (constants.MHC_I_MIN_LEN <= len(self.peptide) <= constants.MHC_I_MAX_LEN):
            raise ValueError(
                f"Peptide length {len(self.peptide)} outside MHC-I range "
                f"[{constants.MHC_I_MIN_LEN}, {constants.MHC_I_MAX_LEN}]"
            )
        for aa in self.peptide:
            if aa not in constants.AA_TO_IDX:
                raise ValueError(f"Peptide {self.peptide!r} contains non-canonical AA {aa!r}")


def _iedb_label(qualitative: str) -> float:
    return 0.0 if qualitative.strip().lower().startswith("negative") else 1.0


class IedbDataset(Dataset):
    """Loads peptide + allele rows from an IEDB CSV export.

    Expected columns: `Description`, `MHC Allele Name`, `Qualitative Measure`.
    """

    def __init__(
        self,
        csv_path: str | Path,
        lengths: Sequence[int] = constants.MHC_I_LENGTHS,
        alleles: Sequence[str] | None = None,
    ) -> None:
        self._records: list[PeptideRecord] = []
        lengths_set = set(lengths)
        alleles_set = set(a.upper() for a in alleles) if alleles else None
        with open(csv_path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                pep = row["Description"].strip().upper()
                allele = row["MHC Allele Name"].strip().upper()
                if len(pep) not in lengths_set:
                    continue
                if alleles_set is not None and allele not in alleles_set:
                    continue
                try:
                    rec = PeptideRecord(
                        peptide=pep,
                        hla_allele=allele,
                        labels={"binder": _iedb_label(row["Qualitative Measure"])},
                        source="iedb",
                    )
                except ValueError:
                    continue  # skip malformed rows silently
                self._records.append(rec)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> PeptideRecord:
        return self._records[idx]

    def __iter__(self) -> Iterable[PeptideRecord]:
        return iter(self._records)


class UniProtPeptideDataset(Dataset):
    """Enumerates all k-mers of the given lengths from a UniProt FASTA.

    Proteins containing any non-canonical AA are skipped entirely. There is
    no allele or label — these records exist to broaden the peptide manifold
    during CFM pretraining.
    """

    def __init__(
        self,
        fasta_path: str | Path,
        lengths: Sequence[int] = constants.MHC_I_LENGTHS,
    ) -> None:
        self._records: list[PeptideRecord] = []
        canonical = set(constants.AMINO_ACIDS)
        for record in SeqIO.parse(str(fasta_path), "fasta"):
            seq = str(record.seq).upper()
            if any(aa not in canonical for aa in seq):
                continue
            for k in lengths:
                for i in range(len(seq) - k + 1):
                    pep = seq[i : i + k]
                    self._records.append(
                        PeptideRecord(
                            peptide=pep,
                            hla_allele="",
                            labels={},
                            source="uniprot",
                        )
                    )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> PeptideRecord:
        return self._records[idx]

    def __iter__(self) -> Iterable[PeptideRecord]:
        return iter(self._records)
