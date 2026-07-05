"""Labelled peptide dataset for supervised head training."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from torch.utils.data import Dataset

from epitope_gen.data import constants
from epitope_gen.data.dataset import PeptideRecord

LABEL_COLUMNS: tuple[str, ...] = ("peptide", "allele", "binder", "pct_rank_el")


class LabelledPeptideDataset(Dataset):
    """Reads a CSV with columns `peptide, allele, binder, pct_rank_el`."""

    def __init__(
        self,
        csv_path: str | Path,
        lengths: Sequence[int] = constants.MHC_I_LENGTHS,
        alleles: Sequence[str] | None = None,
    ) -> None:
        self._records: list[PeptideRecord] = []
        lengths_set = set(lengths)
        alleles_set = {a.upper() for a in alleles} if alleles else None
        with open(csv_path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            missing = [c for c in LABEL_COLUMNS if c not in reader.fieldnames]
            if missing:
                raise ValueError(f"labels CSV missing columns: {missing}")
            for row in reader:
                pep = row["peptide"].strip().upper()
                allele = row["allele"].strip().upper()
                if len(pep) not in lengths_set:
                    continue
                if alleles_set is not None and allele not in alleles_set:
                    continue
                try:
                    rec = PeptideRecord(
                        peptide=pep,
                        hla_allele=allele,
                        labels={
                            "binder": float(row["binder"]),
                            "pct_rank_el": float(row["pct_rank_el"]),
                        },
                        source="mhc_labels",
                    )
                except ValueError:
                    continue
                self._records.append(rec)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> PeptideRecord:
        return self._records[idx]
