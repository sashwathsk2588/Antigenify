"""Variable-length peptide collator.

Pads to L_max = 11 (the longest MHC-I peptide). AA indices are offset by 1
so that pad_idx = 0 is unambiguous; downstream consumers either ignore
padded positions via `mask` or subtract 1 when mapping back to AA.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from epitope_gen.data import constants
from epitope_gen.data.dataset import PeptideRecord


@dataclass
class PeptideCollator:
    pad_idx: int = 0
    max_len: int = constants.MHC_I_MAX_LEN

    def __call__(self, batch: Sequence[PeptideRecord]) -> dict[str, object]:
        bsz = len(batch)
        aa_idx = torch.full((bsz, self.max_len), self.pad_idx, dtype=torch.long)
        mask = torch.zeros((bsz, self.max_len), dtype=torch.bool)
        for i, rec in enumerate(batch):
            for j, aa in enumerate(rec.peptide):
                aa_idx[i, j] = constants.AA_TO_IDX[aa] + 1  # +1 reserves pad_idx
            mask[i, : len(rec.peptide)] = True
        return {
            "aa_idx": aa_idx,
            "mask": mask,
            "peptides": [r.peptide for r in batch],
            "alleles": [r.hla_allele for r in batch],
            "labels": [r.labels for r in batch],
        }
