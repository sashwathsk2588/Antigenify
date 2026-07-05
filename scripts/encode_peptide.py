"""CLI: encode a peptide + its HLA pseudo-sequence with frozen ESM-2.

Phase 1 smoke script. Loads no checkpoint; uses the default ESM-2 model
unless overridden.

Example:
    python scripts/encode_peptide.py --peptide SLYNTVATL --allele HLA-A*02:01
"""
from __future__ import annotations

import argparse
import sys

from epitope_gen.data import AlleleEncoder
from epitope_gen.data import constants
from epitope_gen.model import Esm2Encoder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peptide", required=True)
    parser.add_argument("--allele", required=True)
    parser.add_argument("--esm2-model", default=constants.DEFAULT_ESM2_MODEL)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    allele_enc = AlleleEncoder()
    pseudoseq = allele_enc.pseudoseq(args.allele)

    esm = Esm2Encoder(model_name=args.esm2_model, device=args.device)
    pep_out = esm.encode([args.peptide])
    pseudo_out = esm.encode([pseudoseq])

    print(f"peptide={args.peptide}")
    print(f"allele={args.allele}")
    print(f"pseudoseq={pseudoseq}")
    print(f"peptide_embedding_shape={tuple(pep_out['embeddings'].shape)}")
    print(f"pseudoseq_embedding_shape={tuple(pseudo_out['embeddings'].shape)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
