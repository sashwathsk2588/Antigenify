"""Static constants shared across epitope_gen.

Lengths follow MHC class I conventions (8-11 aa).
Pseudo-sequence length matches NetMHCpan's 34-residue contact pseudo.
"""

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
IDX_TO_AA = {i: aa for i, aa in enumerate(AMINO_ACIDS)}

MHC_I_MIN_LEN = 8
MHC_I_MAX_LEN = 11
MHC_I_LENGTHS = (8, 9, 10, 11)

HLA_PSEUDOSEQ_LEN = 34

DEFAULT_ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"
DEFAULT_ESM2_DIM = 1280
