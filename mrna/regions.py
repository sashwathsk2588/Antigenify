import re
from dataclasses import dataclass
from typing import Optional

from backbone.mrna.alphabet import (
    POLYA_TOKEN,
    REGION_CLOSE,
    REGION_OPEN,
    validate_construct_string,
)


@dataclass
class Construct:
    five_utr: Optional[str] = None
    cds: Optional[str] = None
    three_utr: Optional[str] = None
    polyA_len: Optional[int] = None


def build_prompt(
    *,
    five_utr: Optional[str] = None,
    cds: Optional[str] = None,
    three_utr: Optional[str] = None,
    include_polyA: bool = False,
) -> str:
    """Assemble a region-tagged mRNA prompt.

    A region passed as ``None`` is omitted entirely. An empty string
    ``cds=""`` is a special prompt-mode marker: it emits an opening
    ``<CDS>`` with no closing tag, so the model continues generation
    inside the CDS region. This asymmetry is intentional and only
    applies to ``cds``.
    """
    if include_polyA and five_utr is None and cds is None and three_utr is None:
        raise ValueError("include_polyA=True requires at least one region")
    parts: list[str] = []
    if five_utr is not None:
        parts.append(f"{REGION_OPEN['5utr']}{five_utr}{REGION_CLOSE['5utr']}")
    if cds is not None:
        parts.append(f"{REGION_OPEN['cds']}{cds}")
        if cds != "":
            parts[-1] += REGION_CLOSE['cds']
    if three_utr is not None:
        parts.append(f"{REGION_OPEN['3utr']}{three_utr}{REGION_CLOSE['3utr']}")
    if include_polyA:
        parts.append(POLYA_TOKEN)
    return "".join(parts)


_REGION_RE = re.compile(
    r"<5UTR>(?P<u5>[ACGUN]*)</5UTR>"
    r"|<CDS>(?P<cds>[ACGUN]*)</CDS>"
    r"|<3UTR>(?P<u3>[ACGUN]*)</3UTR>"
    r"|<polyA>(?P<pa>[ACGUN]*)"
)


def parse_construct(s: str) -> Construct:
    """Parse a region-tagged construct string into a ``Construct``.

    Duplicate regions are tolerated; the last occurrence wins. Region
    order is not enforced — the upstream validator only balances tags.
    """
    validate_construct_string(s)
    out = Construct()
    pos = 0
    for m in _REGION_RE.finditer(s):
        if m.start() != pos:
            raise ValueError(f"unexpected content at position {pos}")
        if m.group("u5") is not None:
            out.five_utr = m.group("u5")
        elif m.group("cds") is not None:
            out.cds = m.group("cds")
        elif m.group("u3") is not None:
            out.three_utr = m.group("u3")
        elif m.group("pa") is not None:
            out.polyA_len = len(m.group("pa"))
        pos = m.end()
    if pos != len(s):
        raise ValueError(f"trailing content at position {pos}")
    return out
