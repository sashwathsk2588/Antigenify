import re

NUCLEOTIDES = "ACGUN"
REGION_OPEN = {"5utr": "<5UTR>", "cds": "<CDS>", "3utr": "<3UTR>"}
REGION_CLOSE = {"5utr": "</5UTR>", "cds": "</CDS>", "3utr": "</3UTR>"}
POLYA_TOKEN = "<polyA>"

_TAG_RE = re.compile(r"</?(?:5UTR|CDS|3UTR|polyA)>")


def validate_construct_string(s: str) -> None:
    """Validate a region-tagged mRNA construct string.

    Raises ValueError on out-of-alphabet characters, unbalanced tags,
    or nested regions.
    """
    region_stack: list[str] = []
    i = 0
    while i < len(s):
        m = _TAG_RE.match(s, i)
        if m:
            tag = m.group(0)
            if tag == POLYA_TOKEN:
                i = m.end()
                continue
            name = tag.strip("<>/").lower()
            if not tag.startswith("</"):
                if region_stack:
                    outer = region_stack[-1]
                    outer_close = REGION_CLOSE[outer]
                    # If the outer close tag appears later, the inner open is
                    # genuinely nested; otherwise the outer is unbalanced.
                    # Distinct messages are spec-required (see test_*_raises).
                    if outer_close in s[m.end():]:
                        raise ValueError(
                            f"nested region {tag} inside {outer} at position {i}"
                        )
                    raise ValueError(
                        f"unbalanced open tag for region {outer!r}: missing {outer_close}"
                    )
                region_stack.append(name)
            else:
                if not region_stack or region_stack[-1] != name:
                    raise ValueError(f"unbalanced close tag {tag} at position {i}")
                region_stack.pop()
            i = m.end()
            continue
        ch = s[i]
        if ch not in NUCLEOTIDES:
            raise ValueError(f"invalid char {ch!r} at position {i}")
        i += 1
    if region_stack:
        raise ValueError(f"unbalanced open tag for region {region_stack[-1]!r}")
