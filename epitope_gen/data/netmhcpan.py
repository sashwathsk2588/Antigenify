"""Thin subprocess wrapper around the NetMHCpan-4.1 binary.

Tests mock `subprocess.run`; production callers need the binary installed and
on `$PATH` as `netMHCpan`. NetMHCpan-4.1 is DTU-licensed for academic use;
this module ships no binary.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class NetMHCpanNotFoundError(FileNotFoundError):
    """Raised when the `netMHCpan` binary is not on $PATH."""


@dataclass(frozen=True)
class NetMHCpanPrediction:
    peptide: str
    allele: str
    score_el: float
    pct_rank_el: float
    score_ba: float
    pct_rank_ba: float
    aff_nm: float
    bind_level: str  # "SB", "WB", or ""


def parse_netmhcpan_stdout(text: str) -> list[NetMHCpanPrediction]:
    preds: list[NetMHCpanPrediction] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 16:
            continue
        try:
            pred = NetMHCpanPrediction(
                peptide=parts[2],
                allele=parts[1],
                score_el=float(parts[11]),
                pct_rank_el=float(parts[12]),
                score_ba=float(parts[13]),
                pct_rank_ba=float(parts[14]),
                aff_nm=float(parts[15]),
                bind_level=parts[16].lstrip("<=") if len(parts) > 16 else "",
            )
        except (ValueError, IndexError):
            continue
        preds.append(pred)
    return preds


def run_netmhcpan(
    peptides: Iterable[str],
    allele: str,
    tmp_dir: str | Path | None = None,
    binary: str = "netMHCpan",
) -> list[NetMHCpanPrediction]:
    if shutil.which(binary) is None:
        raise NetMHCpanNotFoundError(f"{binary!r} not found on PATH")
    tmp_dir = Path(tmp_dir) if tmp_dir is not None else Path.cwd()
    pep_file = tmp_dir / "peptides.pep"
    pep_file.write_text("\n".join(peptides) + "\n", encoding="utf-8")
    result = subprocess.run(
        [binary, "-p", str(pep_file), "-a", allele, "-BA"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"netMHCpan exited with code {result.returncode}: {result.stderr!r}"
        )
    return parse_netmhcpan_stdout(result.stdout)
