"""Small training utilities."""
from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MovingAverage:
    def __init__(self, window: int) -> None:
        self._buf: deque[float] = deque(maxlen=window)

    def update(self, value: float) -> None:
        self._buf.append(float(value))

    @property
    def value(self) -> float:
        if not self._buf:
            return float("nan")
        return sum(self._buf) / len(self._buf)


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    extra: dict[str, Any] | None = None,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "extra": extra or {},
        },
        Path(path),
    )


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    ckpt = torch.load(Path(path), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return {"step": ckpt["step"], "extra": ckpt.get("extra", {})}
