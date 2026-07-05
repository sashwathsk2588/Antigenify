"""Per-layer attention KV + per-cascade conv-state cache.

Trim policy in set_cascade_state:
  short: last (short_filter_length - 1) tokens
  inner (hcs/hcm): last (<category>_filter_length - 1) tokens
  inner (hcl): full prefix (unbounded implicit filter)
"""
from typing import Optional

import torch

from backbone.model.config import BackboneConfig


class Cache:
    def __init__(self, cfg: BackboneConfig):
        self.cfg = cfg
        self._kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._cascade: dict[int, dict[str, torch.Tensor]] = {}

    def attn_past_len(self, layer_idx: int) -> int:
        if layer_idx not in self._kv:
            return 0
        return self._kv[layer_idx][0].shape[1]

    def append_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        if layer_idx in self._kv:
            k0, v0 = self._kv[layer_idx]
            k = torch.cat([k0, k], dim=1)
            v = torch.cat([v0, v], dim=1)
        self._kv[layer_idx] = (k, v)
        return k, v

    def cascade_short_state(self, layer_idx: int) -> Optional[torch.Tensor]:
        slot = self._cascade.get(layer_idx)
        return None if slot is None else slot.get("short")

    def cascade_inner_state(self, layer_idx: int) -> Optional[torch.Tensor]:
        slot = self._cascade.get(layer_idx)
        return None if slot is None else slot.get("inner")

    def set_cascade_state(
        self,
        layer_idx: int,
        *,
        short_full: torch.Tensor,
        inner_full: torch.Tensor,
    ) -> None:
        kind = self.cfg.block_type(layer_idx)
        if kind == "attn":
            raise ValueError(
                f"set_cascade_state called for attn layer {layer_idx}; "
                "attention layers use append_kv instead"
            )
        short_keep = self.cfg.short_filter_length - 1
        if kind == "hcs":
            inner_keep = self.cfg.hcs_filter_length - 1
        elif kind == "hcm":
            inner_keep = self.cfg.hcm_filter_length - 1
        else:  # hcl
            inner_keep = inner_full.shape[1]
        self._cascade[layer_idx] = {
            "short": short_full[:, -short_keep:].detach() if short_keep > 0 else short_full[:, 0:0].detach(),
            "inner": inner_full[:, -inner_keep:].detach() if inner_keep > 0 else inner_full[:, 0:0].detach(),
        }
