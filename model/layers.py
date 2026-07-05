"""Supporting modules used by the StripedHyena2 blocks.

Mirrors the surface of vortex/model/layers.py. Holds:
  RMSNorm           (vortex/model/layers.py:178)
  ParallelGatedMLP  (vortex/model/layers.py:199)
  Embedding         (vortex/model/layers.py:249)

The architectural classes (HyenaCascade, AttentionBlock,
ParallelGatedConvBlock, StripedHyena, get_block) live in
backbone/model/model.py, matching vortex/model/model.py.
"""
import torch
import torch.nn.functional as F
from torch import nn


def grab_first_if_tuple(x):
    """Unwrap TE-style ``(tensor, bias_or_extra)`` returns.

    Our ``nn.Linear`` returns a Tensor directly, but vortex's MLP wraps each
    linear-output in this helper for compatibility with TransformerEngine
    layers that return tuples. Surface parity costs almost nothing.
    """
    if isinstance(x, tuple):
        return x[0]
    return x


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class ParallelGatedMLP(nn.Module):
    """vortex/model/layers.py:199 — pure-PyTorch B-fidelity port.

    Construct with ``(cfg, layer_idx)``. Inner width is computed as
    ``int(2 * hidden_size * 4 / 3)`` rounded up to
    ``inner_size_multiple_of * model_parallel_size`` (vortex default
    `inner_size_multiple_of=64`, `model_parallel_size=1`), then overridden
    by ``cfg.inner_mlp_size`` or ``cfg.intermediate_size`` if either is set.

    Activation comes from ``cfg.mlp_activation`` (`"gelu"` default,
    `"silu"` also supported). When ``cfg.evo2_style_activations`` is True,
    layers with ``layer_idx > 0`` replace the activation with ``Identity``
    (Evo2 production recipe).
    """

    def __init__(self, cfg, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        multiple_of = cfg.get("inner_size_multiple_of", 64)
        self.act_type = cfg.get("mlp_activation", "gelu")
        if self.act_type == "gelu":
            self.act = F.gelu
        elif self.act_type == "silu":
            self.act = F.silu
        else:
            raise NotImplementedError(f"unsupported mlp_activation {self.act_type!r}")

        if self.layer_idx > 0 and cfg.get("evo2_style_activations", False):
            self.act = nn.Identity()

        self.multiple_of = multiple_of * cfg.get("model_parallel_size", 1)

        inner_size = int(2 * cfg.hidden_size * 4 / 3)
        inner_size = self.multiple_of * (
            (inner_size + self.multiple_of - 1) // self.multiple_of
        )
        # Explicit overrides: vortex uses `inner_mlp_size`; we also honour our
        # existing `intermediate_size` field for back-compat with the YAMLs.
        inner_size = cfg.get("inner_mlp_size", cfg.get("intermediate_size", inner_size))

        self.l1 = nn.Linear(cfg.hidden_size, inner_size, bias=False)
        self.l2 = nn.Linear(cfg.hidden_size, inner_size, bias=False)
        self.l3 = nn.Linear(inner_size, cfg.hidden_size, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z1, z2 = self.l1(z), self.l2(z)
        z1, z2 = grab_first_if_tuple(z1), grab_first_if_tuple(z2)
        y = self.l3(self.act(z1) * z2)
        return grab_first_if_tuple(y)


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.weight)
