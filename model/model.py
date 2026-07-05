"""Top-level StripedHyena2 model and its building blocks.

Mirrors vortex/model/model.py in object structure. This file owns:
  AttentionBlock            (vortex/model/model.py:48)
  HyenaCascade              (vortex/model/model.py:125)
  ParallelGatedConvBlock    (vortex/model/model.py:401)
  get_block                 (vortex/model/model.py:578)
  StripedHyena              (vortex/model/model.py:606)

Supporting modules (RMSNorm, ParallelGatedMLP, Embedding) live in
backbone/model/layers.py.

Drops from vortex:
  - HAS_TE / fixup_te_workspace (TransformerEngine integration)
  - use_fp8_input_projections (TE-dependent)
  - flash_fft module (FlashFFTConv kernel; future spec)
  - VocabParallel embedding/unembedding (distributed)
"""
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from backbone.model.attention import MHA
from backbone.model.config import BackboneConfig
from backbone.model.layers import Embedding, ParallelGatedMLP, RMSNorm


class HyenaCascade(nn.Module):
    """SH2 convolutional multi-hybrid cascade. Pure-PyTorch B-fidelity.

    Pipeline (no cache):
      1. 3-projection depthwise short FIR on x: weight (3*D, 1, K), bias (3*D,).
      2. column-split into x1, x2, v of shape (B, D, T).
      3. Inner filter on (x2 * v). For hcs/hcm: depthwise grouped causal FIR
         with weight (G, 1, L). For hcl: implicit MLP-parametrized filter
         applied via FFT.
      4. Optional skip-connection D when inner_filter_length >= 128:
         inner_filter(x2 * v) + D * v.
      5. Gate: x1 * inner_filter_output.
      6. Output projection nn.Linear(D, D, bias=False).
    """

    def __init__(
        self,
        cfg: BackboneConfig,
        layer_idx: int,
        *,
        inner_kind: Literal["hcs", "hcm", "hcl"],
    ):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.inner_kind = inner_kind
        self.hidden_size = cfg.hidden_size
        self.num_attention_heads = cfg.num_attention_heads
        assert cfg.hidden_size % cfg.num_attention_heads == 0
        # Spec restricts cascade to vortex's column_split_hyena=True path;
        # the False legacy path is intentionally not implemented.
        assert cfg.column_split_hyena, "HyenaCascade requires column_split_hyena=True"
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads

        self.short_filter_length = cfg.short_filter_length
        self.short_filter_weight = nn.Parameter(
            torch.randn(3 * cfg.hidden_size, 1, cfg.short_filter_length) * 0.02
        )
        self.short_filter_bias = (
            nn.Parameter(torch.randn(3 * cfg.hidden_size) * 0.02)
            if cfg.short_filter_bias else None
        )

        if inner_kind in ("hcs", "hcm"):
            self._init_explicit_inner(cfg, inner_kind)
        elif inner_kind == "hcl":
            self._init_implicit_inner(cfg)
        else:
            raise ValueError(f"unknown inner_kind {inner_kind}")

        self.D = (
            nn.Parameter(torch.zeros(cfg.hidden_size))
            if self._inner_filter_length() >= 128 else None
        )

        self.out_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def _init_explicit_inner(self, cfg: BackboneConfig, kind: str) -> None:
        if kind == "hcs":
            L, G = cfg.hcs_filter_length, cfg.hcs_filter_groups
        else:
            L, G = cfg.hcm_filter_length, cfg.hcm_filter_groups
        assert cfg.hidden_size % G == 0, (
            f"hidden_size {cfg.hidden_size} must be divisible by inner_groups {G}"
        )
        self._inner_groups = G
        self._inner_filter_length_val = L
        self.h = nn.Parameter(torch.randn(G, 1, L) * 0.02)
        self.pos = None
        self.filter_mlp = None

    def _init_implicit_inner(self, cfg: BackboneConfig) -> None:
        L_pos = cfg.max_seqlen
        G = cfg.hcl_filter_groups
        order = cfg.hcl_filter_order
        assert cfg.hidden_size % G == 0, (
            f"hidden_size {cfg.hidden_size} must be divisible by hcl_filter_groups {G}"
        )
        self._inner_groups = G
        self._inner_filter_length_val = L_pos
        self.pos = nn.Parameter(torch.linspace(-1, 1, L_pos).unsqueeze(-1))
        self.filter_mlp = nn.Sequential(
            nn.Linear(1, order),
            nn.SiLU(),
            nn.Linear(order, G),
        )
        self.h = None

    def _inner_filter_length(self) -> int:
        return self._inner_filter_length_val

    def forward(self, x: torch.Tensor, cache=None) -> torch.Tensor:
        B, T, D = x.shape
        assert D == self.hidden_size

        x_in = x.transpose(1, 2)

        if cache is not None:
            short_prefix = cache.cascade_short_state(self.layer_idx)
            if short_prefix is not None:
                x_in_full = torch.cat([short_prefix.transpose(1, 2), x_in], dim=2)
            else:
                x_in_full = x_in
        else:
            x_in_full = x_in

        x_padded = F.pad(x_in_full, (self.short_filter_length - 1, 0))
        proj = F.conv1d(
            x_padded,
            self.short_filter_weight,
            bias=self.short_filter_bias,
            groups=self.hidden_size,
        )
        proj = proj[..., -T:] if cache is not None else proj

        proj_split = proj.view(B, 3, self.num_attention_heads, self.head_dim, T)
        x1 = proj_split[:, 0].reshape(B, self.hidden_size, T)
        x2 = proj_split[:, 1].reshape(B, self.hidden_size, T)
        v  = proj_split[:, 2].reshape(B, self.hidden_size, T)

        inner_input = x2 * v
        if cache is not None:
            inner_prefix = cache.cascade_inner_state(self.layer_idx)
            if inner_prefix is not None:
                inner_input_full = torch.cat([inner_prefix.transpose(1, 2), inner_input], dim=2)
            else:
                inner_input_full = inner_input
        else:
            inner_input_full = inner_input

        if self.inner_kind in ("hcs", "hcm"):
            inner_out = self._apply_explicit_inner(inner_input_full)
        else:
            inner_out = self._apply_implicit_inner(inner_input_full)

        inner_out = inner_out[..., -T:]

        if self.D is not None:
            inner_out = inner_out + self.D.view(1, -1, 1) * v

        gated = x1 * inner_out
        gated = gated.transpose(1, 2)

        if cache is not None:
            cache.set_cascade_state(
                self.layer_idx,
                short_full=x_in_full.transpose(1, 2),
                inner_full=inner_input_full.transpose(1, 2),
            )

        return self.out_proj(gated)

    def _apply_explicit_inner(self, h: torch.Tensor) -> torch.Tensor:
        D = self.hidden_size
        G = self._inner_groups
        L = self._inner_filter_length_val
        channels_per_group = D // G

        w = self.h.repeat_interleave(channels_per_group, dim=0)
        h_padded = F.pad(h, (L - 1, 0))
        return F.conv1d(h_padded, w, groups=D)

    def _apply_implicit_inner(self, h: torch.Tensor) -> torch.Tensor:
        """Implicit-filter causal conv via FFT, applied to h of shape (B, D, S)."""
        B, D, S = h.shape
        G = self._inner_groups
        channels_per_group = D // G

        k = self.filter_mlp(self.pos[:S]).transpose(0, 1)  # (G, S)
        k = k.repeat_interleave(channels_per_group, dim=0)  # (D, S)
        n = 2 * S
        H = torch.fft.rfft(h, n=n)
        K = torch.fft.rfft(k.unsqueeze(0), n=n)
        y = torch.fft.irfft(H * K, n=n)[..., :S]
        return y


class AttentionBlock(nn.Module):
    """Vortex-faithful AttentionBlock (vortex/model/model.py:48).

    Surface matches vortex: `pre_norm`/`post_norm` (not norm1/norm2),
    `inner_mha_cls` attribute holding the attention module, `counter`,
    config-driven `print_activations`/`proj_groups` knobs read via
    ``cfg.get(...)`` with sensible defaults, optional `padding_mask`
    masking in forward, and the `(u, None)` tuple return.

    Wires to our pure-PyTorch SelfAttention rather than vortex's MHA;
    proj_groups maps to num_kv_heads internally.
    """

    def __init__(self, cfg: BackboneConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = cfg
        self.pre_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.layer_idx = layer_idx
        self.print_activations = cfg.get("print_activations", False)
        self.proj_groups = cfg.get("proj_groups", cfg.num_attention_heads // cfg.num_kv_heads)
        self.num_attention_heads = cfg.num_attention_heads
        self.hidden_size = cfg.hidden_size
        self.hidden_size_per_attention_head = cfg.hidden_size // cfg.num_attention_heads

        self.counter = 0
        self.inner_mha_cls = MHA(
            embed_dim=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            num_heads_kv=cfg.num_attention_heads // self.proj_groups,
            rotary_emb_dim=cfg.hidden_size // cfg.num_attention_heads,
            qkv_proj_bias=cfg.get("qkv_proj_bias", False),
            out_proj_bias=cfg.get("mha_out_proj_bias", False),
            rotary_emb_base=cfg.rotary_emb_base,
            causal=True,
            layer_idx=layer_idx,
            use_flash_attn=cfg.get("use_flash_attn", False),
        )
        self.mlp = ParallelGatedMLP(cfg, layer_idx)

    def forward(
        self,
        u: torch.Tensor,
        cache=None,
        inference_params=None,
        padding_mask=None,
        *args,
        **kwargs,
    ):
        if isinstance(padding_mask, torch.Tensor):
            u = u * padding_mask[..., None]

        if cache is None and inference_params is not None:
            cache = inference_params

        u = self.inner_mha_cls(self.pre_norm(u), inference_params=cache) + u

        if isinstance(padding_mask, torch.Tensor):
            u = u * padding_mask[..., None]

        u = self.mlp(self.post_norm(u)) + u
        return u, None


class ParallelGatedConvBlock(nn.Module):
    """Vortex-faithful ParallelGatedConvBlock (vortex/model/model.py:401).

    Same surface treatment as AttentionBlock: `pre_norm`/`post_norm`,
    `filter` attribute holding the HyenaCascade, optional padding_mask
    masking in forward, `(u, None)` tuple return.
    """

    def __init__(
        self,
        cfg: BackboneConfig,
        layer_idx: int,
        *,
        inner_kind: Literal["hcs", "hcm", "hcl"],
    ) -> None:
        super().__init__()
        self.config = cfg
        self.pre_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.layer_idx = layer_idx
        self.print_activations = cfg.get("print_activations", False)
        self.counter = 0
        self.filter = HyenaCascade(cfg, layer_idx, inner_kind=inner_kind)
        self.mlp = ParallelGatedMLP(cfg, layer_idx)

    def forward(
        self,
        u: torch.Tensor,
        cache=None,
        inference_params=None,
        padding_mask=None,
        *args,
        **kwargs,
    ):
        if isinstance(padding_mask, torch.Tensor):
            u = u * padding_mask[..., None]

        if cache is None and inference_params is not None:
            cache = inference_params

        u = self.filter(self.pre_norm(u), cache=cache) + u

        if isinstance(padding_mask, torch.Tensor):
            u = u * padding_mask[..., None]

        u = self.mlp(self.post_norm(u)) + u
        return u, None


def get_block(cfg: BackboneConfig, layer_idx: int) -> nn.Module:
    """Mirrors vortex/model/model.py:578 — index-based dispatch."""
    kind = cfg.block_type(layer_idx)
    if kind == "attn":
        return AttentionBlock(cfg, layer_idx)
    if kind in ("hcs", "hcm", "hcl"):
        return ParallelGatedConvBlock(cfg, layer_idx, inner_kind=kind)
    raise ValueError(f"unknown block type {kind} for layer {layer_idx}")


class StripedHyena(nn.Module):
    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList(
            [get_block(cfg, layer_idx) for layer_idx in range(cfg.num_layers)]
        )
        self.final_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.unembed = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.unembed.weight = self.embed.weight

    def forward(self, input_ids: torch.Tensor, cache=None) -> torch.Tensor:
        x = self.embed(input_ids)
        for blk in self.blocks:
            x, _ = blk(x, cache=cache)
        return self.unembed(self.final_norm(x))
