"""Attention modules — vortex/model/attention.py B-fidelity port.

Public surface:
  get_alibi_slopes          (vortex/model/attention.py:32)
  FlashSelfAttention        (vortex/model/attention.py:48)
  FlashCrossAttention       (vortex/model/attention.py:133)
  SelfAttention             (vortex/model/attention.py:230) — packed-qkv, pure-PyTorch
  CrossAttention            (vortex/model/attention.py:286) — packed q+kv, pure-PyTorch
  MHA                       (vortex/model/attention.py:381) — multi-head with rotary, GQA, cross-attn

Skipped from vortex MHA (deferred / not B-fidelity-relevant):
  - dwconv branch                              (vortex-specific 1d-conv pre-mix)
  - fused_bias_fc / FusedDense / LinearResidual (TransformerEngine wrappers)
  - mixer_subset                               (ViT-only)
  - _apply_rotary_update_kvcache_attention     (Flash + Megatron-style fast decode path)
  - rotary_emb_scale_base / interleaved        (xPos / interleaved rotary variants)
  - checkpointing                              (torch.utils.checkpoint wiring)

The MHA class accepts ``inference_params`` (vortex naming) — in this codebase
it's our :class:`backbone.model.cache.Cache`. The Megatron-style fields
(``key_value_memory_dict``, ``lengths_per_sample``, ``seqlen_offset``) are not
required; MHA uses ``cache.attn_past_len(layer_idx)`` and ``cache.append_kv``.
"""
import math
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from backbone.model.config import BackboneConfig
from backbone.model.rotary import RotaryEmbedding, apply_rotary_emb_torch

try:
    from backbone.ops.attn_interface import (
        local_flash_attn_kvpacked_func,
        local_flash_attn_qkvpacked_func,
        local_flash_attn_varlen_kvpacked_func,
        local_flash_attn_varlen_qkvpacked_func,
    )
except (ImportError, ModuleNotFoundError):
    local_flash_attn_qkvpacked_func = None
    local_flash_attn_kvpacked_func = None
    local_flash_attn_varlen_qkvpacked_func = None
    local_flash_attn_varlen_kvpacked_func = None


# ---------------------------------------------------------------------------
# ALiBi
# ---------------------------------------------------------------------------

def get_alibi_slopes(nheads: int) -> list[float]:
    """Per-head ALiBi slopes (vortex/model/attention.py:32)."""

    def get_slopes_power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(nheads).is_integer():
        return get_slopes_power_of_2(nheads)
    closest_power_of_2 = 2 ** math.floor(math.log2(nheads))
    return (
        get_slopes_power_of_2(closest_power_of_2)
        + get_alibi_slopes(2 * closest_power_of_2)[0::2][: nheads - closest_power_of_2]
    )


# ---------------------------------------------------------------------------
# FlashAttention wrappers (CUDA-only; gated by kernel availability)
# ---------------------------------------------------------------------------

class FlashSelfAttention(nn.Module):
    """FlashAttention-2 self-attention wrapper (vortex/model/attention.py:48)."""

    def __init__(
        self,
        layer_number: int,
        causal: bool = False,
        softmax_scale: float | None = None,
        attention_dropout: float = 0.0,
        window_size: tuple[int, int] = (-1, -1),
        alibi_slopes: torch.Tensor | None = None,
        deterministic: bool = False,
    ):
        super().__init__()
        assert local_flash_attn_varlen_qkvpacked_func is not None, "FlashAttention is not installed"
        assert local_flash_attn_qkvpacked_func is not None, "FlashAttention is not installed"
        self.layer_number = layer_number
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)
        self.register_buffer("alibi_slopes", alibi_slopes, persistent=False)
        self.window_size = window_size
        self.deterministic = deterministic

    def forward(self, qkv, causal=None, cu_seqlens=None, max_seqlen=None):
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda
        causal = self.causal if causal is None else causal
        unpadded = cu_seqlens is not None
        if self.alibi_slopes is not None:
            self.alibi_slopes = self.alibi_slopes.to(torch.float32)
        if unpadded:
            assert cu_seqlens.dtype == torch.int32
            assert max_seqlen is not None and isinstance(max_seqlen, int)
            return local_flash_attn_varlen_qkvpacked_func(
                qkv, cu_seqlens, max_seqlen,
                self.drop.p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal,
                alibi_slopes=self.alibi_slopes, window_size=self.window_size,
                deterministic=self.deterministic,
            )
        return local_flash_attn_qkvpacked_func(
            qkv,
            self.drop.p if self.training else 0.0,
            softmax_scale=self.softmax_scale, causal=causal,
            alibi_slopes=self.alibi_slopes, window_size=self.window_size,
            deterministic=self.deterministic,
        )


class FlashCrossAttention(nn.Module):
    """FlashAttention-2 cross-attention wrapper (vortex/model/attention.py:133)."""

    def __init__(
        self,
        causal: bool = False,
        softmax_scale: float | None = None,
        attention_dropout: float = 0.0,
        alibi_slopes: torch.Tensor | None = None,
        window_size: tuple[int, int] = (-1, -1),
        deterministic: bool = False,
    ):
        super().__init__()
        assert local_flash_attn_varlen_kvpacked_func is not None, "FlashAttention is not installed"
        assert local_flash_attn_kvpacked_func is not None, "FlashAttention is not installed"
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)
        self.register_buffer("alibi_slopes", alibi_slopes, persistent=False)
        self.window_size = window_size
        self.deterministic = deterministic

    def forward(
        self, q, kv, causal=None, cu_seqlens=None, max_seqlen=None,
        cu_seqlens_k=None, max_seqlen_k=None,
    ):
        assert q.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda and kv.is_cuda
        causal = self.causal if causal is None else causal
        unpadded = cu_seqlens is not None
        if self.alibi_slopes is not None:
            self.alibi_slopes = self.alibi_slopes.to(torch.float32)

        if unpadded:
            assert cu_seqlens.dtype == torch.int32 and isinstance(max_seqlen, int)
            assert cu_seqlens_k is not None and cu_seqlens_k.dtype == torch.int32
            assert isinstance(max_seqlen_k, int)
            return local_flash_attn_varlen_kvpacked_func(
                q, kv, cu_seqlens, cu_seqlens_k, max_seqlen, max_seqlen_k,
                self.drop.p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal,
                alibi_slopes=self.alibi_slopes, window_size=self.window_size,
                deterministic=self.deterministic,
            )
        assert kv.shape[0] == q.shape[0], "batch size mismatch between q and kv"
        assert kv.shape[4] == q.shape[3], "head dim mismatch between q and kv"
        return local_flash_attn_kvpacked_func(
            q, kv,
            self.drop.p if self.training else 0.0,
            causal=causal, softmax_scale=self.softmax_scale,
            alibi_slopes=self.alibi_slopes, window_size=self.window_size,
            deterministic=self.deterministic,
        )


# ---------------------------------------------------------------------------
# Pure-PyTorch attention cores (packed inputs) — vortex parity
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    """Packed-qkv self-attention (vortex/model/attention.py:230)."""

    def __init__(self, causal: bool = False, softmax_scale: float | None = None, attention_dropout: float = 0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    def forward(self, qkv: torch.Tensor, causal: bool | None = None, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """qkv: (B, S, 3, H, D). Returns (B, S, H, D)."""
        q, k, v = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        _, _, seqlen, d = q.shape

        scale = self.softmax_scale if self.softmax_scale is not None else 1.0 / math.sqrt(d)
        q = q * (scale * math.sqrt(d))

        attn_mask = None
        if key_padding_mask is not None:
            # (B, S) bool → (B, T=seqlen, S) additive mask: 0 for keep, -10000 for mask.
            mask = key_padding_mask[:, None, :].expand(-1, seqlen, -1)
            attn_mask = torch.where(mask, 0.0, -10000.0)

        is_causal = self.causal if causal is None else causal
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.drop.p if self.training else 0.0,
            is_causal=is_causal,
        )
        return output.permute(0, 2, 1, 3)


class CrossAttention(nn.Module):
    """Packed q + kv cross-attention (vortex/model/attention.py:286)."""

    def __init__(self, causal: bool = False, softmax_scale: float | None = None, attention_dropout: float = 0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        causal: bool | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """q: (B, Sq, H, D), kv: (B, Sk, 2, H_kv, D). Returns (B, Sq, H, D)."""
        B, Sq = q.shape[0], q.shape[1]
        Sk = kv.shape[1]
        assert kv.shape[0] == B and kv.shape[4] == q.shape[3]
        causal = self.causal if causal is None else causal

        # GQA: repeat kv heads to match q heads.
        if kv.shape[3] != q.shape[2]:
            g = q.shape[2] // kv.shape[3]
            kv = kv.repeat_interleave(g, dim=3)
        k, v = kv.unbind(dim=2)

        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)

        if key_padding_mask is not None:
            padding_mask = torch.full(
                (B, Sk), -10000.0, dtype=scores.dtype, device=scores.device,
            )
            padding_mask.masked_fill_(key_padding_mask, 0.0)
            scores = scores + padding_mask[:, None, None, :]

        if causal:
            row_idx = torch.arange(Sq, device=q.device).view(Sq, 1)
            col_idx = torch.arange(Sk, device=kv.device)
            if key_padding_mask is None:
                sk = Sk
            else:
                sk = key_padding_mask.sum(-1).view(B, 1, 1, 1)
            causal_mask = col_idx > row_idx + sk - Sq
            scores = scores.masked_fill(causal_mask, -10000.0)

        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention = self.drop(attention)
        return torch.einsum("bhts,bshd->bthd", attention, v)


# ---------------------------------------------------------------------------
# MHA — vortex/model/attention.py:381
# ---------------------------------------------------------------------------

class MHA(nn.Module):
    """Multi-head self-attention or cross-attention.

    Pruned vortex port — pure-PyTorch math when ``use_flash_attn=False``,
    delegates to FlashSelf/CrossAttention on CUDA otherwise. Cache integration
    routes through our :class:`backbone.model.cache.Cache` via the standard
    ``attn_past_len(layer_idx)`` / ``append_kv(layer_idx, k, v)`` interface
    that ``SelfAttention`` (block-level) was using.

    Not ported (see module docstring for full list): dwconv, fused_bias_fc,
    mixer_subset, _apply_rotary_update_kvcache_attention fast decode path,
    xPos / interleaved rotary, return_residual fused-grad path, checkpointing.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_heads_kv: int | None = None,
        cross_attn: bool = False,
        qkv_proj_bias: bool = False,
        out_proj_bias: bool = False,
        dropout: float = 0.0,
        softmax_scale: float | None = None,
        causal: bool = False,
        layer_idx: int | None = None,
        rotary_emb_dim: int = 0,
        rotary_emb_base: float = 10000.0,
        use_alibi: bool = False,
        window_size: tuple[int, int] = (-1, -1),
        use_flash_attn: bool = False,
        return_residual: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.cross_attn = cross_attn
        self.causal = causal
        self.layer_idx = layer_idx
        self.rotary_emb_dim = rotary_emb_dim
        self.use_flash_attn = use_flash_attn
        self.return_residual = return_residual

        if use_alibi:
            assert use_flash_attn, "ALiBi code path requires flash_attn"
            alibi_slopes = torch.tensor(get_alibi_slopes(num_heads))
        else:
            alibi_slopes = None
        if window_size != (-1, -1):
            assert use_flash_attn, "Sliding-window attention requires flash_attn"

        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
        assert num_heads % self.num_heads_kv == 0, "num_heads must be divisible by num_heads_kv"
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads

        qkv_dim = self.head_dim * (num_heads + 2 * self.num_heads_kv)
        kv_dim = 2 * self.head_dim * self.num_heads_kv

        if rotary_emb_dim > 0:
            assert not cross_attn, "MHA with rotary embedding does not support cross-attention"
            self.rotary_emb = RotaryEmbedding(rotary_emb_dim, base=rotary_emb_base)

        if not cross_attn:
            self.Wqkv = nn.Linear(embed_dim, qkv_dim, bias=qkv_proj_bias)
        else:
            self.Wq = nn.Linear(embed_dim, embed_dim, bias=qkv_proj_bias)
            self.Wkv = nn.Linear(embed_dim, kv_dim, bias=qkv_proj_bias)

        inner_attn_cls = (
            partial(FlashSelfAttention, layer_number=self.layer_idx,
                    alibi_slopes=alibi_slopes, window_size=window_size)
            if use_flash_attn else SelfAttention
        )
        inner_cross_attn_cls = (
            partial(FlashCrossAttention, alibi_slopes=alibi_slopes, window_size=window_size)
            if use_flash_attn else CrossAttention
        )
        self.inner_attn = inner_attn_cls(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout,
        )
        self.inner_cross_attn = inner_cross_attn_cls(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout,
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=out_proj_bias)

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=None):
        """Vortex-parity helper (vortex/model/attention.py:475).

        Returns a pre-allocated (B, max_seqlen, 2, H_kv, D) buffer. Our Cache
        grows on demand via ``append_kv``, so this is not strictly required —
        kept for API parity.
        """
        dtype = self.out_proj.weight.dtype if dtype is None else dtype
        device = self.out_proj.weight.device
        return torch.empty(
            batch_size, max_seqlen, 2, self.num_heads_kv, self.head_dim,
            dtype=dtype, device=device,
        )

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor, past_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary to q/k with position offset for incremental decoding."""
        T = q.shape[1]
        cos, sin = self.rotary_emb(seq_len=past_len + T, device=q.device, dtype=q.dtype)
        cos, sin = cos[past_len:], sin[past_len:]
        q = apply_rotary_emb_torch(q, cos, sin)
        k = apply_rotary_emb_torch(k, cos, sin)
        return q, k

    def forward(
        self,
        x: torch.Tensor,
        x_kv: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        mixer_subset=None,
        inference_params=None,
        **kwargs,
    ):
        """
        x: (B, S, embed_dim). x_kv: cross-attn only (B, Sk, embed_dim).
        key_padding_mask: (B, S) boolean (True=keep) — pure-PyTorch path only.
        cu_seqlens, max_seqlen: varlen-flash inputs.
        mixer_subset: not supported (vortex ViT feature).
        inference_params: optional Cache for incremental decoding.
        """
        assert mixer_subset is None, "mixer_subset not supported"
        if cu_seqlens is not None:
            assert max_seqlen is not None
            assert key_padding_mask is None
            assert self.use_flash_attn
            assert self.rotary_emb_dim == 0, "varlen flash path is incompatible with our rotary"
        if key_padding_mask is not None:
            assert cu_seqlens is None and max_seqlen is None
            assert not self.use_flash_attn, "key_padding_mask is for the pure-PyTorch path"

        B, T = x.shape[:2]
        past_len = (
            inference_params.attn_past_len(self.layer_idx) if inference_params is not None else 0
        )

        # ---- Self-attention case (square H = H_kv) ----
        if not self.cross_attn and self.num_heads_kv == self.num_heads:
            assert x_kv is None
            qkv = self.Wqkv(x)
            qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)
            if self.rotary_emb_dim > 0:
                q, k, v = qkv.unbind(dim=2)
                q, k = self._apply_rotary(q, k, past_len)
                # If we have a cache, route through kvpacked path; else stick with qkv path.
                if inference_params is not None:
                    k, v = inference_params.append_kv(self.layer_idx, k, v)
                    kv = torch.stack([k, v], dim=2)
                    context = self.inner_cross_attn(q, kv, causal=(past_len == 0))
                else:
                    qkv = torch.stack([q, k, v], dim=2)
                    if self.use_flash_attn:
                        context = self.inner_attn(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
                    else:
                        context = self.inner_attn(qkv)
            else:
                if inference_params is not None:
                    q, k, v = qkv.unbind(dim=2)
                    k, v = inference_params.append_kv(self.layer_idx, k, v)
                    kv = torch.stack([k, v], dim=2)
                    context = self.inner_cross_attn(q, kv, causal=(past_len == 0))
                else:
                    if self.use_flash_attn:
                        context = self.inner_attn(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
                    else:
                        context = self.inner_attn(qkv)
        # ---- Cross-attention or GQA case ----
        else:
            if self.cross_attn:
                q = self.Wq(x)
                kv = self.Wkv(x_kv if x_kv is not None else x)
            else:
                assert self.num_heads_kv != self.num_heads
                qkv = self.Wqkv(x)
                q_size = self.num_heads * self.head_dim
                q = qkv[..., :q_size]
                kv = qkv[..., q_size:]
            q = q.view(B, T, self.num_heads, self.head_dim)
            Sk = kv.shape[1]
            kv = kv.view(B, Sk, 2, self.num_heads_kv, self.head_dim)

            if self.rotary_emb_dim > 0:
                k = kv[:, :, 0]
                v = kv[:, :, 1]
                q, k = self._apply_rotary(q, k, past_len)
                if inference_params is not None:
                    k, v = inference_params.append_kv(self.layer_idx, k, v)
                kv = torch.stack([k, v], dim=2)
            elif inference_params is not None:
                k = kv[:, :, 0]
                v = kv[:, :, 1]
                k, v = inference_params.append_kv(self.layer_idx, k, v)
                kv = torch.stack([k, v], dim=2)

            causal = (past_len == 0) if inference_params is not None else None
            context = self.inner_cross_attn(q, kv, causal=causal)

        out = self.out_proj(context.reshape(B, T, self.num_heads * self.head_dim))
        return out if not self.return_residual else (out, x)
