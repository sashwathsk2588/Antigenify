from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class BackboneConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    intermediate_size: int
    max_seqlen: int

    attn_layer_idxs: list[int]
    hcs_layer_idxs: list[int]
    hcm_layer_idxs: list[int]
    hcl_layer_idxs: list[int]

    short_filter_length: int
    short_filter_bias: bool
    state_size: int
    num_filters: int
    column_split_hyena: bool
    inference_mode: bool

    hcs_filter_length: int
    hcs_filter_groups: int
    hcm_filter_length: int
    hcm_filter_groups: int
    hcl_filter_order: int
    hcl_filter_groups: int

    rotary_emb_base: float
    rms_norm_eps: float
    tie_word_embeddings: bool
    dtype: str

    def __post_init__(self) -> None:
        all_idxs = (
            list(self.attn_layer_idxs)
            + list(self.hcs_layer_idxs)
            + list(self.hcm_layer_idxs)
            + list(self.hcl_layer_idxs)
        )
        if len(all_idxs) != len(set(all_idxs)):
            raise ValueError(
                f"layer indices overlap across attn/hcs/hcm/hcl lists: {sorted(all_idxs)}"
            )
        if set(all_idxs) != set(range(self.num_layers)):
            missing = set(range(self.num_layers)) - set(all_idxs)
            extra = set(all_idxs) - set(range(self.num_layers))
            raise ValueError(
                f"layer-idx coverage broken: not assigned={sorted(missing)}, "
                f"out of range={sorted(extra)}"
            )
        if self.num_attention_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_kv_heads {self.num_kv_heads} must divide "
                f"num_attention_heads {self.num_attention_heads}"
            )
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size {self.hidden_size} must divide "
                f"num_attention_heads {self.num_attention_heads}"
            )
        if self.num_filters > self.hidden_size or self.hidden_size % self.num_filters != 0:
            raise ValueError(
                f"num_filters {self.num_filters} must divide hidden_size "
                f"{self.hidden_size} and be <= hidden_size"
            )
        if self.state_size <= 0:
            raise ValueError(f"state_size must be > 0, got {self.state_size}")
        for name, length in [
            ("short_filter_length", self.short_filter_length),
            ("hcs_filter_length", self.hcs_filter_length),
            ("hcm_filter_length", self.hcm_filter_length),
        ]:
            if length <= 0:
                raise ValueError(f"{name} must be > 0, got {length}")
        for name, groups in [
            ("hcs_filter_groups", self.hcs_filter_groups),
            ("hcm_filter_groups", self.hcm_filter_groups),
            ("hcl_filter_groups", self.hcl_filter_groups),
        ]:
            if groups <= 0 or self.hidden_size % groups != 0:
                raise ValueError(
                    f"{name} {groups} must be > 0 and divide hidden_size "
                    f"{self.hidden_size}"
                )

    def block_type(self, layer_idx: int) -> Literal["attn", "hcs", "hcm", "hcl"]:
        if layer_idx in self.attn_layer_idxs:
            return "attn"
        if layer_idx in self.hcs_layer_idxs:
            return "hcs"
        if layer_idx in self.hcm_layer_idxs:
            return "hcm"
        if layer_idx in self.hcl_layer_idxs:
            return "hcl"
        raise ValueError(f"layer {layer_idx} not assigned a block type")

    def to_dict(self) -> dict:
        return asdict(self)

    def get(self, key: str, default=None):
        """vortex-style optional-attribute lookup with default."""
        return getattr(self, key, default)

    @classmethod
    def from_yaml(cls, path) -> "BackboneConfig":
        data = yaml.safe_load(Path(path).read_text())
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config fields: {sorted(unknown)}")
        return cls(**data)
