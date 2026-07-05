"""Engine: load config, instantiate StripedHyena, expose generate()."""
import warnings
from pathlib import Path
from typing import Optional

import torch

from backbone.logging import get_logger
from backbone.model.config import BackboneConfig
from backbone.model.generation import GenerationResult, generate
from backbone.model.model import StripedHyena
from backbone.model.tokenizer import MRNATokenizer

_log = get_logger(__name__)

_TOKENIZER_DIR = Path(__file__).resolve().parents[2] / "tokenizer" / "mrna_v1"
_DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


class BackboneEngine:
    def __init__(self, model: StripedHyena, tokenizer: MRNATokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        config_path,
        checkpoint_path: Optional[object] = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> "BackboneEngine":
        cfg = BackboneConfig.from_yaml(config_path)
        torch_dtype = _DTYPE_MAP[dtype]
        model = StripedHyena(cfg).to(device=device, dtype=torch_dtype)
        if checkpoint_path is None:
            warnings.warn(
                "model initialized with random weights — generation output is not meaningful"
            )
        else:
            state = torch.load(checkpoint_path, map_location=device)
            mismatches = _validate_state_dict(model.state_dict(), state)
            if mismatches:
                raise RuntimeError(
                    "checkpoint mismatch:\n  " + "\n  ".join(mismatches[:5])
                )
            model.load_state_dict(state, strict=True)
        tok = MRNATokenizer(_TOKENIZER_DIR / "vocab.json")
        model.eval()
        return cls(model=model, tokenizer=tok, device=device)

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop_token_ids: Optional[list[int]] = None,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if stop_token_ids is None:
            stop_token_ids = [self.tokenizer.eos_id]
        return generate(
            self.model,
            prompt_ids=ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_token_ids=stop_token_ids,
            seed=seed,
        )


def _validate_state_dict(model_sd, ckpt_sd) -> list[str]:
    out = []
    for k, v in model_sd.items():
        if k not in ckpt_sd:
            out.append(f"missing key in ckpt: {k}")
            continue
        if v.shape != ckpt_sd[k].shape:
            out.append(
                f"shape mismatch on {k}: model {tuple(v.shape)} vs ckpt {tuple(ckpt_sd[k].shape)}"
            )
    for k in ckpt_sd:
        if k not in model_sd:
            out.append(f"extra key in ckpt: {k}")
    return out
