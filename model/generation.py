from dataclasses import dataclass
from typing import Literal, Optional

import torch

from backbone.model.cache import Cache
from backbone.model.model import StripedHyena
from backbone.model.sample import sample_logits


@dataclass
class GenerationResult:
    token_ids: list[int]
    prompt_len: int
    stop_reason: Literal["eos", "max_tokens", "stop_token"]


@torch.no_grad()
def generate(
    model: StripedHyena,
    *,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    stop_token_ids: list[int],
    seed: Optional[int] = None,
) -> GenerationResult:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")
    if len(prompt_ids) > model.cfg.max_seqlen:
        raise ValueError(
            f"prompt length {len(prompt_ids)} exceeds max_seqlen {model.cfg.max_seqlen}"
        )
    device = next(model.parameters()).device
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    cache = Cache(model.cfg)
    ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    logits = model(ids, cache=cache)
    out = list(prompt_ids)
    stop_reason: Literal["eos", "max_tokens", "stop_token"] = "max_tokens"
    stop_set = set(stop_token_ids)
    for _ in range(max_new_tokens):
        nxt = sample_logits(
            logits[0, -1].cpu().float(),
            temperature=temperature, top_k=top_k, top_p=top_p, generator=gen,
        )
        out.append(nxt)
        if nxt in stop_set:
            stop_reason = "stop_token"
            break
        logits = model(
            torch.tensor([[nxt]], device=device, dtype=torch.long),
            cache=cache,
        )
    return GenerationResult(token_ids=out, prompt_len=len(prompt_ids), stop_reason=stop_reason)
