from typing import Optional

import torch


def sample_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    generator: Optional[torch.Generator] = None,
) -> int:
    """Sample one token id from a 1D logits tensor."""
    if temperature == 0.0:
        return int(torch.argmax(logits).item())

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        v, _ = torch.topk(logits, k=min(top_k, logits.shape[-1]))
        thresh = v[-1]
        logits = torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        cutoff = (cum > top_p).nonzero(as_tuple=True)[0]
        if cutoff.numel() > 0:
            k = int(cutoff[0].item()) + 1
            keep = sorted_idx[:k]
            mask = torch.full_like(logits, float("-inf"))
            mask[keep] = logits[keep]
            logits = mask

    probs = torch.softmax(logits, dim=-1)
    idx = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(idx.item())
