"""Frozen HuggingFace ESM-2 wrapper for peptide embedding."""
from __future__ import annotations

import torch
from torch import nn

from epitope_gen.data import constants


def _load_hf_model(model_name: str):
    from transformers import AutoModel
    return AutoModel.from_pretrained(model_name)


def _load_hf_tokenizer(model_name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name)


class Esm2Encoder(nn.Module):
    """Frozen ESM-2. Returns per-residue embeddings, padding-trimmed.

    The returned `embeddings` tensor has shape `(batch, max_pep_len, hidden)`
    where `max_pep_len` is the longest peptide in the batch — i.e. the
    HuggingFace CLS and EOS tokens are stripped. `mask` marks valid
    (non-padded) residues.
    """

    def __init__(
        self,
        model_name: str = constants.DEFAULT_ESM2_MODEL,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self._model = _load_hf_model(model_name)
        self._tokenizer = _load_hf_tokenizer(model_name)
        self._device = device
        self._model.to(device)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)
        self.hidden_size = self._model.config.hidden_size

    @torch.no_grad()
    def encode(self, peptides: list[str]) -> dict[str, torch.Tensor]:
        if not peptides:
            raise ValueError("encode() requires at least one peptide")
        toks = self._tokenizer(peptides, return_tensors="pt", padding=True)
        input_ids = toks["input_ids"].to(self._device)
        attn = toks["attention_mask"].to(self._device)
        out = self._model(input_ids=input_ids, attention_mask=attn)
        hidden = out.last_hidden_state  # (B, S, H) — includes CLS + EOS

        max_pep_len = max(len(p) for p in peptides)
        emb = torch.zeros(
            (len(peptides), max_pep_len, self.hidden_size),
            dtype=hidden.dtype,
            device=self._device,
        )
        mask = torch.zeros((len(peptides), max_pep_len), dtype=torch.bool, device=self._device)
        for i, pep in enumerate(peptides):
            emb[i, : len(pep)] = hidden[i, 1 : 1 + len(pep)]
            mask[i, : len(pep)] = True
        return {"embeddings": emb, "mask": mask}
