import json
import warnings
from pathlib import Path
from typing import Iterable


class MRNATokenizer:
    """Greedy longest-match tokenizer over the mRNA-v1 vocab.

    Region tokens like ``<5UTR>`` tokenize as one ID. T is normalized to U
    by default so DNA prompts work transparently.
    """

    def __init__(self, vocab_path, *, normalize_dna: bool = True):
        self._vocab: dict[str, int] = json.loads(Path(vocab_path).read_text())
        self._inv: dict[int, str] = {v: k for k, v in self._vocab.items()}
        self._tokens_by_len: list[str] = sorted(self._vocab, key=len, reverse=True)
        self._normalize_dna = normalize_dna
        self._special = {"<pad>", "<bos>", "<eos>", "<unk>"}

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def pad_id(self) -> int:
        return self._vocab["<pad>"]

    @property
    def bos_id(self) -> int:
        return self._vocab["<bos>"]

    @property
    def eos_id(self) -> int:
        return self._vocab["<eos>"]

    @property
    def unk_id(self) -> int:
        return self._vocab["<unk>"]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids: list[int] = []
        i = 0
        saw_unk = False
        while i < len(text):
            matched = False
            for tok in self._tokens_by_len:
                if text.startswith(tok, i):
                    ids.append(self._vocab[tok])
                    i += len(tok)
                    matched = True
                    break
            if not matched:
                ch = text[i]
                if self._normalize_dna and ch in ("T", "t"):
                    ids.append(self._vocab["U"])
                else:
                    ids.append(self.unk_id)
                    saw_unk = True
                i += 1
        if saw_unk:
            warnings.warn("encountered characters outside vocab; encoded as <unk>")
        if add_special_tokens:
            ids = [self.bos_id, *ids, self.eos_id]
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = False) -> str:
        out: list[str] = []
        for i in ids:
            tok = self._inv.get(int(i), "<unk>")
            if skip_special_tokens and tok in self._special:
                continue
            out.append(tok)
        return "".join(out)
