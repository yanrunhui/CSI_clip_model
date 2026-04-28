from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class TokenizedCaption:
    ids: torch.Tensor
    mask: torch.Tensor


class CaptionTokenizer:
    def __init__(self):
        self.word2id = {"<pad>": 0, "<unk>": 1}
        self.id2word = {0: "<pad>", 1: "<unk>"}
        self.next_id = 2

    @staticmethod
    def _normalize(caption: str) -> list[str]:
        return caption.lower().replace(",", " ,").replace(".", " .").split()

    def build_vocab(self, captions: Iterable[str]) -> None:
        for caption in captions:
            for word in self._normalize(caption):
                if word not in self.word2id:
                    self.word2id[word] = self.next_id
                    self.id2word[self.next_id] = word
                    self.next_id += 1

    def encode(self, caption: str, max_len: int = 48) -> TokenizedCaption:
        words = self._normalize(caption)[:max_len]
        ids = [self.word2id.get(word, 1) for word in words]
        mask = [True] * len(ids)
        ids += [0] * (max_len - len(ids))
        mask += [False] * (max_len - len(mask))
        return TokenizedCaption(
            ids=torch.tensor(ids, dtype=torch.long),
            mask=torch.tensor(mask, dtype=torch.bool),
        )
