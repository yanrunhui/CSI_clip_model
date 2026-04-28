from __future__ import annotations

import torch
from torch import nn


class PhysicsTextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 300,
        max_len: int = 48,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        d_clip: int = 256,
        token_dropout: float = 0.1,
    ):
        super().__init__()
        self.token_dropout = token_dropout
        self.unk_id = 1
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_clip)

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.training and self.token_dropout > 0:
            drop_mask = (torch.rand_like(token_ids.float()) < self.token_dropout) & mask
            token_ids = token_ids.clone()
            token_ids[drop_mask] = self.unk_id
        B, L = token_ids.shape
        pos = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)
        x = self.tok_embed(token_ids) + self.pos_embed(pos)
        x = self.transformer(x, src_key_padding_mask=~mask)
        x = self.norm(x)
        x = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.proj(x)
