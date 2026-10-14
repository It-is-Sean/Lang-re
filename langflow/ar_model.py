"""Causal Transformer used by the LangFlow paper's AR baseline.

This mirrors the causal DiT path in the Duo codebase: rotary position
embeddings, pre-normalized Transformer blocks, causal attention, and separate
input/output token embeddings.  There is deliberately no timestep condition.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x.float(), (self.dim,)).to(x.dtype) * self.weight


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return x * cos[None, :, None, :] + rotated * sin[None, :, None, :]


class CausalBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * dim, dim),
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        qkv = self.attn_qkv(self.norm1(x)).view(
            batch, length, 3, self.n_heads, dim // self.n_heads
        )
        q, k, v = qkv.unbind(dim=2)
        q = _apply_rotary(q, cos, sin).transpose(1, 2)
        k = _apply_rotary(k, cos, sin).transpose(1, 2)
        v = v.transpose(1, 2)
        attention = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=True
        ).transpose(1, 2).reshape(batch, length, dim)
        x = x + F.dropout(self.attn_out(attention), p=self.dropout, training=self.training)
        x = x + F.dropout(self.mlp(self.norm2(x)), p=self.dropout, training=self.training)
        return x


class LangFlowAR(nn.Module):
    """Autoregressive baseline with the causal Duo/LangFlow backbone."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        n_blocks: int = 12,
        n_heads: int = 12,
        dropout: float = 0.1,
        rotary_base: float = 10_000.0,
    ):
        super().__init__()
        if hidden_size % n_heads:
            raise ValueError("hidden_size must be divisible by n_heads")
        head_dim = hidden_size // n_heads
        if head_dim % 2:
            raise ValueError("attention head dimension must be even")

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.dropout = dropout
        self.vocab_embed = nn.Embedding(vocab_size, hidden_size)
        nn.init.kaiming_uniform_(self.vocab_embed.weight, a=math.sqrt(5))
        self.blocks = nn.ModuleList(
            CausalBlock(hidden_size, n_heads, dropout) for _ in range(n_blocks)
        )
        self.norm_final = LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)
        # The reference Duo DiT initializes its final projection to zero.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

        inv_freq = 1.0 / (
            rotary_base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device, dtype=self.inv_freq.dtype)
        frequencies = torch.outer(positions, self.inv_freq)
        cos = torch.cat((frequencies.cos(), frequencies.cos()), dim=-1)
        sin = torch.cat((frequencies.sin(), frequencies.sin()), dim=-1)

        x = self.vocab_embed(input_ids)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.output(self.norm_final(x))

    def config_dict(self) -> dict[str, int | float | bool]:
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "n_blocks": self.n_blocks,
            "n_heads": self.n_heads,
            "dropout": self.dropout,
            "tie_word_embeddings": False,
            "position_encoding": "rotary",
            "causal_attention": True,
        }
