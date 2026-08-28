"""Maximum practical GPT architecture for the single-novel v4 experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class GPTConfig:
    """Versioned model parameters that must be frozen before cloud training."""

    vocab_size: int
    block_size: int = 1024
    embedding_size: int = 256
    num_layers: int = 8
    num_heads: int = 8
    ffn_multiplier: int = 4
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initialization_std: float = 0.02
    tie_embeddings: bool = True

    def validate(self) -> None:
        integer_fields = {
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "embedding_size": self.embedding_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "ffn_multiplier": self.ffn_multiplier,
        }
        invalid = [name for name, value in integer_fields.items() if value <= 0]
        if invalid:
            raise ValueError(f"positive integer configuration required: {invalid}")
        if self.embedding_size % self.num_heads != 0:
            raise ValueError("embedding_size must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.layer_norm_epsilon <= 0.0:
            raise ValueError("layer_norm_epsilon must be positive")
        if self.initialization_std <= 0.0:
            raise ValueError("initialization_std must be positive")

    @property
    def head_size(self) -> int:
        return self.embedding_size // self.num_heads

    @property
    def ffn_size(self) -> int:
        return self.embedding_size * self.ffn_multiplier

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CausalSelfAttention(torch.nn.Module):
    """Multi-head causal attention with a fused QKV projection."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_size = config.head_size
        self.dropout = config.dropout
        self.qkv = torch.nn.Linear(
            config.embedding_size,
            3 * config.embedding_size,
            bias=False,
        )
        self.output = torch.nn.Linear(config.embedding_size, config.embedding_size)
        self.residual_dropout = torch.nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, embedding_size = x.shape
        query, key, value = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch_size,
                sequence_length,
                self.num_heads,
                self.head_size,
            ).transpose(1, 2)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)
        attended = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            embedding_size,
        )
        return self.residual_dropout(self.output(attended))


class FeedForward(torch.nn.Module):
    """Four-times expansion MLP used inside each Transformer block."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(config.embedding_size, config.ffn_size),
            torch.nn.GELU(),
            torch.nn.Linear(config.ffn_size, config.embedding_size),
            torch.nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TransformerBlock(torch.nn.Module):
    """Pre-normalized attention and feed-forward residual block."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attention_norm = torch.nn.LayerNorm(
            config.embedding_size,
            eps=config.layer_norm_epsilon,
        )
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = torch.nn.LayerNorm(
            config.embedding_size,
            eps=config.layer_norm_epsilon,
        )
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.feed_forward(self.feed_forward_norm(x))


class GPTLanguageModelV4(torch.nn.Module):
    """Decoder-only GPT with optional tied input and output token weights."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = torch.nn.Embedding(
            config.vocab_size,
            config.embedding_size,
        )
        self.position_embedding = torch.nn.Embedding(
            config.block_size,
            config.embedding_size,
        )
        self.embedding_dropout = torch.nn.Dropout(config.dropout)
        self.blocks = torch.nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.final_norm = torch.nn.LayerNorm(
            config.embedding_size,
            eps=config.layer_norm_epsilon,
        )
        self.output = torch.nn.Linear(
            config.embedding_size,
            config.vocab_size,
            bias=True,
        )
        self.apply(self._initialize_module)
        if config.tie_embeddings:
            self.output.weight = self.token_embedding.weight

    def _initialize_module(self, module: torch.nn.Module) -> None:
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initialization_std,
            )
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        token_ids: torch.Tensor,
        target_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        _, sequence_length = token_ids.shape
        if sequence_length > self.config.block_size:
            raise ValueError("sequence length exceeds configured context window")

        positions = torch.arange(sequence_length, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        hidden = self.embedding_dropout(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.output(self.final_norm(hidden))

        loss = None
        if target_ids is not None:
            if target_ids.shape != token_ids.shape:
                raise ValueError("target_ids must match token_ids shape")
            loss = functional.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                target_ids.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
