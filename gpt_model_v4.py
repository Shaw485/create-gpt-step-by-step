"""Maximum practical GPT architecture for the single-novel v4 experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

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


@dataclass(frozen=True)
class GPTInferenceCache:
    """Per-layer KV tensors for evaluation-only autoregressive decoding.

    The cache is deliberately not an ``nn.Module`` and therefore never appears
    in a checkpoint or changes the model parameter/state-dict contract.  Each
    layer entry stores key/value tensors shaped
    ``[batch, heads, cached_sequence, head_size]``.
    """

    layers: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    sequence_length: int
    batch_size: int
    key_valid_mask: torch.Tensor | None = None

    def select_rows(self, rows: Sequence[int] | torch.Tensor) -> GPTInferenceCache:
        """Keep active batch rows after one or more rows emit EOS."""

        if not self.layers:
            raise ValueError("inference cache has no layers")
        device = self.layers[0][0].device
        indices = torch.as_tensor(rows, dtype=torch.long, device=device)
        if indices.ndim != 1:
            raise ValueError("cache row indices must be one-dimensional")
        if indices.numel() == 0:
            raise ValueError("cannot retain an empty inference cache")
        if bool(torch.any(indices < 0)) or bool(torch.any(indices >= self.batch_size)):
            raise ValueError("cache row index is out of range")
        selected = tuple(
            (
                key.index_select(0, indices),
                value.index_select(0, indices),
            )
            for key, value in self.layers
        )
        selected_key_valid_mask = (
            None
            if self.key_valid_mask is None
            else self.key_valid_mask.index_select(0, indices)
        )
        return GPTInferenceCache(
            layers=selected,
            sequence_length=self.sequence_length,
            batch_size=int(indices.numel()),
            key_valid_mask=selected_key_valid_mask,
        )


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

    def forward_inference(
        self,
        x: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        key_valid_mask: torch.Tensor | None = None,
        query_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run attention and return reusable keys/values for eval decoding."""

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
        past_length = 0
        if past_key_value is not None:
            past_key, past_value = past_key_value
            expected_prefix = (batch_size, self.num_heads)
            if past_key.ndim != 4 or past_value.ndim != 4:
                raise ValueError("cached key/value tensors must be four-dimensional")
            if past_key.shape != past_value.shape:
                raise ValueError("cached key/value tensor shapes differ")
            if tuple(past_key.shape[:2]) != expected_prefix:
                raise ValueError("cached key/value batch or head shape differs")
            if int(past_key.shape[3]) != self.head_size:
                raise ValueError("cached key/value head size differs")
            if past_key.device != key.device or past_value.device != value.device:
                raise ValueError("cached key/value device differs")
            if past_key.dtype != key.dtype or past_value.dtype != value.dtype:
                raise ValueError("cached key/value dtype differs")
            past_length = int(past_key.shape[2])
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)

        if key_valid_mask is not None:
            total_length = past_length + sequence_length
            if key_valid_mask.shape != (batch_size, total_length):
                raise ValueError("attention key mask shape differs")
            if query_valid_mask is None:
                query_valid_mask = torch.ones(
                    (batch_size, sequence_length),
                    dtype=torch.bool,
                    device=x.device,
                )
            if query_valid_mask.shape != (batch_size, sequence_length):
                raise ValueError("attention query mask shape differs")
            causal = torch.ones(
                (sequence_length, total_length),
                dtype=torch.bool,
                device=x.device,
            ).tril(diagonal=past_length)
            attention_mask = (
                causal[None, None, :, :]
                & key_valid_mask[:, None, None, :]
            )
            # Left-padding queries have no real key to attend.  Their outputs
            # are discarded, but giving each one its own padding key avoids an
            # all-masked SDPA row on CPU/MPS without exposing it to real tokens.
            fallback = torch.zeros_like(causal)
            query_indices = torch.arange(sequence_length, device=x.device)
            fallback[query_indices, past_length + query_indices] = True
            attention_mask = attention_mask | (
                ~query_valid_mask[:, None, :, None]
                & fallback[None, None, :, :]
            )
            attended = functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
        elif past_length == 0:
            attended = functional.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=True,
            )
        elif sequence_length == 1:
            # A single appended query may attend every cached key and itself.
            attended = functional.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
            )
        else:
            total_length = past_length + sequence_length
            causal_mask = torch.ones(
                (sequence_length, total_length),
                dtype=torch.bool,
                device=x.device,
            ).tril(diagonal=past_length)
            attended = functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=causal_mask,
                dropout_p=0.0,
                is_causal=False,
            )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            embedding_size,
        )
        output = self.residual_dropout(self.output(attended))
        return output, (key, value)


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

    def forward_inference(
        self,
        x: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        key_valid_mask: torch.Tensor | None = None,
        query_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Inference-only block path that also returns its attention KV cache."""

        attention_output, present_key_value = self.attention.forward_inference(
            self.attention_norm(x),
            past_key_value,
            key_valid_mask=key_valid_mask,
            query_valid_mask=query_valid_mask,
        )
        x = x + attention_output
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x, present_key_value


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

    @property
    def supports_left_padded_inference(self) -> bool:
        """Declare support for the masked heterogeneous-batch cache contract."""

        return True

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

    @torch.no_grad()
    def forward_inference(
        self,
        token_ids: torch.Tensor,
        cache: GPTInferenceCache | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, GPTInferenceCache]:
        """Evaluate tokens with an optional KV cache, without changing ``forward``.

        Position indices continue from the cached prefix.  Callers must discard
        and rebuild the cache whenever their context window slides, because the
        regular model resets cropped-window positions to zero.
        """

        if self.training:
            raise RuntimeError("forward_inference requires model.eval()")
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        batch_size, sequence_length = token_ids.shape
        if sequence_length <= 0:
            raise ValueError("inference token sequence must not be empty")
        current_valid_mask: torch.Tensor | None = None
        if attention_mask is not None:
            if attention_mask.shape != token_ids.shape:
                raise ValueError("inference attention mask must match token_ids")
            current_valid_mask = attention_mask.to(
                device=token_ids.device,
                dtype=torch.bool,
            )
            if bool(torch.any(current_valid_mask.sum(dim=1) == 0)):
                raise ValueError("each inference row must contain a valid token")

        past_length = 0
        layer_caches: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None
        past_valid_mask: torch.Tensor | None = None
        if cache is not None:
            if len(cache.layers) != len(self.blocks):
                raise ValueError("inference cache layer count differs")
            if cache.batch_size != batch_size:
                raise ValueError("inference cache batch size differs")
            past_length = int(cache.sequence_length)
            layer_caches = cache.layers
            past_valid_mask = cache.key_valid_mask
            if past_valid_mask is not None:
                if past_valid_mask.shape != (batch_size, past_length):
                    raise ValueError("inference cache key mask shape differs")
                past_valid_mask = past_valid_mask.to(
                    device=token_ids.device,
                    dtype=torch.bool,
                )
            for key, value in layer_caches:
                if int(key.shape[2]) != past_length or int(value.shape[2]) != past_length:
                    raise ValueError("inference cache sequence lengths differ")

        total_length = past_length + sequence_length
        if total_length > self.config.block_size:
            raise ValueError("cached sequence length exceeds configured context window")

        combined_valid_mask: torch.Tensor | None = None
        if past_valid_mask is not None or current_valid_mask is not None:
            if past_valid_mask is None:
                past_valid_mask = torch.ones(
                    (batch_size, past_length),
                    dtype=torch.bool,
                    device=token_ids.device,
                )
            if current_valid_mask is None:
                current_valid_mask = torch.ones(
                    (batch_size, sequence_length),
                    dtype=torch.bool,
                    device=token_ids.device,
                )
            combined_valid_mask = torch.cat(
                (past_valid_mask, current_valid_mask),
                dim=1,
            )
            past_valid_counts = past_valid_mask.to(torch.long).sum(dim=1)
            current_offsets = current_valid_mask.to(torch.long).cumsum(dim=1) - 1
            positions = past_valid_counts[:, None] + current_offsets
            positions = torch.where(
                current_valid_mask,
                positions,
                torch.zeros_like(positions),
            )
        else:
            positions = torch.arange(
                past_length,
                total_length,
                device=token_ids.device,
            )
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        hidden = self.embedding_dropout(hidden)
        present_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_index, block in enumerate(self.blocks):
            past_key_value = (
                None if layer_caches is None else layer_caches[layer_index]
            )
            hidden, present_key_value = block.forward_inference(
                hidden,
                past_key_value,
                key_valid_mask=combined_valid_mask,
                query_valid_mask=current_valid_mask,
            )
            present_layers.append(present_key_value)
        logits = self.output(self.final_norm(hidden))
        return logits, GPTInferenceCache(
            layers=tuple(present_layers),
            sequence_length=total_length,
            batch_size=batch_size,
            key_valid_mask=combined_valid_mask,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
