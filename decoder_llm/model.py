"""Configurable decoder architecture; source attribution is in PROVENANCE.md."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import math
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class LLMConfig:
    """Configuration for decoder-only transformer LLM."""
    vocab_size: int = 50257  # GPT-2 vocab size
    context_length: int = 1024  # Max sequence length
    d_model: int = 768  # Hidden dimension
    num_layers: int = 12  # Number of transformer layers
    num_heads: int = 12  # Number of attention heads
    d_ff: int = 3072  # Feed-forward hidden dimension
    sep_token_id: int = -1  # Optional prompt/target separator
    dropout: float = 0.3
    bias: bool = True


    def __post_init__(self):
        for name in ("vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if (not isinstance(self.sep_token_id, int) or isinstance(self.sep_token_id, bool)
                or not -1 <= self.sep_token_id < self.vocab_size):
            raise ValueError("sep_token_id must be -1 or a vocabulary index")
        if not isinstance(self.bias, bool):
            raise ValueError("bias must be a boolean")


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalizes by RMS across the last dimension:
        x / sqrt(mean(x^2) + eps) * weight
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS over last dimension (feature dimension)
        # Accumulate squares in float32 for half/bfloat16 stability.
        value = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        rms = torch.sqrt(torch.mean(value * value, dim=-1, keepdim=True) + self.eps)
        return (value / rms).to(x.dtype) * self.weight


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with causal masking."""

    def __init__(self, config: LLMConfig):
        """Initialize the CausalSelfAttention module."""
        super().__init__()
        assert config.d_model % config.num_heads == 0, "d_model must be divisible by num_heads"
        self.n_heads = config.num_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.num_heads
        self.dropout = config.dropout
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Project input -> Q, K, V in one go for efficiency
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.attn_drop = nn.Dropout(config.dropout)
        self.proj_drop = nn.Dropout(config.dropout)

        # Causal mask: prevents attending to future tokens.
        mask = torch.tril(torch.ones(config.context_length, config.context_length))
        self.register_buffer("causal_mask", mask, persistent=False)


    def forward(self, x: torch.Tensor, past_kv: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        B, T, C = x.shape

        qkv = self.qkv(x)                          # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)             # each (B, T, C)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)

        if past_kv is not None:
            past_k, past_v = past_kv               # (B, H, T_past, D)
            k = torch.cat([past_k, k], dim=2)      # (B, H, T_total, D)
            v = torch.cat([past_v, v], dim=2)      # (B, H, T_total, D)

        present_kv = (k, v)

        att = (q @ k.transpose(-2, -1)) * self.scale   # (B, H, T, T_total)

        T_total = k.size(2)
        q_start = T_total - T
        mask = self.causal_mask[q_start:q_start + T, :T_total].to(torch.bool)  # (T, T_total)

        att = att.masked_fill(~mask.view(1, 1, T, T_total), float("-inf"))

        att = att - att.max(dim=-1, keepdim=True).values
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v                                    # (B, H, T, D)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.proj_drop(y)
        return y, present_kv



class FeedForward(nn.Module):
    """Feed-forward network (MLP) in transformer."""

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=config.bias),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Single transformer decoder block."""

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor, past_kv: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        # Pre-norm transformer: normalize BEFORE sublayer (more stable for deep nets)
        attn_out, present_kv = self.attn(self.norm1(x), past_kv=past_kv)
        x = x + attn_out  # residual connection

        ff_out = self.ff(self.norm2(x))
        x = x + ff_out    # residual connection
        return x, present_kv

class DecoderOnlyTransformer(nn.Module):
    """Simple decoder-only transformer for language modeling."""

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.context_length, config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.d_model)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights with a GPT-style normal init.

        This isn't strictly required for correctness, but helps training stability
        and matches common transformer initialization.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, input_ids: torch.Tensor, past_kv: Optional[list] = None) -> Dict[str, Any]:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a rank-2 torch.long tensor")
        B, T = input_ids.shape
        if B < 1 or T < 1:
            raise ValueError("input_ids must contain at least one sequence and token")
        if past_kv is not None:
            if len(past_kv) != self.config.num_layers:
                raise ValueError("cache must have one key/value pair per layer")
            lengths = []
            expected = (B, self.config.num_heads)
            for pair in past_kv:
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise ValueError("each cache entry must be a key/value pair")
                k, v = pair
                if (not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor)
                        or k.ndim != 4 or k.shape != v.shape or k.shape[:2] != expected
                        or k.shape[3] != self.config.d_model // self.config.num_heads
                        or k.device != input_ids.device or v.device != input_ids.device
                        or k.dtype != self.token_embed.weight.dtype or v.dtype != k.dtype):
                    raise ValueError("cache shape, device or dtype does not match model input")
                lengths.append(k.shape[2])
            if len(set(lengths)) != 1:
                raise ValueError("cache lengths must agree across layers")

        # Determine the position indices for the input tokens, accounting for any past tokens in the cache.
        past_len = 0
        if past_kv is not None and len(past_kv) > 0 and past_kv[0] is not None:
            # past_kv[0][0] is k with shape (B, heads, past_len, head_dim)
            past_len = past_kv[0][0].size(2)

        if past_len + T > self.config.context_length:
            raise ValueError("input plus cache exceeds context_length")
        pos = torch.arange(past_len, past_len + T, device=input_ids.device).unsqueeze(0)


        x = self.token_embed(input_ids) + self.pos_embed(pos)
        x = self.embed_dropout(x)

        presents = []
        for i, block in enumerate(self.blocks):
            pkv = past_kv[i] if past_kv is not None else None
            x, present_kv = block(x, past_kv=pkv)
            presents.append(present_kv)

        x = self.norm(x)
        logits = self.lm_head(x)  # (B, T, vocab)
        return {"logits": logits, "past_kv": presents}


def compute_loss(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy loss for language modeling."""
    if logits.ndim != 3 or target_ids.shape != logits.shape[:2]:
        raise ValueError("targets must match the batch and time dimensions of logits")
    if not torch.any(target_ids != -100):
        raise ValueError("batch contains no supervised targets")
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           target_ids.reshape(-1), ignore_index=-100)



def build_model(cfg: LLMConfig) -> DecoderOnlyTransformer:
    return DecoderOnlyTransformer(cfg)
