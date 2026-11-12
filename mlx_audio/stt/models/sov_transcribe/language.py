"""SOV's Gemma 4 MoE decoder, with FP32 expert routing."""

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.lm.models.base import create_attention_mask
from mlx_audio.lm.models.cache import KVCache, RotatingKVCache
from mlx_audio.lm.models.rope_utils import initialize_rope
from mlx_audio.lm.models.switch_layers import SwitchGLU

from .config import TextConfig


class GeGLU(nn.Module):
    def __call__(self, x, gate):
        return nn.gelu_approx(gate) * x


class MLP(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, x):
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class Router(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.eps = config.rms_norm_eps
        self.top_k = config.top_k_experts
        self.root_size = config.hidden_size**-0.5
        self.scale = mx.ones(config.hidden_size)
        self.per_expert_scale = mx.ones(config.num_experts)
        self.proj = nn.Linear(config.hidden_size, config.num_experts, bias=False)

    def __call__(self, x):
        x = mx.fast.rms_norm(x.astype(mx.float32), None, self.eps)
        scores = self.proj(x * self.scale * self.root_size)
        probabilities = mx.softmax(scores, axis=-1, precise=True)
        indices = mx.argpartition(-probabilities, kth=self.top_k - 1, axis=-1)[
            ..., : self.top_k
        ]
        weights = mx.take_along_axis(probabilities, indices, axis=-1)
        weights = weights / weights.sum(axis=-1, keepdims=True)
        return indices, weights * self.per_expert_scale[indices]


class Attention(nn.Module):
    def __init__(self, config: TextConfig, layer_type: str):
        super().__init__()
        self.is_sliding = layer_type == "sliding_attention"
        self.head_dim = config.head_dim if self.is_sliding else config.global_head_dim
        self.num_heads = config.num_attention_heads
        self.k_eq_v = config.attention_k_eq_v and not self.is_sliding
        self.num_kv_heads = (
            config.num_global_key_value_heads
            if self.k_eq_v
            else config.num_key_value_heads
        )
        self.eps = config.rms_norm_eps
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        if not self.k_eq_v:
            self.v_proj = nn.Linear(
                config.hidden_size,
                self.num_kv_heads * self.head_dim,
                bias=config.attention_bias,
            )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=self.eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=self.eps)
        rope = config.rope_parameters[layer_type]
        self.rope = initialize_rope(
            self.head_dim,
            base=rope["rope_theta"],
            traditional=False,
            scaling_config=rope,
            max_position_embeddings=config.max_position_embeddings,
        )

    def __call__(self, x, mask, cache):
        batch, length, _ = x.shape
        q = self.q_norm(
            self.q_proj(x).reshape(batch, length, self.num_heads, self.head_dim)
        )
        k = self.k_proj(x).reshape(batch, length, self.num_kv_heads, self.head_dim)
        v = (
            k
            if self.k_eq_v
            else self.v_proj(x).reshape(batch, length, self.num_kv_heads, self.head_dim)
        )
        offset = cache.offset if cache is not None else 0
        q = self.rope(q.transpose(0, 2, 1, 3), offset=offset)
        k = self.rope(self.k_norm(k).transpose(0, 2, 1, 3), offset=offset)
        v = mx.fast.rms_norm(v, None, self.eps).transpose(0, 2, 1, 3)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=mask)
        return self.o_proj(y.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class DecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_type: str):
        super().__init__()
        self.self_attn = Attention(config, layer_type)
        self.mlp = MLP(config)
        norms = [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ]
        self.enable_moe = config.enable_moe_block
        if self.enable_moe:
            self.router = Router(config)
            self.experts = SwitchGLU(
                config.hidden_size,
                config.moe_intermediate_size,
                config.num_experts,
                activation=GeGLU(),
            )
            norms += [
                "post_feedforward_layernorm_1",
                "post_feedforward_layernorm_2",
                "pre_feedforward_layernorm_2",
            ]
        for name in norms:
            setattr(self, name, nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps))
        self.layer_scalar = mx.ones((1,))

    def __call__(self, x, mask, cache):
        x = x + self.post_attention_layernorm(
            self.self_attn(self.input_layernorm(x), mask, cache)
        )
        h = self.mlp(self.pre_feedforward_layernorm(x))
        if self.enable_moe:
            indices, weights = self.router(x)
            experts = self.experts(self.pre_feedforward_layernorm_2(x), indices)
            # The reference weights each expert in FP32, then accumulates in
            # the activation dtype. Do not promote the decoder residual to FP32.
            experts = (experts * weights[..., None]).astype(x.dtype).sum(axis=-2)
            h = self.post_feedforward_layernorm_1(
                h
            ) + self.post_feedforward_layernorm_2(experts)
        return (x + self.post_feedforward_layernorm(h)) * self.layer_scalar.astype(
            x.dtype
        )


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayer(config, kind) for kind in config.layer_types]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed(self, tokens):
        x = self.embed_tokens(tokens)
        return x * mx.array(self.config.hidden_size**0.5, dtype=x.dtype)

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.config.sliding_window, keep=0)
                if kind == "sliding_attention"
                else KVCache()
            )
            for kind in self.config.layer_types
        ]

    def __call__(self, inputs, cache=None, input_embeddings=None):
        x = self.embed(inputs) if input_embeddings is None else input_embeddings
        if cache is None:
            cache = [None] * len(self.layers)
        masks = {}
        for layer, kind, state in zip(self.layers, self.config.layer_types, cache):
            if kind not in masks:
                window = (
                    self.config.sliding_window if kind == "sliding_attention" else None
                )
                masks[kind] = create_attention_mask(x, state, window_size=window)
            x = layer(x, masks[kind], state)
        return self.norm(x)
