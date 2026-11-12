"""Whisper encoder and frame-stacking projector used by SOV-Transcribe."""

import mlx.core as mx
import mlx.nn as nn

from .config import AudioConfig, ModelConfig


class Attention(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        width = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.scale = (width // self.num_heads) ** -0.5
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)

    def __call__(self, x):
        batch, length, _ = x.shape
        # Salesforce scales the query before attention, including BF16 rounding.
        q = self.q_proj(x) * self.scale
        k, v = self.k_proj(x), self.v_proj(x)
        q, k, v = [
            t.reshape(batch, length, self.num_heads, -1).transpose(0, 2, 1, 3)
            for t in (q, k, v)
        ]
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0)
        return self.out_proj(y.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class EncoderLayer(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)

    def __call__(self, x):
        x = x + self.self_attn(self.self_attn_layer_norm(x))
        return x + self.fc2(nn.gelu(self.fc1(self.final_layer_norm(x))))


class AudioEncoder(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv1d(config.num_mel_bins, config.d_model, 3, padding=1)
        self.conv2 = nn.Conv1d(config.d_model, config.d_model, 3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(config.max_source_positions, config.d_model)
        self.layers = [EncoderLayer(config) for _ in range(config.encoder_layers)]
        self.layer_norm = nn.LayerNorm(config.d_model)

    def __call__(self, features):
        """Encode whole windows in (batch, frames, mel bins) layout."""
        if features.shape[1:] != (
            self.config.max_source_positions * 2,
            self.config.num_mel_bins,
        ):
            raise ValueError("Expected complete, padded audio feature windows")
        x = features.astype(self.conv1.weight.dtype)
        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))
        x = (x + self.embed_positions.weight).astype(x.dtype)
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)


class AudioProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_size = config.audio_config.d_model * config.downsample_factor
        width = config.text_config.hidden_size
        self.linear_1 = nn.Linear(self.input_size, width, bias=False)
        self.linear_2 = nn.Linear(width, width, bias=False)

    def __call__(self, x):
        return self.linear_2(nn.gelu(self.linear_1(x.reshape(-1, self.input_size))))
