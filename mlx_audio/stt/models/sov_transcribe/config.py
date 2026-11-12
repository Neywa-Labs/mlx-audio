"""Configuration for Salesforce SOV-Transcribe-1."""

from dataclasses import dataclass, field

from mlx_audio.base import BaseModelArgs


@dataclass
class AudioConfig(BaseModelArgs):
    d_model: int = 1280
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    num_mel_bins: int = 128
    max_source_positions: int = 1500


@dataclass
class TextConfig(BaseModelArgs):
    model_type: str = "gemma4_text"
    hidden_size: int = 2816
    intermediate_size: int = 2112
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    num_global_key_value_heads: int = 2
    head_dim: int = 256
    global_head_dim: int = 512
    rms_norm_eps: float = 1e-6
    vocab_size: int = 262144
    sliding_window: int = 1024
    max_position_embeddings: int = 262144
    attention_k_eq_v: bool = True
    attention_bias: bool = False
    enable_moe_block: bool = True
    num_experts: int = 128
    top_k_experts: int = 8
    moe_intermediate_size: int = 704
    num_kv_shared_layers: int = 0
    hidden_size_per_layer_input: int = 0
    final_logit_softcapping: float | None = 30.0
    layer_types: list[str] | None = None
    rope_parameters: dict = field(
        default_factory=lambda: {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {
                "rope_type": "proportional",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25,
            },
        }
    )

    def __post_init__(self):
        if self.model_type != "gemma4_text":
            raise ValueError("SOV-Transcribe requires a Gemma 4 text decoder")
        if self.num_kv_shared_layers or self.hidden_size_per_layer_input:
            raise ValueError(
                "SOV-Transcribe does not use shared KV or per-layer inputs"
            )
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if (i + 1) % 6 == 0 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers or not set(
            self.layer_types
        ).issubset({"sliding_attention", "full_attention"}):
            raise ValueError("Expected one supported attention type per decoder layer")
        if self.enable_moe_block and not 0 < self.top_k_experts <= self.num_experts:
            raise ValueError("top_k_experts must be in [1, num_experts]")


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "sovereignai_transcribe"
    audio_config: AudioConfig = field(default_factory=AudioConfig)
    text_config: TextConfig = field(default_factory=TextConfig)
    audio_token_id: int = 258881
    downsample_factor: int = 4
    projector_hidden_act: str = "gelu"
    tie_word_embeddings: bool = True
    eos_token_id: int = 1
    pad_token_id: int = 0

    def __post_init__(self):
        if isinstance(self.audio_config, dict):
            self.audio_config = AudioConfig.from_dict(self.audio_config)
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        if self.projector_hidden_act != "gelu":
            raise ValueError("SOV-Transcribe requires a GELU audio projector")
        if self.downsample_factor < 1 or (
            self.audio_config.max_source_positions % self.downsample_factor
        ):
            raise ValueError("downsample_factor must divide the encoder frame count")
