import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_audio import utils
from mlx_audio.stt.models.qwen3_asr.config import (
    AudioEncoderConfig,
    ModelConfig,
    TextConfig,
)
from mlx_audio.stt.models.qwen3_asr.qwen3_asr import Qwen3ASRModel
from mlx_audio.stt.models.qwen3_asr.qwen3_forced_aligner import (
    ForcedAlignerConfig,
    ForcedAlignerModel,
)

GROUP = 32


def _audio_config():
    return AudioEncoderConfig(
        num_mel_bins=8,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=64,
        d_model=32,
        output_dim=64,
        downsample_hidden_size=32,
    )


def _text_config():
    return TextConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        tie_word_embeddings=True,
    )


def _quantize(model, audio_tower=True):
    nn.quantize(
        model,
        group_size=GROUP,
        bits=8,
        class_predicate=lambda p, m: hasattr(m, "to_quantized")
        and (audio_tower or not p.startswith("audio_tower"))
        and m.weight.shape[-1] % GROUP == 0,
    )
    return dict(tree_flatten(model.parameters()))


def _load(model, weights):
    """The steps of mlx_audio.utils.base_load_model, with strict loading."""
    weights = model.sanitize(weights)
    utils.apply_quantization(
        model,
        {"quantization": {"group_size": GROUP, "bits": 8}},
        weights,
        model.model_quant_predicate,
    )
    model.load_weights(list(weights.items()), strict=True)
    return model


class TestQwen3ASRQuantizedLoad(unittest.TestCase):
    def assert_loaded(self, model, weights):
        for key, value in tree_flatten(model.parameters()):
            self.assertTrue(mx.array_equal(value, weights[key]), key)

    def test_asr_checkpoint_with_quantized_audio_tower_and_tied_head(self):
        config = ModelConfig(audio_config=_audio_config(), text_config=_text_config())
        weights = _quantize(Qwen3ASRModel(config))
        # Exports that keep a copy of the tied head also store its quantization.
        for name in ("weight", "scales", "biases"):
            weights[f"lm_head.{name}"] = weights[f"model.embed_tokens.{name}"]

        model = _load(Qwen3ASRModel(config), dict(weights))

        self.assertIsInstance(model.audio_tower.layers[0].fc1, nn.QuantizedLinear)
        self.assertIsInstance(model.model.layers[0].mlp.up_proj, nn.QuantizedLinear)
        self.assertIsNone(model.lm_head)
        self.assert_loaded(model, weights)

    def test_asr_checkpoint_with_float_audio_tower_still_loads_float(self):
        config = ModelConfig(audio_config=_audio_config(), text_config=_text_config())
        weights = _quantize(Qwen3ASRModel(config), audio_tower=False)

        model = _load(Qwen3ASRModel(config), dict(weights))

        self.assertIsInstance(model.audio_tower.layers[0].fc1, nn.Linear)
        self.assertNotIsInstance(model.audio_tower.layers[0].fc1, nn.QuantizedLinear)
        self.assertIsInstance(model.model.layers[0].mlp.up_proj, nn.QuantizedLinear)
        self.assert_loaded(model, weights)

    def test_forced_aligner_checkpoint_with_quantized_audio_tower(self):
        config = ForcedAlignerConfig(
            audio_config=_audio_config(), text_config=_text_config(), classify_num=64
        )
        weights = _quantize(ForcedAlignerModel(config))

        model = _load(ForcedAlignerModel(config), dict(weights))

        self.assertIsInstance(model.audio_tower.layers[0].fc1, nn.QuantizedLinear)
        self.assertIsInstance(model.lm_head, nn.QuantizedLinear)
        self.assert_loaded(model, weights)


if __name__ == "__main__":
    unittest.main()
