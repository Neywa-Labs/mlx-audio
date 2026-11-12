"""SOV input contracts, conversion, cache, and optional Transformers parity."""

import json
from dataclasses import asdict
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_audio.registry import classify_model
from mlx_audio.stt.models.sov_transcribe import Model, ModelConfig
from mlx_audio.stt.models.sov_transcribe.config import AudioConfig, TextConfig
from mlx_audio.stt.models.sov_transcribe.sov_transcribe import (
    TranscriptionOutput,
    audio_features,
)


@pytest.fixture
def config():
    return ModelConfig(
        audio_config=AudioConfig(
            d_model=16,
            encoder_layers=2,
            encoder_attention_heads=2,
            encoder_ffn_dim=32,
            num_mel_bins=8,
            max_source_positions=8,
        ),
        text_config=TextConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_global_key_value_heads=1,
            head_dim=16,
            global_head_dim=32,
            vocab_size=64,
            num_experts=4,
            top_k_experts=2,
            moe_intermediate_size=64,
            sliding_window=4,
            layer_types=["sliding_attention", "full_attention"],
        ),
        audio_token_id=63,
    )


def test_registry_and_nested_config(config):
    assert classify_model("sovereignai_transcribe") == "stt"
    assert classify_model("sov_transcribe") == "stt"
    assert ModelConfig.from_dict(asdict(config)) == config
    with pytest.raises(ValueError, match="shared KV"):
        TextConfig(num_kv_shared_layers=1)
    with pytest.raises(ValueError, match="divide"):
        ModelConfig(downsample_factor=7)


def test_sanitize_and_quantization_roundtrip(config):
    model = Model(config)
    source = {}
    params = dict(tree_flatten(model.parameters()))
    for key, value in params.items():
        if key.endswith((".experts.gate_proj.weight", ".experts.up_proj.weight")):
            if key.endswith("gate_proj.weight"):
                source["model." + key.replace("gate_proj.weight", "gate_up_proj")] = (
                    mx.concatenate(
                        [
                            value,
                            params[key.replace("gate_proj.weight", "up_proj.weight")],
                        ],
                        axis=-2,
                    )
                )
            continue
        if key.endswith(".experts.down_proj.weight"):
            key = key.removesuffix(".weight")
        if key in {"audio_tower.conv1.weight", "audio_tower.conv2.weight"}:
            value = value.transpose(0, 2, 1)
        source["model." + key] = value
    source["lm_head.weight"] = params["language_model.embed_tokens.weight"]
    clean = model.sanitize(source)
    model.load_weights(list(clean.items()), strict=True)
    assert set(clean) == set(params)
    for key in clean:
        np.testing.assert_array_equal(np.array(clean[key]), np.array(params[key]))
    assert model.sanitize(clean).keys() == clean.keys()
    nn.quantize(
        model,
        group_size=32,
        bits=4,
        class_predicate=lambda p, m: hasattr(m, "to_quantized")
        and model.model_quant_predicate(p, m),
    )
    assert isinstance(model.language_model.layers[0].mlp.up_proj, nn.QuantizedLinear)
    assert isinstance(model.language_model.layers[0].router.proj, nn.Linear)
    assert model.language_model.layers[0].router.proj.weight.dtype == mx.float32
    assert isinstance(model.embed_audio.linear_1, nn.Linear)
    quantized = dict(tree_flatten(model.parameters()))
    restored = Model(config)
    nn.quantize(
        restored,
        group_size=32,
        bits=4,
        class_predicate=lambda p, m: hasattr(m, "to_quantized")
        and restored.model_quant_predicate(p, m),
    )
    restored.load_weights(list(restored.sanitize(quantized).items()), strict=True)
    model.eval()
    restored.eval()
    ids = mx.array([[1, 2, 3]])
    np.testing.assert_array_equal(np.array(model(ids)), np.array(restored(ids)))


@pytest.mark.parametrize("chunks", [[1] * 13, [3, 4, 2, 4], [8, 4, 1]])
def test_cached_decode_crosses_sliding_window(config, chunks):
    # CPU FP32 avoids M5's default reduced-precision GEMM in this strict check.
    with mx.stream(mx.cpu):
        mx.random.seed(7)
        model = Model(config)
        model.eval()
        ids = mx.array([list(range(13))])
        expected = np.array(model(ids))
        cache = model.make_cache()
        pieces, offset = [], 0
        for size in chunks:
            pieces.append(np.array(model(ids[:, offset : offset + size], cache=cache)))
            offset += size
        np.testing.assert_allclose(
            np.concatenate(pieces, axis=1), expected, atol=2e-5, rtol=2e-5
        )
        assert cache[0].size() == config.text_config.sliding_window
        assert cache[1].offset == 13


@pytest.mark.parametrize("audio", [[], [[1.0]], [float("nan")], [float("inf")]])
def test_invalid_audio(audio):
    with pytest.raises(ValueError):
        audio_features(audio)


def test_generation_validation_and_stream_aggregation(config):
    model = Model(config)
    for kwargs in (
        {"max_tokens": 0},
        {"prefill_step_size": 0},
        {"audio_batch_size": 0},
        {"word_timestamps": True},
    ):
        with pytest.raises(ValueError):
            model.generate(mx.zeros(10), **kwargs)
    chunks = [
        TranscriptionOutput(text="Hello", is_final=False),
        TranscriptionOutput(text=" world", generation_tokens=2, finish_reason="stop"),
    ]
    with patch.object(model, "_generate_stream", return_value=iter(chunks)) as generate:
        output = model.generate(mx.zeros(10), hotwords=["Nativ"], language="en")
        assert output.text == "Hello world"
        assert output.generation_tokens == 2
        assert output.finish_reason == "stop"
        assert "Nativ" in generate.call_args.kwargs["prompt"]
    with patch.object(model, "_generate_stream", return_value=iter(chunks)):
        assert list(model.generate(mx.zeros(10), stream=True)) == chunks


@pytest.mark.parametrize("stream", [False, True])
def test_cli_json_export_and_untimed_subtitle_rejection(config, tmp_path, stream):
    from mlx_audio.stt.generate import generate_transcription

    model = Model(config)
    chunks = [
        TranscriptionOutput(text="Hello", is_final=False),
        TranscriptionOutput(text=" world"),
    ]
    with patch.object(
        model, "_generate_stream", side_effect=lambda *a, **k: iter(chunks)
    ):
        output = generate_transcription(
            model=model,
            audio=mx.zeros(10),
            output_path=str(tmp_path / "transcript"),
            format="json",
            stream=stream,
        )
        assert output.text == "Hello world"
        saved = json.loads((tmp_path / "transcript.json").read_text())
        assert saved["text"] == "Hello world"
        assert saved["segments"][0]["text"] == "Hello world"
        with pytest.raises(ValueError, match="untimed|timestamps"):
            generate_transcription(
                model=model,
                audio=mx.zeros(10),
                output_path=str(tmp_path / "subtitles"),
                format="srt",
                stream=stream,
            )


@pytest.mark.parametrize("eos", [1, 106, 50])
def test_generate_stops_on_all_checkpoint_end_tokens(config, eos):
    class Tokenizer:
        def decode(self, ids, **kwargs):
            return "".join({4: "hello ", 5: "world"}.get(i, "BAD") for i in ids)

    model = Model(config)
    model._tokenizer = Tokenizer()
    tokens = mx.array([2, 3])
    embeddings = mx.zeros((2, config.text_config.hidden_size))
    module = "mlx_audio.stt.models.sov_transcribe.sov_transcribe"
    with (
        patch.object(model, "_prepare_inputs", return_value=(tokens, embeddings)),
        patch(
            module + ".generate_step",
            return_value=iter([(4, None), (5, None), (eos, None), (4, None)]),
        ),
    ):
        output = model.generate(mx.zeros(10))
    assert output.text == "hello world"
    assert output.finish_reason == "stop"
    assert output.generation_tokens == 2
    assert output.total_tokens == 4


def test_audio_embeddings_replace_placeholders_without_text_scaling(config):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "prompt<|audio|>response"

        def encode(self, text, **kwargs):
            assert text == "prompt<|audio>" + "<|audio|>" * 4 + "<audio|>response"
            return [2, 3, 63, 63, 63, 63, 4]

    model = Model(config)
    model._tokenizer = Tokenizer()
    audio_embeddings = mx.ones((4, config.text_config.hidden_size))
    module = "mlx_audio.stt.models.sov_transcribe.sov_transcribe"
    with (
        patch(module + ".audio_features", return_value=mx.zeros((2, 16, 8))),
        patch.object(model, "get_audio_features", return_value=audio_embeddings),
    ):
        tokens, embeddings = model._prepare_inputs(mx.zeros(10), "prompt", 1)
    np.testing.assert_array_equal(np.array(embeddings[2:6]), np.array(audio_embeddings))
    np.testing.assert_array_equal(
        np.array(embeddings[:2]), np.array(model.language_model.embed(tokens[:2]))
    )
    np.testing.assert_array_equal(
        np.array(embeddings[-1:]), np.array(model.language_model.embed(tokens[-1:]))
    )


@pytest.mark.parametrize("samples", [1, 16000, 480000, 480160])
def test_features_match_transformers(samples):
    from transformers import WhisperFeatureExtractor

    rng = np.random.default_rng(1)
    audio = rng.normal(0, 0.05, samples).astype(np.float32)
    if samples > 480000:
        audio[480000:] *= 100
    windows = (samples + 479999) // 480000
    expected = WhisperFeatureExtractor(feature_size=128)(
        [audio],
        sampling_rate=16000,
        padding="max_length",
        max_length=windows * 480000,
        truncation=False,
        return_tensors="np",
    )["input_features"][0].T.reshape(windows, 3000, 128)
    # As with the decoder check, compare full FP32 independently of the GPU's
    # default TF32 setting. GPU inference is exercised by the smoke tests.
    with mx.stream(mx.cpu):
        actual = np.array(audio_features(audio))
    # The shared MLX Whisper mel bank is constructed in FP32, while HF builds
    # its Slaney filters in FP64. The largest observed log-mel delta is 1.8e-4
    # on CPU (9.4e-5 on GPU with TF32 disabled), near a filter boundary.
    np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=1e-4)


def test_encoder_and_projector_match_torch(config):
    torch = pytest.importorskip("torch")
    from transformers import WhisperConfig
    from transformers.models.whisper.modeling_whisper import WhisperEncoder

    torch.manual_seed(7)
    cfg = WhisperConfig(**asdict(config.audio_config))
    cfg._attn_implementation = "eager"
    reference = WhisperEncoder(cfg).eval()
    model = Model(config)
    weights = {
        "model.audio_tower." + key: mx.array(value.detach().numpy())
        for key, value in reference.state_dict().items()
    }
    model.load_weights(list(model.sanitize(weights).items()), strict=False)
    model.eval()
    inputs = np.random.default_rng(3).normal(size=(2, 16, 8)).astype(np.float32)
    with torch.no_grad():
        expected = reference(
            torch.from_numpy(inputs).transpose(1, 2)
        ).last_hidden_state.numpy()
    with mx.stream(mx.cpu):
        actual = np.array(model.audio_tower(mx.array(inputs)))
        np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-6)
        projected = np.array(model.embed_audio(mx.array(actual)))
    w1, w2 = [
        torch.from_numpy(np.array(l.weight))
        for l in (model.embed_audio.linear_1, model.embed_audio.linear_2)
    ]
    stacked = torch.from_numpy(expected.reshape(-1, 64))
    want = torch.nn.functional.linear(
        torch.nn.functional.gelu(torch.nn.functional.linear(stacked, w1)), w2
    )
    np.testing.assert_allclose(projected, want.numpy(), atol=2e-6, rtol=2e-6)


def test_gemma4_decoder_matches_transformers(config):
    torch = pytest.importorskip("torch")
    gemma = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
    from transformers import Gemma4TextConfig

    torch.manual_seed(3)
    cfg = Gemma4TextConfig(**asdict(config.text_config), use_double_wide_mlp=False)
    cfg._attn_implementation = "eager"
    reference = gemma.Gemma4TextModel(cfg).eval()
    model = Model(config)
    weights = {
        "model.language_model." + key: mx.array(value.detach().numpy())
        for key, value in reference.state_dict().items()
    }
    model.load_weights(list(model.sanitize(weights).items()), strict=False)
    model.eval()
    ids = np.array([[3, 7, 2, 8, 9, 10, 11, 12, 13]])
    with torch.no_grad():
        expected = reference(torch.from_numpy(ids)).last_hidden_state.numpy()
    with mx.stream(mx.cpu):
        actual = np.array(model.language_model(mx.array(ids)))
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
