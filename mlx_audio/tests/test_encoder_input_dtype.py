"""Audio encoders must run in their own weight dtype, not the feature dtype.

HF feature extractors and mel frontends hand float32 tensors to the encoder.
Published checkpoints store encoder weights in bfloat16 or float16, and MLX
convs/matmuls promote instead of erroring, so an uncast float32 input silently
ran the entire audio tower in float32 — diverging from the reference and
roughly doubling encoder memory traffic on every transcription.

The same leak came back through positional embeddings: fixed sinusoid tables
and embedding weights can be float32 in the checkpoint, so ``x + pos_emb``
re-promoted the activations even when the input had been cast.

Each test builds an encoder in a low-precision dtype and feeds fp32 input;
the output must stay in the weight dtype. All of them fail on main. Position
tables are left at the dtype a real load produces, so the ``astype`` guards
after the pos-emb add are genuinely exercised.

fun_asr_nano and sensevoice are deliberately absent: their SANM encoders are
fp32-only by design upstream, so the promotion is the intended behavior there.
Call-site-only casts (granite_speech_nar) and encoders that already cast
correctly are also untested here.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

ENCODER_DTYPES = [mx.bfloat16, mx.float16]


def _cast(module: nn.Module, dtype, keep_fp32=()) -> nn.Module:
    """Recreate a low-precision checkpoint, optionally keeping fp32 buffers."""
    module.set_dtype(dtype)
    for path in keep_fp32:
        obj = module
        parts = path.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], getattr(obj, parts[-1]).astype(mx.float32))
    return module


def _assert_dtype(module, features, dtype, *args):
    out = module(features, *args) if args else module(features)
    mx.eval(out)
    assert out.dtype == dtype


@pytest.mark.parametrize("dtype", ENCODER_DTYPES)
def test_qwen3_asr_audio_encoder_dtype(dtype):
    from mlx_audio.stt.models.qwen3_asr.config import AudioEncoderConfig
    from mlx_audio.stt.models.qwen3_asr.qwen3_asr import AudioEncoder

    enc = AudioEncoder(
        AudioEncoderConfig(
            num_mel_bins=16,
            encoder_layers=1,
            encoder_attention_heads=2,
            encoder_ffn_dim=64,
            d_model=32,
            max_source_positions=64,
            n_window=8,
            n_window_infer=16,
            conv_chunksize=16,
            output_dim=32,
            downsample_hidden_size=32,
        )
    )
    # The sinusoid table is a persistent=False buffer built in fp32 and is not
    # in the checkpoint, so it stays fp32 even on a bf16 checkpoint.
    _cast(enc, dtype, keep_fp32=("positional_embedding._positional_embedding",))
    features = mx.random.normal((1, 16, 64))
    # 50 valid frames over T=64 exercises the mask, padding and ragged chunks.
    mask = mx.concatenate([mx.ones((1, 50)), mx.zeros((1, 14))], axis=1).astype(
        mx.bool_
    )
    _assert_dtype(enc, features, dtype, mask)


@pytest.mark.parametrize("dtype", ENCODER_DTYPES)
def test_voxtral_audio_encoder_dtype(dtype):
    from mlx_audio.stt.models.voxtral.config import AudioConfig
    from mlx_audio.stt.models.voxtral.voxtral import Encoder

    enc = Encoder(
        AudioConfig(
            num_mel_bins=16,
            encoder_layers=1,
            encoder_attention_heads=2,
            encoder_ffn_dim=64,
            d_model=32,
            max_source_positions=32,
        )
    )
    # embed_positions is stored fp32 in published checkpoints.
    _cast(enc, dtype, keep_fp32=("embed_positions.weight",))
    # (B, T, n_mels); conv2's stride-2 halves T to max_source_positions
    _assert_dtype(enc, mx.random.normal((1, 64, 16)), dtype)


@pytest.mark.parametrize("dtype", ENCODER_DTYPES)
def test_qwen2_audio_encoder_dtype(dtype):
    from mlx_audio.stt.models.qwen2_audio.config import EncoderConfig
    from mlx_audio.stt.models.qwen2_audio.qwen2_audio import Qwen2AudioEncoder

    enc = Qwen2AudioEncoder(
        EncoderConfig(
            num_mel_bins=16,
            encoder_layers=1,
            encoder_attention_heads=2,
            encoder_ffn_dim=64,
            d_model=32,
            max_source_positions=64,
        )
    )
    # _embed_positions is computed in fp32 at init and only recast by
    # post_load_hook; a bare encoder still relies on the guard in __call__.
    _cast(enc, dtype, keep_fp32=("_embed_positions",))
    _assert_dtype(enc, mx.random.normal((1, 16, 64)), dtype)


@pytest.mark.parametrize("dtype", ENCODER_DTYPES)
def test_higgs_audio_3_encoder_dtype(dtype):
    from mlx_audio.stt.models.higgs_audio_3.config import AudioEncoderConfig
    from mlx_audio.stt.models.higgs_audio_3.higgs_audio_3 import HiggsAudioEncoder

    enc = HiggsAudioEncoder(
        AudioEncoderConfig(
            num_mel_bins=16,
            encoder_layers=1,
            encoder_attention_heads=2,
            encoder_ffn_dim=64,
            d_model=32,
            max_source_positions=64,
        )
    )
    # Keeping embed_positions fp32 exercises the recast after the pos-emb add;
    # whisper-family tables are stored fp32 in several published checkpoints.
    _assert_dtype(
        _cast(enc, dtype, keep_fp32=("embed_positions.weight",)),
        mx.random.normal((1, 16, 64)),
        dtype,
    )


@pytest.mark.parametrize("dtype", ENCODER_DTYPES)
def test_mimo_v2_asr_audio_encoder_dtype(dtype):
    from mlx_audio.stt.models.mimo_v2_asr.audio_encoder import (
        AudioEncoder,
        AudioEncoderConfig,
    )

    enc = AudioEncoder(
        AudioEncoderConfig(
            n_mels=16,
            d_model=32,
            encoder_layers=1,
            encoder_attention_heads=2,
            encoder_ffn_dim=64,
            encoder_skip_layer_id=None,
            max_audio_seconds=1,
        )
    )
    # inv_freq is a non-persistent buffer, so a real load leaves it fp32.
    _cast(enc, dtype, keep_fp32=("position_embedding.inv_freq",))
    # mel is (n_mels, L)
    _assert_dtype(enc._get_features, mx.random.normal((16, 40)), dtype)


@pytest.mark.parametrize("dtype", ENCODER_DTYPES)
def test_vibevoice_asr_tokenizer_encoder_dtype(dtype):
    from mlx_audio.stt.models.vibevoice_asr.audio_encoder import TokenizerEncoder

    enc = TokenizerEncoder(
        channels=1,
        vae_dim=8,
        n_filters=8,
        ratios=[2],
        depths=[1, 1],
        causal=False,
    )
    _cast(enc, dtype)
    _assert_dtype(enc, mx.random.normal((1, 64)), dtype)


@pytest.mark.parametrize("dtype", ENCODER_DTYPES)
def test_voxtral_realtime_encoder_dtype(dtype):
    from mlx_audio.stt.models.voxtral_realtime.config import EncoderConfig
    from mlx_audio.stt.models.voxtral_realtime.encoder import AudioEncoder

    enc = AudioEncoder(
        EncoderConfig(dim=32, n_layers=1, hidden_dim=64, downsample_factor=4)
    )
    _cast(enc, dtype)
    # conv_stem takes log-mel (128, frames); conv in-channels is fixed at 128
    _assert_dtype(enc.conv_stem, mx.random.normal((128, 32)), dtype)


@pytest.mark.parametrize("dtype", ENCODER_DTYPES)
def test_nemotron_subsampling_dtype(dtype):
    from mlx_audio.stt.models.nemotron_asr.config import ConformerArgs
    from mlx_audio.stt.models.nemotron_asr.conformer import CausalDwStridingSubsampling

    # Single choke point for every mel path: Conformer.__call__, the streaming
    # _encode_mel_chunk path, and the buffered paths all reach this module.
    sub = CausalDwStridingSubsampling(
        ConformerArgs(
            feat_in=64,
            d_model=32,
            subsampling_factor=8,
            subsampling_conv_channels=8,
        )
    )
    _cast(sub, dtype)
    out, _ = sub(mx.random.normal((1, 40, 64)), mx.array([40], dtype=mx.int32))
    mx.eval(out)
    assert out.dtype == dtype


@pytest.mark.parametrize("dtype", ENCODER_DTYPES)
def test_moss_whisper_encoder_dtype(dtype):
    from mlx_audio.stt.models.moss_transcribe_diarize.config import AudioConfig
    from mlx_audio.stt.models.moss_transcribe_diarize.moss_transcribe_diarize import (
        MossWhisperEncoder,
    )

    enc = MossWhisperEncoder(
        AudioConfig(
            num_mel_bins=16,
            d_model=32,
            encoder_layers=1,
            encoder_attention_heads=2,
            encoder_ffn_dim=64,
            max_source_positions=32,
        )
    )
    # An fp32 embed_positions table would re-promote the activations without
    # the guard after the pos-emb add.
    _cast(enc, dtype, keep_fp32=("embed_positions.weight",))
    _assert_dtype(enc, mx.random.normal((1, 64, 16)), dtype)


@pytest.mark.parametrize("dtype", ENCODER_DTYPES)
@pytest.mark.parametrize("use_rope", [True, False])
def test_glmasr_whisper_encoder_dtype(dtype, use_rope):
    from mlx_audio.stt.models.glmasr.config import WhisperConfig
    from mlx_audio.stt.models.glmasr.glmasr import WhisperEncoder

    enc = WhisperEncoder(
        WhisperConfig(
            num_mel_bins=16,
            encoder_layers=1,
            encoder_attention_heads=2,
            encoder_ffn_dim=64,
            d_model=32,
            max_source_positions=64,
        ),
        use_rope=use_rope,
    )
    # Keeping embed_positions fp32 exercises the recast after the pos-emb
    # add; only the use_rope=False arm reads it.
    _assert_dtype(
        _cast(enc, dtype, keep_fp32=("embed_positions.weight",)),
        mx.random.normal((1, 64, 16)),
        dtype,
    )
