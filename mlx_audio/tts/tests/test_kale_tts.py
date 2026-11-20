"""kale-tts-1 (tiny / nano): building blocks and a full forward pass on small random configs (no downloads)."""

import json

import mlx.core as mx
import numpy as np
import pytest

from mlx_audio.tts.models.kale_tts.convert import _LSTM_GATES, _wn
from mlx_audio.tts.models.kale_tts.frontend import kn_ipa
from mlx_audio.tts.models.kale_tts.head import F0_HOP, istft, sine_bank, stft
from mlx_audio.tts.models.kale_tts.kale_tts import Model, ModelConfig


@pytest.mark.parametrize(("n_fft", "hop"), [(600, 150), (150, 75)])
def test_stft_istft_round_trip(n_fft, hop):
    x = mx.random.normal((F0_HOP * 20,), key=mx.random.key(0))
    y = istft(stft(x, n_fft, hop), n_fft, hop, x.shape[0])
    assert y.shape == x.shape
    assert float(mx.abs(y - x).max()) < 1e-4


def test_stft_matches_numpy_reference():
    x = np.random.default_rng(0).standard_normal(3000).astype(np.float32)
    n, hop = 600, 150
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)
    xp = np.pad(x, n // 2, mode="reflect")
    frames = np.stack([xp[i : i + n] * w for i in range(0, len(xp) - n + 1, hop)])
    ref = np.fft.rfft(frames, axis=-1).T
    out = np.array(stft(mx.array(x), n, hop))
    assert np.abs(out - ref).max() < 1e-3


def test_sine_bank_unvoiced_is_silent_and_harmonics_stop_at_11khz():
    e, cyc, uv = sine_bank(mx.zeros((10,)))
    assert float(mx.abs(e).max()) == 0 and float(uv.sum()) == 0
    # 3 kHz fundamental: only harmonics 1..3 are below 11 kHz -> peak amplitude <= 3 * 0.1
    e, _, uv = sine_bank(mx.full((10,), 3000.0))
    assert float(uv.min()) == 1
    assert float(mx.abs(e).max()) <= 0.3 + 1e-5


def _voice(floor=120.0, tail=36.0, thr=50.0, unvoice=80.0):
    return {
        k: mx.array([v], dtype=mx.float32)
        for k, v in dict(
            pitch_floor=floor, clamp_tail=tail, clamp_thr=thr, unvoice_hz=unvoice
        ).items()
    }


def test_end_clamp_raises_the_last_voiced_frames_only():
    f0 = np.full(100, 200.0, np.float32)
    f0[80:90] = 90.0  # a falling ending below the floor
    f0[90:] = 0.0  # silence after the sentence
    out = np.array(Model._end_clamp(mx.array(f0)[None], _voice(floor=120.0)))[0]
    assert (out[80:90] == 120.0).all()
    assert (out[:80] == 200.0).all() and (out[90:] == 0).all()
    # floor 0 = no clamp; frames under the unvoicing floor become unvoiced
    out = np.array(
        Model._end_clamp(mx.array(f0)[None], _voice(floor=0.0, unvoice=95.0))
    )[0]
    assert (out[80:90] == 0).all() and (out[:80] == 200.0).all()


def test_weight_norm_split_restores_the_weight():
    from mlx_audio.tts.models.kokoro.istftnet import weight_norm

    w = np.random.default_rng(0).standard_normal((8, 5, 3)).astype(np.float32)
    v, g = _wn(w)
    assert np.allclose(
        np.array(weight_norm(mx.array(v), mx.array(g), dim=0)),
        w.transpose(0, 2, 1),
        atol=1e-5,
    )


def test_lstm_gate_order_onnx_to_torch():
    # ONNX stores gates as i, o, f, c; MLX / torch expect i, f, g(c), o
    onnx_order = ["i", "o", "f", "c"]
    assert [onnx_order[k] for k in _LSTM_GATES] == ["i", "f", "c", "o"]


def test_kokoro_native_symbols():
    assert kn_ipa(list("ðə vɑlkˈeɪnoʊ"), gb=False) == list("ðə vɑlkˈAnO")
    assert kn_ipa(list("ɡˈəʊ"), gb=True) == ["ɡ", "ˈ", "Q"]
    assert kn_ipa(list("bˈʌɾɚ"), gb=False) == ["b", "ˈ", "ʌ", "T", "ə", "ɹ"]


VOCAB = {c: i + 1 for i, c in enumerate("abdefhijklmnopstuvwz ")}


def _config(text_side, tmp_path):
    cfg = dict(
        text_side=text_side,
        n_token=40,
        hidden_dim=32,
        style_dim=8 if text_side == "albert" else 128,
        n_layer=2,
        decoder=dict(
            width=24, up_channels=16, style_dim=4, head=dict(dim=20, blocks=1)
        ),
        vocab=VOCAB,
        phonemes="ipa",
        voices=[{"id": "v0", "name": "Test", "accent": "en-us"}],
        model_path=str(tmp_path),
    )
    if text_side == "albert":
        cfg["plbert"] = dict(
            num_hidden_layers=2,
            num_attention_heads=2,
            hidden_size=16,
            intermediate_size=32,
            max_position_embeddings=64,
            embedding_size=8,
        )
    else:
        cfg["ettin"] = dict(
            vocab_size=50,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=3,
            num_attention_heads=2,
        )
        cfg["student"] = dict(c=16, nconv=1, ntf=1, wdim=16, heads=2)
    return cfg


@pytest.mark.parametrize("text_side", ["albert", "ettin"])
def test_forward_pass_shapes(text_side, tmp_path):
    cfg = _config(text_side, tmp_path)
    (tmp_path / "voices").mkdir()
    voice = {
        "style": mx.random.normal((64, 256)),
        "sound": mx.random.normal((4,)),
        "prosody": mx.random.normal((8,)),
        **{k: v for k, v in _voice().items()},
    }
    mx.save_safetensors(str(tmp_path / "voices" / "test.safetensors"), voice)
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    model = Model(ModelConfig.from_dict(cfg))
    ids = [VOCAB[c] for c in "hello this"]
    ettin = None
    if text_side == "ettin":
        ettin = {
            "ids": [1, 5, 7, 2],
            "S": 4,
            "W": 2,
            "wavg": [0, 0.5, 0.5, 0, 0, 0, 0, 1],
            "widx": [0] * 5 + [1] * 5,
        }
    y = model.synthesize(ids, "test", 1.0, ettin)
    mx.eval(y)
    assert y.ndim == 1 and y.shape[0] > 0 and y.shape[0] % (2 * F0_HOP) == 0
    assert bool(mx.isfinite(y).all())
