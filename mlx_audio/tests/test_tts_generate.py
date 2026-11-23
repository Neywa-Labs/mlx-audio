"""Tests for mlx_audio.tts.generate.generate_audio file output paths."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mlx_audio.audio_io import read
from mlx_audio.tts.generate import generate_audio


def test_generate_audio_writes_bfloat16_output(tmp_path):
    """A bfloat16 model result must survive the non-stream write path.

    np.array() cannot express MLX bfloat16, so generate casts to float32
    before conversion — the same promotion write() performs internally.
    """
    sample_rate = 16000
    t = np.arange(1600) / sample_rate
    audio = mx.array(0.5 * np.sin(2 * np.pi * 440 * t), dtype=mx.bfloat16)
    model = SimpleNamespace(
        sample_rate=sample_rate,
        generate=lambda **kwargs: [
            SimpleNamespace(audio=audio, sample_rate=sample_rate)
        ],
    )

    assert generate_audio(
        "hello",
        model=model,
        output_path=str(tmp_path),
        file_prefix="out",
        audio_format="wav",
        verbose=False,
    )

    written, rate = read(tmp_path / "out_000.wav", dtype="int16")
    assert rate == sample_rate
    assert np.abs(written.astype(np.int32)).max() > 1000
