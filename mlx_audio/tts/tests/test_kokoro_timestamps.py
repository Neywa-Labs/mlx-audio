from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_audio.tts.models.kokoro.pipeline import KokoroPipeline


def token(phonemes, whitespace):
    return SimpleNamespace(
        phonemes=phonemes, whitespace=whitespace, start_ts=None, end_ts=None
    )


def test_join_timestamps_uses_predicted_durations():
    tokens = [token("hI", " "), token("ju", "")]
    pred_dur = mx.array([5, 3, 4, 2, 6, 1, 7])

    KokoroPipeline.join_timestamps(tokens, pred_dur)

    assert tokens[0].start_ts == pytest.approx(0.05)
    assert tokens[0].end_ts == pytest.approx(0.25)
    assert tokens[1].start_ts == pytest.approx(0.25)
    assert tokens[1].end_ts == pytest.approx(0.45)
