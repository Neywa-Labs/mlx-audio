from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.tts.models.kokoro.kokoro import Model
from mlx_audio.tts.models.kokoro.pipeline import KokoroPipeline


class OneSecondPipeline:
    """Yields one segment shaped like the Kokoro decoder output: (1, samples)."""

    voices = {}

    def __call__(self, text, voice=None, speed=1, split_pattern=None):
        output = SimpleNamespace(audio=mx.zeros((1, 24000)), pred_dur=None)
        yield KokoroPipeline.Result(graphemes=text, phonemes="hello", output=output)


def test_generate_reports_the_number_of_audio_samples():
    model = Model.__new__(Model)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(sample_rate=24000)
    model._pipelines = {"a": OneSecondPipeline()}

    (result,) = model.generate("hello", voice="af_heart")

    assert result.audio.shape == (24000,)
    assert result.samples == 24000
    assert result.audio_samples["samples"] == 24000
    assert result.audio_duration == "00:00:01.000"
