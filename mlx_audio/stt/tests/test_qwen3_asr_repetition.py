import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mlx.core as mx
import numpy as np

from mlx_audio.stt.models.qwen3_asr.qwen3_asr import (
    Qwen3ASRModel,
    _chunk_token_budget,
    _ends_in_repetition,
    detect_and_fix_repetitions,
)

EOS = 99


class _Tokenizer:
    eos_token_id = EOS
    unk_token_id = -1

    def convert_tokens_to_ids(self, token):
        return {"<|endoftext|>": EOS}.get(token, self.unk_token_id)

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)


class _Embeddings:
    def __call__(self, input_ids):
        return mx.zeros((*input_ids.shape, 1))

    def as_linear(self, hidden):
        return mx.zeros((*hidden.shape[:2], 128))


def _model():
    model = Qwen3ASRModel.__new__(Qwen3ASRModel)
    model.config = SimpleNamespace(
        text_config=SimpleNamespace(num_hidden_layers=0), support_languages=[]
    )
    model._tokenizer = _Tokenizer()
    model._feature_extractor = object()
    model.model = Mock(side_effect=lambda inputs_embeds, cache=None: inputs_embeds)
    model.model.embed_tokens = _Embeddings()
    model.lm_head = None
    model.get_audio_features = Mock(return_value=mx.zeros((1, 1)))
    model._preprocess_audio = Mock(return_value=(mx.zeros((1, 1)), None, 1))
    model._build_prompt = Mock(return_value=mx.array([[0]]))
    model._build_inputs_embeds = Mock(return_value=mx.zeros((1, 1, 1)))
    return model


class TestRepetitionHelpers(unittest.TestCase):
    def test_ends_in_repetition(self):
        self.assertTrue(_ends_in_repetition([1, 2] + [7] * 20))
        self.assertFalse(_ends_in_repetition([1, 2] + [7] * 19))
        self.assertTrue(_ends_in_repetition([5] + [1, 2] * 20))
        self.assertFalse(_ends_in_repetition([5] + [1, 2] * 19))
        self.assertTrue(_ends_in_repetition(list(range(10)) * 20))
        # A phrase said a few times in a row is not a loop.
        self.assertFalse(_ends_in_repetition(list(range(10)) * 3))
        self.assertFalse(_ends_in_repetition(list(range(40))))

    def test_detect_and_fix_repetitions(self):
        # Expected values from the official qwen_asr.inference.utils function.
        cases = {
            "a" * 25 + "b": "ab",
            "x" + "哈" * 21 + "y": "x哈y",
            "hello " + "ha" * 25 + "!": "hello ha!",
            "a" * 20: "a" * 20,
            "the cat sat on the mat": "the cat sat on the mat",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(detect_and_fix_repetitions(text), expected)

    def test_chunk_token_budget_scales_with_duration(self):
        self.assertEqual(_chunk_token_budget(5 * 16000, 16000, 8192), 128)
        self.assertEqual(_chunk_token_budget(30 * 16000, 16000, 8192), 392)
        self.assertEqual(_chunk_token_budget(1200 * 16000, 16000, 8192), 8192)
        self.assertEqual(_chunk_token_budget(30 * 16000, 16000, 100), 100)


class TestDecodeGuards(unittest.TestCase):
    def test_stream_generate_stops_on_a_loop(self):
        model = _model()
        looping = ((mx.array(7), None) for _ in itertools.count())
        with patch("mlx_audio.lm.generate.generate_step", return_value=looping):
            tokens = list(model.stream_generate(np.zeros(16000), max_tokens=8192))
        self.assertEqual(len(tokens), 20)

    def test_each_chunk_gets_a_duration_budget(self):
        model = _model()
        budgets = []

        def single(chunk_audio, **kwargs):
            budgets.append(kwargs["max_tokens"])
            return "text", 1, 1

        model._generate_single_chunk = single
        model.generate(
            np.zeros(16000 * 40), chunk_duration=30, language="English", max_tokens=8192
        )
        self.assertEqual(len(budgets), 2)
        self.assertTrue(all(128 <= b <= 392 for b in budgets), budgets)

    def test_transcript_gets_the_official_cleanup(self):
        model = _model()
        model._generate_single_chunk = Mock(
            return_value=("language English<asr_text>yes " + "no " * 30, 5, 1)
        )
        out = model.generate(np.zeros(16000), language=None)
        self.assertEqual(out.text, "yes no ")

    def test_batched_rows_stop_on_their_own_loop_and_budget(self):
        model = _model()
        step = itertools.count()

        def sample(logprobs):
            i = next(step)
            # row 0 loops on token 7; row 1 counts up and never repeats.
            return mx.array([7, 1000 + i])

        model._forward_with_embeds = lambda embeds, cache: mx.zeros(
            (embeds.shape[0], 1, 128)
        )
        texts, gen_tokens, _, _ = model._generate_chunks_batched(
            [(np.zeros(16000), 0.0), (np.zeros(16000), 1.0)],
            max_tokens=8192,
            sampler=sample,
            logits_processors=None,
            language="English",
            system_prompt=None,
            batch_size=2,
            verbose=False,
        )
        self.assertEqual(gen_tokens, [20, 128])
        self.assertEqual(texts[0], " ".join(["7"] * 20))


if __name__ == "__main__":
    unittest.main()
