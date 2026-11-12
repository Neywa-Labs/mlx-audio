"""Native MLX inference for Salesforce SOV-Transcribe-1."""

import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.lm.generate import StreamingDetokenizer, generate_step, generation_stream
from mlx_audio.lm.sample_utils import make_repetition_penalty, make_sampler
from mlx_audio.stt.models.base import STTOutput
from mlx_audio.stt.models.whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram
from mlx_audio.stt.utils import load_audio, merge_hotwords, wired_limit

from .config import ModelConfig
from .encoder import AudioEncoder, AudioProjector
from .language import LanguageModel

DEFAULT_PROMPT = (
    "Transcribe the audio verbatim, outputting only spoken words in sequence."
)


def audio_features(audio, n_mels=128):
    """Normalize over the entire padded recording, then split into 30s windows."""
    audio = mx.array(audio, dtype=mx.float32)
    if audio.ndim != 1 or not audio.size:
        raise ValueError("Expected nonempty mono audio sampled at 16000 Hz")
    if not mx.all(mx.isfinite(audio)).item():
        raise ValueError("Audio contains non-finite samples")
    windows = math.ceil(audio.size / N_SAMPLES)
    features = log_mel_spectrogram(
        audio, n_mels=n_mels, padding=windows * N_SAMPLES - audio.size
    )
    return features.reshape(windows, N_FRAMES, n_mels)


@dataclass
class TranscriptionOutput(STTOutput):
    is_final: bool = True
    finish_reason: str | None = None
    start_time: float | None = None
    end_time: float | None = None


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.audio_tower = AudioEncoder(config.audio_config)
        self.embed_audio = AudioProjector(config)
        self.language_model = LanguageModel(config.text_config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(
                config.text_config.hidden_size,
                config.text_config.vocab_size,
                bias=False,
            )
        self._tokenizer = None
        self._eos_token_ids = {config.eos_token_id, 106, 50}

    @property
    def sample_rate(self):
        return 16000

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def __call__(self, inputs, cache=None, input_embeddings=None):
        h = self.language_model(inputs, cache=cache, input_embeddings=input_embeddings)
        logits = (
            self.language_model.embed_tokens.as_linear(h)
            if self.config.tie_word_embeddings
            else self.lm_head(h)
        )
        cap = self.config.text_config.final_logit_softcapping
        return mx.tanh(logits / cap) * cap if cap is not None else logits

    @property
    def cast_predicate(self):
        return lambda key: ".router." not in key

    def model_quant_predicate(self, path, module):
        # Keep the acoustic frontend and the FP32 MoE routers unquantized.
        return (
            path.startswith(("language_model.", "lm_head")) and ".router." not in path
        )

    def sanitize(self, weights):
        result = {}
        for key, value in weights.items():
            source_format = key.startswith("model.")
            if source_format:
                key = key.removeprefix("model.")
            if key == "lm_head.weight" and self.config.tie_word_embeddings:
                continue
            if source_format and key in {
                "audio_tower.conv1.weight",
                "audio_tower.conv2.weight",
            }:
                value = value.transpose(0, 2, 1)
            if key.endswith(".experts.gate_up_proj"):
                prefix = key.removesuffix("gate_up_proj")
                gate, up = mx.split(value, 2, axis=-2)
                result[prefix + "gate_proj.weight"] = gate
                result[prefix + "up_proj.weight"] = up
                continue
            if key.endswith(".experts.down_proj"):
                key += ".weight"
            if ".router." in key:
                value = value.astype(mx.float32)
            result[key] = value
        return result

    @classmethod
    def post_load_hook(cls, model, model_path):
        from transformers import GemmaTokenizer

        path = Path(model_path)
        model._tokenizer = GemmaTokenizer.from_pretrained(path)
        for filename, key in [
            ("processor_config.json", "generation_eos_token_ids"),
            ("generation_config.json", "eos_token_id"),
        ]:
            if (path / filename).exists():
                ids = json.loads((path / filename).read_text()).get(key)
                if ids is not None:
                    model._eos_token_ids.update([ids] if isinstance(ids, int) else ids)
        return model

    def get_audio_features(self, features, batch_size=1):
        if batch_size < 1:
            raise ValueError("audio_batch_size must be positive")
        projected = []
        for start in range(0, features.shape[0], batch_size):
            x = self.embed_audio(self.audio_tower(features[start : start + batch_size]))
            mx.eval(x)
            projected.append(x)
        return mx.concatenate(projected, axis=0)

    def _prepare_inputs(self, audio, prompt, audio_batch_size):
        if self._tokenizer is None:
            raise RuntimeError("Load SOV-Transcribe using mlx_audio.stt.load")
        if isinstance(audio, (str, Path)):
            audio = load_audio(str(audio), sr=self.sample_rate)
        features = audio_features(audio, self.config.audio_config.num_mel_bins)
        count = (
            features.shape[0]
            * self.config.audio_config.max_source_positions
            // self.config.downsample_factor
        )
        # Use the checkpoint's canonical chat template, including its whitespace.
        text = self._tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "audio"},
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        if text.count("<|audio|>") != 1:
            raise ValueError("The prompt must not contain audio placeholder tokens")
        text = text.replace("<|audio|>", "<|audio>" + "<|audio|>" * count + "<audio|>")
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        if len(ids) >= self.config.text_config.max_position_embeddings:
            raise ValueError("Audio and prompt exceed the model context window")
        positions = [
            i for i, token in enumerate(ids) if token == self.config.audio_token_id
        ]
        if len(positions) != count or positions != list(
            range(positions[0], positions[0] + count)
        ):
            raise ValueError("Audio placeholder count does not match audio features")
        audio_embeddings = self.get_audio_features(features, audio_batch_size)
        tokens = mx.array(ids)
        embeddings = self.language_model.embed(tokens)
        start = positions[0]
        embeddings = mx.concatenate(
            [
                embeddings[:start],
                audio_embeddings.astype(embeddings.dtype),
                embeddings[start + count :],
            ]
        )
        mx.eval(embeddings)
        return tokens, embeddings

    def _generate_stream(
        self,
        audio,
        *,
        prompt,
        language,
        audio_batch_size,
        max_tokens,
        temperature,
        top_p,
        top_k,
        min_p,
        repetition_penalty,
        repetition_context_size,
        prefill_step_size,
        verbose,
    ):
        with wired_limit(self, [generation_stream]):
            started = time.perf_counter()
            if verbose:
                print("Encoding audio...")
            tokens, embeddings = self._prepare_inputs(audio, prompt, audio_batch_size)
            if (
                len(tokens) + max_tokens
                > self.config.text_config.max_position_embeddings
            ):
                raise ValueError(
                    "Requested generation exceeds the model context window"
                )
            decoder = StreamingDetokenizer(self._tokenizer, skip_special_tokens=True)
            processors = (
                [make_repetition_penalty(repetition_penalty, repetition_context_size)]
                if repetition_penalty is not None
                else None
            )
            count = 0
            finish_reason = "length"
            decode_started = None
            for token, _ in generate_step(
                tokens,
                self,
                input_embeddings=embeddings,
                max_tokens=max_tokens,
                sampler=make_sampler(
                    temperature, top_p=top_p, top_k=top_k, min_p=min_p
                ),
                logits_processors=processors,
                prefill_step_size=prefill_step_size,
            ):
                if decode_started is None:
                    decode_started = time.perf_counter()
                if token in self._eos_token_ids:
                    finish_reason = "stop"
                    break
                count += 1
                decoder.add_token(token)
                if delta := decoder.last_segment:
                    yield TranscriptionOutput(
                        text=delta,
                        is_final=False,
                        language=language,
                        prompt_tokens=len(tokens),
                        generation_tokens=count,
                    )
            decoder.finalize()
            ended = time.perf_counter()
            yield TranscriptionOutput(
                text=decoder.last_segment,
                language=language,
                prompt_tokens=len(tokens),
                generation_tokens=count,
                total_tokens=len(tokens) + count,
                total_time=ended - started,
                prompt_tps=len(tokens) / max(decode_started - started, 1e-9),
                generation_tps=max(count - 1, 0) / max(ended - decode_started, 1e-9),
                finish_reason=finish_reason,
            )

    def generate(
        self,
        audio,
        *,
        prompt=None,
        language=None,
        hotwords=None,
        max_tokens=8192,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        repetition_penalty=None,
        repetition_context_size=64,
        prefill_step_size=512,
        audio_batch_size=1,
        verbose=False,
        stream=False,
        task="transcribe",
        word_timestamps=False,
        **kwargs,
    ):
        """Transcribe a recording; stream=True yields text deltas after audio encoding."""
        if max_tokens < 1 or prefill_step_size < 1 or audio_batch_size < 1:
            raise ValueError("Token and batch sizes must be positive")
        if task not in {"asr", "transcribe"}:
            raise ValueError(
                "Use a natural-language prompt to control SOV transcription"
            )
        if word_timestamps:
            raise ValueError("SOV-Transcribe does not produce word timestamps")
        prompt = prompt or DEFAULT_PROMPT
        if language:
            prompt += f"\nThe spoken language is {language}."
        if words := merge_hotwords(None, hotwords):
            prompt += f"\nContextual vocabulary: {words}"
        results = self._generate_stream(
            audio,
            prompt=prompt,
            language=language,
            audio_batch_size=audio_batch_size,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
            prefill_step_size=prefill_step_size,
            verbose=verbose,
        )
        if stream:
            return results
        pieces = []
        for result in results:
            pieces.append(result.text)
        text = "".join(pieces).strip()
        return replace(result, text=text, segments=[{"text": text}] if text else [])
