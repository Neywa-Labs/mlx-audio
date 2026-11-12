# SOV-Transcribe-1

Native MLX support for
[Salesforce SOV-Transcribe-1](https://huggingface.co/salesforce-sovereign-intelligence/SOV-Transcribe-1):
a Whisper large-v3 encoder, a frame-stacking audio projector, and a Gemma 4
26B-A4B mixture-of-experts decoder. The checkpoint has 26.6B total parameters
and supports English, French, German, Spanish, Italian, Portuguese, and Dutch.

## Transcription

```python
from mlx_audio.stt import load

model = load("salesforce-sovereign-intelligence/SOV-Transcribe-1")
result = model.generate("recording.wav")
print(result.text)

# Optional transcription instructions and contextual vocabulary.
result = model.generate(
    "recording.wav",
    prompt="Transcribe the audio with punctuation and capitalization.",
    hotwords=["Nativ", "MLX"],
    max_tokens=8192,
)
```

Paths are decoded, mixed to mono, and resampled to 16 kHz. NumPy and MLX arrays
must already contain mono audio at 16 kHz. The default prompt matches Salesforce's
verbatim transcription request. `language` adds a language hint to the prompt.

```bash
python -m mlx_audio.stt.generate \
  --model salesforce-sovereign-intelligence/SOV-Transcribe-1 \
  --audio recording.wav --output-path transcript --format txt
```

Recordings are padded to complete 30-second windows. Log-mel normalization is
computed across the entire padded recording before splitting it into windows;
all 375 projected tokens per window are kept, including padding. The decoder
receives every window in a single request. `audio_batch_size` controls how many
windows are encoded together (default 1), and `prefill_step_size` controls text
prefill (default 512). Salesforce reports testing recordings up to 30 minutes;
this port's validation does not establish quality across that full duration.

`stream=True` yields incremental text after the recording has been encoded:

```python
for chunk in model.generate("recording.wav", stream=True):
    print(chunk.text, end="", flush=True)
```

This is an offline ASR model. Live audio streaming, word timestamps, and the
separate DFlash draft model are not implemented. `finish_reason` is `stop` on an
end token or `length` when the output token limit is reached.

## Conversion and quantization

The original BF16 checkpoint loads directly and requires approximately 53 GB
for downloaded weights, plus inference memory. The standard converter supports
an MLX checkpoint and optional decoder quantization:

```bash
python -m mlx_audio.convert \
  --hf-path salesforce-sovereign-intelligence/SOV-Transcribe-1 \
  --mlx-path ./SOV-Transcribe-1-4bit \
  --dtype bfloat16 --quantize --q-bits 4 --q-group-size 64
```

The audio encoder and projector retain their floating-point precision. MoE
routers remain FP32 and are excluded from quantization and dtype conversion.
Quantization changes numerical outputs and may change transcripts.

For strict FP32 numerical comparisons on GPUs that use reduced-precision FP32
matrix multiplication, start Python with `MLX_ENABLE_TF32=0`.
See [MLX's precision discussion](https://github.com/ml-explore/mlx/issues/3860).

## Reference and validation

The port follows the configuration, processor, and model code at upstream
revision [`505d8fa`](https://huggingface.co/salesforce-sovereign-intelligence/SOV-Transcribe-1/tree/505d8fae243924c0d8acdc622f7f0abab344eba3).
Tests cover source weight mapping, quantized weight reloads, sliding-window KV
caches, feature extraction across window boundaries, and small encoder/decoder
comparisons against Transformers. PyTorch is optional and only needed for those
reference comparisons, not for MLX inference.

```bash
MLX_ENABLE_TF32=0 python -m pytest mlx_audio/stt/tests/test_sov_transcribe.py -q
```

The upstream checkpoint is licensed under Apache 2.0.
