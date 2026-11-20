# kale-tts-1

kale-tts-1 is a family of small English TTS models with 10 designed voices (8 US English, 2 British English), made by
Kale. All three are Apache-2.0.

| Model | Parameters | What it is | HuggingFace |
|-------|-----------:|------------|-------------|
| `kale-tts-1-nano` | 7.5M | small ALBERT + prosody predictor, frame-rate source-filter decoder | [KaleComputer/kale-tts-1-nano](https://huggingface.co/KaleComputer/kale-tts-1-nano) |
| `kale-tts-1-tiny` | 45.7M | Ettin-encoder-17m word reader + a small student text side, Kokoro's prosody predictor, source-filter decoder | [KaleComputer/kale-tts-1-tiny](https://huggingface.co/KaleComputer/kale-tts-1-tiny) |
| `kale-tts-1-base` | 82M | Kokoro-82M fine-tuned on the 10 voices (runs through the [Kokoro](kokoro.md) model) | [KaleComputer/kale-tts-1-base](https://huggingface.co/KaleComputer/kale-tts-1-base) |

tiny and nano are distilled from base. Their decoder is not Kokoro's iSTFTNet: a few ConvNeXt blocks at the frame
rate shape the spectrum of a 64-harmonic sine bank built from the predicted pitch, plus pitch-pulsing and short-burst
noise. Both end every sentence with a per-voice pitch clamp and a per-voice unvoicing floor, stored with each voice.

## Installation

tiny and nano read text with the release's own front end (number, date and term reading, a checked CMUdict lexicon,
a small g2p model, British rules for the British voices). It ships with the model files and runs in QuickJS:

```bash
pip install quickjs
# converting the release files on first load also needs:
pip install onnx          # tiny / nano
pip install torch misaki  # base (Kokoro's .pth checkpoint and Kokoro's own front end)
```

## Usage

The first load converts the release files (`kale-tts-1-{tiny,nano}.fp32.onnx` for tiny / nano, the Kokoro `.pth` for base) into MLX
weights in `~/.cache/mlx_audio/kale_tts/` (`MLX_AUDIO_CACHE` to move it); later loads use the cache.

=== "Python"

    ```python
    from mlx_audio.tts.utils import load_model

    model = load_model("KaleComputer/kale-tts-1-nano")   # or kale-tts-1-tiny
    print(model.voice_names())   # kenzie, simone, amy, frankie, darlene, owen, toby, jack, earl, magnus

    for result in model.generate("A volcano is a place where very hot, melted rock comes up out of the ground.",
                                 voice="kenzie"):
        audio = result.audio   # 24 kHz, one result per sentence
    ```

=== "CLI"

    ```bash
    mlx_audio.tts.generate --model KaleComputer/kale-tts-1-tiny \
        --text "The museum opens at 9 o'clock and tickets cost £12." --voice amy
    ```

=== "base (Kokoro)"

    ```python
    model = load_model("KaleComputer/kale-tts-1-base")   # converted to a `kokoro` model
    # voices are <lang_code>_<name>: a_kenzie ... a_magnus, b_amy, b_jack
    for result in model.generate("Hello there!", voice="b_amy", lang_code="b"):
        ...
    ```

Convert once by hand (for example to publish MLX weights):

```bash
python -m mlx_audio.tts.models.kale_tts.convert --hf-path KaleComputer/kale-tts-1-nano --mlx-path ./kale-tts-1-nano-mlx
```

## Notes

- The text side (small, sequential LSTMs) runs on the CPU stream by default, where its rounding of the durations
  matches the release graphs exactly; the decoder runs on the GPU. `model.text_device = mx.gpu` is ~20 % faster on
  an M5 Max but can round an occasional phoneme duration differently.
- `speed` scales the predicted durations (1.0 = each voice's natural pace).
- Checked against the release ONNX graphs (20 check lines x 2 voices, same front-end ids, same decoder noise):
  every sentence has the same length, median waveform SNR 41 dB (nano) / 43 dB (tiny) against the float32 graphs,
  Whisper large-v3-turbo word errors equal to the ONNX runtime's.
