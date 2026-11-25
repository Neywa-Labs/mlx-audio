"""kale-tts-1-tiny and kale-tts-1-nano (Kale, Apache-2.0) in MLX.

Both are distilled from kale-tts-1-base (Kokoro-82M fine-tuned on ten voices designed with Maya-1):
  nano (7.5M): a small ALBERT + Kokoro-style prosody predictor at ~1/3 width with one 32-d prosody row per voice,
        its own text encoder, a 3.6M AdaIN trunk and a frame-rate source-filter head (no iSTFTNet).
  tiny (45.7M): the frozen Ettin-encoder-17m word reader + a 3M student in place of ALBERT, Kokoro's full-size prosody
        predictor and text encoder, and the same kind of trunk + source-filter head.
Both graphs end with a per-voice end-of-sentence pitch clamp and a per-voice unvoicing floor (voice inputs).
kale-tts-1-base is plain Kokoro and loads through the `kokoro` model type (see convert.py).
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import BaseModelArgs, GenerationResult
from ..kokoro.istftnet import AdainResBlk1d, ConvWeighted
from ..kokoro.modules import (
    LSTM,
    AlbertModelArgs,
    CustomAlbert,
    ProsodyPredictor,
    TextEncoder,
)
from .head import SourceFilterHead
from .text_side import Ettin, EttinArgs, StudentTextSide


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "kale_tts"
    variant: str = "nano"
    text_side: str = "albert"  # "albert" (nano) or "ettin" (tiny)
    n_token: int = 178
    hidden_dim: int = 192
    style_dim: int = 32
    n_layer: int = 3
    max_dur: int = 50
    text_encoder_kernel_size: int = 5
    plbert: Optional[dict] = None
    ettin: Optional[dict] = None
    student: Optional[dict] = None
    decoder: Optional[dict] = None
    vocab: Dict[str, int] = field(default_factory=dict)
    phonemes: str = "kokoro-native"
    voices: List[dict] = field(default_factory=list)
    sample_rate: int = 24000
    model_path: Optional[str] = None


class AffineInstanceNorm1d(nn.Module):
    """InstanceNorm1d with learned scale and shift (the students train them; stock Kokoro keeps them at 1 / 0)."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((num_features,))
        self.bias = mx.zeros((num_features,))

    def __call__(self, x):  # [B, C, T]
        mean = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        return (x - mean) * mx.rsqrt(var + self.eps) * self.weight[:, None] + self.bias[
            :, None
        ]


class AffineAdaIN1d(nn.Module):
    def __init__(self, style_dim: int, num_features: int):
        super().__init__()
        self.norm = AffineInstanceNorm1d(num_features)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def __call__(self, x, s):
        gamma, beta = mx.split(self.fc(s)[..., None], 2, axis=1)
        return (1 + gamma) * self.norm(x) + beta


def _affine_norms(block: AdainResBlk1d) -> AdainResBlk1d:
    for name in ("norm1", "norm2"):
        fc = getattr(block, name).fc
        setattr(block, name, AffineAdaIN1d(fc.weight.shape[1], fc.weight.shape[0] // 2))
    return block


def _block(dim_in, dim_out, style_dim, upsample=False):
    return _affine_norms(
        AdainResBlk1d(
            dim_in,
            dim_out,
            style_dim,
            upsample=True if upsample else "none",
            conv_type=mx.conv1d,
        )
    )


@mx.compile
def _lstm_step(xt, h, c, Wh):
    """one step of both directions at once: xt [2, 4H] (input projection), h / c [2, H], Wh [2, H, 4H]"""
    g = xt + (h[:, None, :] @ Wh)[:, 0]
    i, f, z, o = mx.split(g, 4, axis=-1)
    c = mx.sigmoid(f) * c + mx.sigmoid(i) * mx.tanh(z)
    return mx.sigmoid(o) * mx.tanh(c), c


class FastBiLSTM(LSTM):
    """Kokoro's bidirectional LSTM (same parameters), batch 1: both directions advance together through one compiled
    step, which is several times faster than two Python loops of separate ops."""

    def __call__(self, x, *args, **kwargs):
        if x.ndim == 3 and x.shape[0] != 1:
            return super().__call__(x, *args, **kwargs)
        x = x.reshape(-1, x.shape[-1])
        xf = x @ self.Wx_forward.T + (self.bias_ih_forward + self.bias_hh_forward)
        xb = x[::-1] @ self.Wx_backward.T + (
            self.bias_ih_backward + self.bias_hh_backward
        )
        X = mx.stack([xf, xb])  # [2, T, 4H]
        Wh = mx.stack([self.Wh_forward.T, self.Wh_backward.T])
        h = mx.zeros((2, self.hidden_size))
        c = mx.zeros((2, self.hidden_size))
        outs = []
        for t in range(X.shape[1]):
            h, c = _lstm_step(X[:, t], h, c, Wh)
            outs.append(h)
        H = mx.stack(outs, axis=1)  # [2, T, H]
        out = mx.concatenate([H[0], H[1][::-1]], axis=-1)[None]
        return out, None


def _fast_lstms(module: nn.Module):
    for _, m in module.named_modules():
        if type(m) is LSTM:
            m.__class__ = FastBiLSTM


class Trunk(nn.Module):
    """Kokoro's decoder AdaIN trunk (widths configurable, external 16-d voice row) feeding the source-filter head."""

    def __init__(
        self,
        dim_in,
        width=256,
        up_channels=128,
        style_dim=16,
        head: Optional[dict] = None,
    ):
        super().__init__()
        self.encode = _block(dim_in + 2, width, style_dim)
        self.decode = [_block(width + 2 + 64, width, style_dim) for _ in range(3)]
        self.decode.append(
            _block(width + 2 + 64, up_channels, style_dim, upsample=True)
        )
        self.F0_conv = ConvWeighted(1, 1, kernel_size=3, stride=2, padding=1)
        self.N_conv = ConvWeighted(1, 1, kernel_size=3, stride=2, padding=1)
        self.asr_res = [ConvWeighted(dim_in, 64, kernel_size=1, padding=0)]
        self.head = SourceFilterHead(in_ch=up_channels, **(head or {}))

    def __call__(self, asr, F0_curve, N, s, noise=None):
        F0 = self.F0_conv(F0_curve[:, :, None], mx.conv1d).swapaxes(2, 1)
        Nn = self.N_conv(N[:, :, None], mx.conv1d).swapaxes(2, 1)
        x = self.encode(mx.concatenate([asr, F0, Nn], axis=1), s)
        asr_res = self.asr_res[0](asr.swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)
        res = True
        for block in self.decode:
            if res:
                x = mx.concatenate([x, asr_res, F0, Nn], axis=1)
            x = block(x, s)
            if block.upsample_type != "none":
                res = False
        return self.head(x.swapaxes(2, 1), F0_curve[0], noise)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab = config.vocab
        c = config
        if c.text_side == "ettin":
            self.ettin = Ettin(EttinArgs(**(c.ettin or {})))
            self.student = StudentTextSide(
                n_token=c.n_token, out=c.hidden_dim, **(c.student or {})
            )
        else:
            self.bert = CustomAlbert(AlbertModelArgs(vocab_size=c.n_token, **c.plbert))
            for (
                group
            ) in self.bert.encoder.albert_layer_groups:  # ALBERT's default gelu_new
                for layer in group.albert_layers:
                    layer.activation = nn.GELU(approx="tanh")
            self.bert_encoder = nn.Linear(self.bert.config.hidden_size, c.hidden_dim)
        self.predictor = ProsodyPredictor(
            style_dim=c.style_dim,
            d_hid=c.hidden_dim,
            nlayers=c.n_layer,
            max_dur=c.max_dur,
        )
        for blocks in (self.predictor.F0, self.predictor.N):
            for b in blocks:
                _affine_norms(b)
        self.text_encoder = TextEncoder(
            channels=c.hidden_dim,
            kernel_size=c.text_encoder_kernel_size,
            depth=c.n_layer,
            n_symbols=c.n_token,
        )
        d = dict(c.decoder or {})
        self.decoder = Trunk(dim_in=c.hidden_dim, **d)
        self._voices: Dict[str, dict] = {}
        self._frontend = None
        self.text_device = mx.cpu
        _fast_lstms(self)

    @property
    def sample_rate(self):
        return self.config.sample_rate

    def sanitize(self, weights):
        # weights are written in this module's layout by convert.py
        return {k: v for k, v in weights.items() if not k.startswith("bert.pooler")} | {
            k: v for k, v in self._pooler_defaults().items() if k not in weights
        }

    def _pooler_defaults(self):
        if "bert" not in self:
            return {}
        h = self.bert.config.hidden_size
        return {
            "bert.pooler.weight": mx.zeros((h, h)),
            "bert.pooler.bias": mx.zeros((h,)),
        }

    # ---------------------------------------------------------------- voices
    def voice_names(self) -> List[str]:
        return [v["name"].lower() for v in self.config.voices]

    def _voice_entry(self, voice: Optional[str]) -> dict:
        voices = self.config.voices
        if voice is None:
            return voices[0]
        k = str(voice).strip().lower()
        for v in voices:
            if k in (v["id"].lower(), v["name"].lower()):
                return v
        raise ValueError(
            f"unknown voice {voice!r}; voices: {', '.join(self.voice_names())}"
        )

    def load_voice(self, voice: Optional[str] = None) -> dict:
        meta = self._voice_entry(voice)
        key = meta["name"].lower()
        if key not in self._voices:
            path = Path(self.config.model_path) / "voices" / f"{key}.safetensors"
            self._voices[key] = {"meta": meta, **mx.load(str(path))}
        return self._voices[key]

    # ---------------------------------------------------------------- synthesis
    def _durations(self, d_en, s, speed):
        T = d_en.shape[-1]
        mask = mx.zeros((1, T), dtype=mx.bool_)
        lengths = mx.array([T])
        d = self.predictor.text_encoder(d_en, s, lengths, mask)
        x, _ = self.predictor.lstm(d)
        dur = mx.sigmoid(self.predictor.duration_proj(x)).sum(-1)[0] / speed
        n = np.maximum(np.round(np.array(dur)), 1).astype(np.int64)
        return d, n, mask, lengths

    @staticmethod
    def _end_clamp(F0, v):
        """per-voice end-of-sentence pitch clamp + unvoicing floor (the release graphs' kale_clamp / unvoice nodes)"""
        f = F0[0]
        idx = mx.arange(f.shape[0]).astype(mx.float32)
        voiced = f > v["clamp_thr"].reshape(())
        last = mx.max(mx.where(voiced, idx, -1e9))
        tail = (idx >= last - v["clamp_tail"].reshape(())) & (idx <= last)
        ok = voiced.sum() >= 5
        f = mx.where(voiced & tail & ok, mx.maximum(f, v["pitch_floor"].reshape(())), f)
        f = mx.where(f < v["unvoice_hz"].reshape(()), 0.0, f)
        return f[None]

    def _text_side(self, ids, v, speed, ettin):
        """ids -> (asr [1, C, F], F0 [1, 2F], N [1, 2F]): text side, durations, pitch / loudness, end clamp"""
        x = mx.array([[0, *ids, 0]])
        if self.config.text_side == "ettin":
            if ettin is None:
                raise ValueError(
                    "kale-tts-1-tiny needs the sentence's Ettin inputs; use generate()"
                )
            h = self.ettin(mx.array([ettin["ids"]]))
            wfeat = (
                mx.array(
                    np.asarray(ettin["wavg"], np.float32).reshape(
                        1, ettin["W"], ettin["S"]
                    )
                )
                @ h
            )
            w = list(ettin["widx"][: len(ids)])
            d_en = self.student(x, wfeat, mx.array([[w[0], *w, w[-1]]])).transpose(
                0, 2, 1
            )
            s = v["style"][min(len(ids), v["style"].shape[0]) - 1][None, 128:]
        else:
            bert_out, _ = self.bert(x, attention_mask=mx.ones_like(x))
            d_en = self.bert_encoder(bert_out).transpose(0, 2, 1)
            s = v["prosody"].reshape(1, -1)
        d, n, mask, lengths = self._durations(d_en, s, speed)
        aln = (
            mx.arange(x.shape[1])[:, None]
            == mx.array(np.repeat(np.arange(x.shape[1]), n))[None, :]
        )
        aln = aln.astype(mx.float32)[None]
        F0, N = self.predictor.F0Ntrain(d.transpose(0, 2, 1) @ aln, s)
        F0 = self._end_clamp(F0, v)
        asr = self.text_encoder(x, lengths, mask) @ aln
        mx.eval(asr, F0, N)
        return asr, F0, N

    def synthesize(
        self,
        ids: List[int],
        voice: Optional[str] = None,
        speed: float = 1.0,
        ettin: Optional[dict] = None,
        noise: Optional[mx.array] = None,
    ) -> mx.array:
        """one sentence of Kokoro-table ids (no pads) -> float32 audio [S] at 24 kHz. Models with the Ettin text side need
        the sentence's Ettin inputs (`frontend.ettin_inputs`).

        The text side (small, sequential LSTMs) runs on `text_device` (default: CPU), where its rounding of the
        durations matches the release graphs exactly; the decoder runs on the default device.
        """
        v = self.load_voice(voice)
        ids = list(ids)[:510]
        with mx.stream(self.text_device):
            asr, F0, N = self._text_side(ids, v, speed, ettin)
        return self.decoder(asr, F0, N, v["sound"].reshape(1, -1), noise)

    # ---------------------------------------------------------------- text
    @property
    def frontend(self):
        if self._frontend is None:
            from .frontend import KaleFrontEnd

            self._frontend = KaleFrontEnd(
                Path(self.config.model_path),
                self.vocab,
                self.config.phonemes,
                ettin=self.config.text_side == "ettin",
            )
        return self._frontend

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        lang_code: Optional[str] = None,
        gap_s: float = 0.12,
        **kwargs,
    ):
        """text -> one GenerationResult per sentence (the release front end splits sentences, reads numbers / dates /
        terms, and picks the British front end for British voices)."""
        meta = self._voice_entry(voice)
        accent = meta.get("accent", "en-us")
        fe = self.frontend
        prev = ""
        for i, sent in enumerate(fe.sentences(text, accent)):
            ids = fe.ids(sent["ipa"], accent)
            if not ids:
                continue
            e = (
                fe.ettin_inputs(sent["text"], prev, ids, accent)
                if self.config.text_side == "ettin"
                else None
            )
            t0 = time.perf_counter()
            audio = self.synthesize(ids, meta["name"], speed, e)
            mx.eval(audio)
            dt = time.perf_counter() - t0
            prev = sent["text"]
            secs = audio.shape[0] / self.sample_rate
            yield GenerationResult(
                audio=audio,
                samples=audio.shape[0],
                sample_rate=self.sample_rate,
                segment_idx=i,
                token_count=len(ids),
                audio_duration=f"{int(secs // 3600):02d}:{int(secs // 60 % 60):02d}:{int(secs % 60):02d}.{int(secs % 1 * 1000):03d}",
                real_time_factor=round(dt / secs, 4) if secs else 0,
                prompt={
                    "tokens": len(ids),
                    "tokens-per-sec": round(len(ids) / dt, 2) if dt else 0,
                },
                audio_samples={
                    "samples": audio.shape[0],
                    "samples-per-sec": round(audio.shape[0] / dt, 2) if dt else 0,
                },
                processing_time_seconds=dt,
                peak_memory_usage=mx.get_peak_memory() / 1e9,
            )
