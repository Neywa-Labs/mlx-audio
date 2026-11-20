"""Frame-rate source-filter head of kale-tts-1-tiny / -nano.

The head replaces Kokoro's upsampling iSTFTNet generator. A few ConvNeXt blocks at the trunk's frame rate predict,
per STFT frame (n_fft 600, hop 150), a smooth filter (gain + phase at G mel-spaced points) applied to the STFT of a
64-harmonic sine bank built from the pitch curve, a pulse shift, gains for plain and pitch-pulsing noise, and a
short-window noise burst path (n_fft 150, hop 75). Port of SourceFilterHead2 (paradee, Apache-2.0) as exported in
the kale-tts-1 release graphs.
"""

import math
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn
import numpy as np

SAMPLE_RATE = 24000
F0_HOP = 300  # samples per pitch frame (80 frames a second)


@lru_cache(maxsize=None)
def _warp_matrix(nb: int, points: int) -> mx.array:
    """[nb, points] linear interpolation from mel-spaced points to linear STFT bins (paradee warp_matrix)."""
    f = np.linspace(0, SAMPLE_RATE / 2, nb, dtype=np.float32)
    mel = lambda x: np.log1p(np.asarray(x, np.float32) / np.float32(700))  # noqa: E731
    pos = mel(f) / mel(np.float32(SAMPLE_RATE / 2)) * np.float32(points - 1)
    i = np.clip(np.floor(pos), 0, points - 2).astype(np.int64)
    w = (pos - i).astype(np.float32)
    m = np.zeros((nb, points), np.float32)
    m[np.arange(nb), i] = 1 - w
    m[np.arange(nb), i + 1] += w
    return mx.array(m)


@lru_cache(maxsize=None)
def _hann(n: int) -> mx.array:
    """periodic Hann window (torch.hann_window default)"""
    return mx.array(
        (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float32)
    )


def stft(x: mx.array, n_fft: int, hop: int) -> mx.array:
    """x [S] -> complex [n_fft // 2 + 1, frames]; Hann window, centred, reflect padding (torch.stft defaults)."""
    p = n_fft // 2
    x = mx.concatenate([x[1 : p + 1][::-1], x, x[-p - 1 : -1][::-1]])
    frames = (x.shape[0] - n_fft) // hop + 1
    idx = mx.arange(frames)[:, None] * hop + mx.arange(n_fft)[None, :]
    return mx.fft.rfft(x[idx] * _hann(n_fft), axis=-1).T


def istft(spec: mx.array, n_fft: int, hop: int, length: int) -> mx.array:
    """complex [n_fft // 2 + 1, frames] -> [length]; inverse of stft() (overlap-add, window-square normalised)."""
    assert n_fft % hop == 0, "overlap-add assumes n_fft is a multiple of hop"
    r = n_fft // hop
    w = _hann(n_fft)
    frames = mx.fft.irfft(spec.T, n=n_fft, axis=-1) * w  # [F, n_fft]
    nf = frames.shape[0]
    total = (nf - 1) * hop + n_fft
    chunks = frames.reshape(nf, r, hop)
    env_chunks = (w * w).reshape(r, hop)
    y = mx.zeros((total,))
    env = mx.zeros((total,))
    for j in range(r):
        pad = [(j * hop, (r - 1 - j) * hop)]
        y = y + mx.pad(chunks[:, j, :].reshape(-1), pad)
        env = env + mx.pad(mx.broadcast_to(env_chunks[j], (nf, hop)).reshape(-1), pad)
    y = y / mx.maximum(env, 1e-8)
    p = n_fft // 2
    return y[p : p + length]


def sine_bank(f0: mx.array, harmonics: int = 64):
    """f0 [n] (80 fps, 0 = unvoiced) -> (excitation [S], cycles [S], voiced [S]); S = n * 300.
    Fundamental phase accumulated per frame, wrapped, linearly interpolated between frame centres; harmonics above
    11 kHz and unvoiced samples muted (float32, as the release graph)."""
    n = f0.shape[0]
    S = n * F0_HOP
    r = f0 * (F0_HOP / SAMPLE_RATE)
    c = mx.cumsum(r)
    c = c - mx.floor(c)
    d = mx.concatenate([r[1:], mx.zeros((1,))])
    t = mx.arange(S).astype(mx.float32)
    p = (t + 0.5) / F0_HOP - 0.5
    i = mx.clip(mx.floor(p), 0, n - 1).astype(mx.int32)
    frac = mx.clip(p - i.astype(mx.float32), 0, 1)
    cyc = c[i] + frac * d[i]
    f0s = mx.repeat(f0, F0_HOP)
    uv = (f0s > 10).astype(mx.float32)
    k = mx.arange(1, harmonics + 1).astype(mx.float32)
    amp = (f0s[:, None] * k[None, :] < 11000).astype(mx.float32) * uv[:, None]
    e = (mx.sin(2 * math.pi * k[None, :] * cyc[:, None]) * amp).sum(-1) * 0.1
    return e, cyc, uv


class ConvNeXt(nn.Module):
    def __init__(self, dim: int, mult: int = 3):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, dim * mult)
        self.pw2 = nn.Linear(dim * mult, dim)
        self.gamma = mx.full((dim,), 1e-2)

    def __call__(self, x):  # x [B, T, C]
        h = self.norm(self.dw(x))
        return x + self.gamma * self.pw2(nn.gelu(self.pw1(h)))


class SourceFilterHead(nn.Module):
    def __init__(
        self,
        in_ch: int = 128,
        dim: int = 160,
        blocks: int = 4,
        n_fft: int = 600,
        hop: int = 150,
        harmonics: int = 64,
        mod: int = 2,
        points: int = 80,
        burst: bool = True,
        burst_points: int = 24,
    ):
        super().__init__()
        self.n_fft, self.hop, self.harmonics, self.mod, self.points = (
            n_fft,
            hop,
            harmonics,
            mod,
            points,
        )
        self.up = F0_HOP // hop
        self.nb = n_fft // 2 + 1
        self.channels = (
            4 + 2 * mod
        )  # harmonic log-gain, phase cos, phase sin, noise log-gain, pulsing-noise gains
        self.burst, self.burst_points = burst, burst_points
        self.inp = nn.Conv1d(in_ch, dim, 7, padding=3)
        self.norm0 = nn.LayerNorm(dim)
        self.blocks = [ConvNeXt(dim) for _ in range(blocks)]
        self.norm1 = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, self.up * (self.channels * points + 1))
        if burst:
            self.out_b = nn.Linear(dim, 4 * burst_points)

    def __call__(self, x: mx.array, f0: mx.array, noise: mx.array = None) -> mx.array:
        """x [1, n, in_ch] trunk features (channels last), f0 [n] Hz -> waveform [n * 300]"""
        n = x.shape[1]
        T = n * self.up
        S = n * F0_HOP
        h = self.norm0(self.inp(x))
        for b in self.blocks:
            h = b(h)
        hn = self.norm1(h)
        o = self.out(hn).reshape(T, self.channels * self.points + 1)
        o = mx.concatenate([o, o[-1:]], 0)  # centred STFT has one more frame
        M = _warp_matrix(self.nb, self.points)
        g = (o[:, :-1].reshape(T + 1, self.channels, self.points) @ M.T).transpose(
            1, 2, 0
        )  # [C, nb, T+1]
        gh = mx.exp(mx.minimum(g[0], 6))
        gn = mx.exp(mx.minimum(g[3], 6))
        um = mx.maximum(mx.sqrt(g[1] ** 2 + g[2] ** 2), 1e-4)
        ur, ui = g[1] / um, g[2] / um

        e, cyc, uv = sine_bank(f0, self.harmonics)
        nz = mx.random.normal((S,)) if noise is None else noise.reshape(-1)[:S]
        E = stft(e, self.n_fft, self.hop)
        Nz = stft(nz, self.n_fft, self.hop)
        j = 0
        for m in range(1, self.mod + 1):
            for fn in (mx.cos, mx.sin):
                Nz = Nz + g[4 + j] * stft(
                    nz * uv * fn(2 * math.pi * m * cyc), self.n_fft, self.hop
                )
                j += 1

        f0f = mx.maximum(
            mx.concatenate([mx.repeat(f0, self.up), f0[-1:]]), 60
        )  # per STFT frame
        fr = mx.array(np.linspace(0, SAMPLE_RATE / 2, self.nb, dtype=np.float32))
        ang = (
            2
            * math.pi
            * (fr[:, None] / f0f[None, :])
            * (0.5 * mx.tanh(o[:, -1]))[None, :]
        )
        Hr = gh * (ur * mx.cos(ang) - ui * mx.sin(ang))
        Hi = gh * (ur * mx.sin(ang) + ui * mx.cos(ang))
        H = Hr + 1j * Hi
        y = istft(E * H + Nz * gn, self.n_fft, self.hop, S)
        if self.burst:
            Mb = _warp_matrix(150 // 2 + 1, self.burst_points)
            gb = mx.exp(
                mx.minimum(self.out_b(hn).reshape(n * 4, self.burst_points) @ Mb.T, 6)
            ).T
            gb = mx.concatenate([gb, gb[:, -1:]], -1)
            Nb = stft(nz[::-1], 150, 75)
            y = y + istft(Nb * gb, 150, 75, S)
        return y
