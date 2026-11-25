"""kale-tts-1-tiny's text side: the frozen Ettin-encoder-17m (a ModernBERT encoder, MIT, jhu-clsp) reads the sentence's
words; a small student (phoneme embedding -> 3 conv blocks -> + Ettin word features per phoneme -> 2 transformer
layers) replaces Kokoro's ALBERT and feeds Kokoro's prosody predictor."""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class EttinArgs:
    vocab_size: int = 50368
    hidden_size: int = 256
    intermediate_size: int = 384
    num_hidden_layers: int = 7
    num_attention_heads: int = 4
    global_attn_every_n_layers: int = 3
    local_attention: int = 128
    global_rope_theta: float = 160000.0
    local_rope_theta: float = 160000.0
    norm_eps: float = 1e-5


class EttinAttention(nn.Module):
    def __init__(self, a: EttinArgs, theta: float):
        super().__init__()
        self.heads = a.num_attention_heads
        self.head_dim = a.hidden_size // a.num_attention_heads
        self.Wqkv = nn.Linear(a.hidden_size, 3 * a.hidden_size, bias=False)
        self.Wo = nn.Linear(a.hidden_size, a.hidden_size, bias=False)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=theta)

    def __call__(self, x, mask):
        B, S, _ = x.shape
        qkv = (
            self.Wqkv(x)
            .reshape(B, S, 3, self.heads, self.head_dim)
            .transpose(2, 0, 3, 1, 4)
        )
        q, k, v = self.rope(qkv[0]), self.rope(qkv[1]), qkv[2]
        y = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim**-0.5, mask=mask
        )
        return self.Wo(y.transpose(0, 2, 1, 3).reshape(B, S, -1))


class EttinMLP(nn.Module):
    def __init__(self, a: EttinArgs):
        super().__init__()
        self.Wi = nn.Linear(a.hidden_size, 2 * a.intermediate_size, bias=False)
        self.Wo = nn.Linear(a.intermediate_size, a.hidden_size, bias=False)

    def __call__(self, x):
        inp, gate = mx.split(self.Wi(x), 2, axis=-1)
        return self.Wo(nn.gelu(inp) * gate)


class EttinLayer(nn.Module):
    def __init__(self, a: EttinArgs, layer_id: int):
        super().__init__()
        self.is_global = layer_id % a.global_attn_every_n_layers == 0
        if layer_id != 0:
            self.attn_norm = nn.LayerNorm(a.hidden_size, eps=a.norm_eps, bias=False)
        self.attn = EttinAttention(
            a, a.global_rope_theta if self.is_global else a.local_rope_theta
        )
        self.mlp_norm = nn.LayerNorm(a.hidden_size, eps=a.norm_eps, bias=False)
        self.mlp = EttinMLP(a)

    def __call__(self, x, global_mask, local_mask):
        h = self.attn_norm(x) if "attn_norm" in self else x
        x = x + self.attn(h, global_mask if self.is_global else local_mask)
        return x + self.mlp(self.mlp_norm(x))


class EttinEmbeddings(nn.Module):
    def __init__(self, a: EttinArgs):
        super().__init__()
        self.tok_embeddings = nn.Embedding(a.vocab_size, a.hidden_size)
        self.norm = nn.LayerNorm(a.hidden_size, eps=a.norm_eps, bias=False)

    def __call__(self, ids):
        return self.norm(self.tok_embeddings(ids))


class Ettin(nn.Module):
    """ModernBERT encoder without its final norm; returns the mean of the last two layers' hidden states."""

    def __init__(self, a: EttinArgs):
        super().__init__()
        self.args = a
        self.embeddings = EttinEmbeddings(a)
        self.layers = [EttinLayer(a, i) for i in range(a.num_hidden_layers)]

    def __call__(self, ids: mx.array) -> mx.array:
        S = ids.shape[1]
        pos = mx.arange(S)
        local = mx.abs(pos[:, None] - pos[None, :]) <= self.args.local_attention // 2
        local_mask = mx.where(local, 0.0, -mx.inf).astype(mx.float32)
        x = self.embeddings(ids)
        prev = x
        for layer in self.layers:
            prev = x
            x = layer(x, None, local_mask)
        return (x + prev) / 2


class ConvBlock(nn.Module):
    def __init__(self, c: int, k: int = 5):
        super().__init__()
        self.conv = nn.Conv1d(c, c, k, padding=k // 2)
        self.norm = nn.LayerNorm(c)

    def __call__(self, x):
        return x + nn.gelu(self.norm(self.conv(x)))


class TFBlock(nn.Module):
    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.n1 = nn.LayerNorm(c)
        self.qkv = nn.Linear(c, 3 * c)
        self.o = nn.Linear(c, c)
        self.n2 = nn.LayerNorm(c)
        self.ff = [nn.Linear(c, 4 * c), nn.GELU(), nn.Linear(4 * c, c)]

    def __call__(self, x):
        B, T, C = x.shape
        hd = C // self.heads
        q, k, v = (
            self.qkv(self.n1(x))
            .reshape(B, T, 3, self.heads, hd)
            .transpose(2, 0, 3, 1, 4)
        )
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=hd**-0.5)
        x = x + self.o(y.transpose(0, 2, 1, 3).reshape(B, T, C))
        h = self.n2(x)
        for f in self.ff:
            h = f(h)
        return x + h


class StudentTextSide(nn.Module):
    """phoneme ids [1, T] + Ettin word features [1, W, 256] + word index per phoneme [1, T] -> [1, T, out]"""

    def __init__(
        self,
        n_token=178,
        c=256,
        nconv=3,
        ntf=2,
        wdim=256,
        out=512,
        heads=4,
        max_len=512,
    ):
        super().__init__()
        self.emb = nn.Embedding(n_token, c)
        self.convs = [ConvBlock(c) for _ in range(nconv)]
        self.wproj = [nn.Linear(wdim, c), nn.GELU(), nn.Linear(c, c)]
        self.pos = nn.Embedding(max_len, c)
        self.tf = [TFBlock(c, heads) for _ in range(ntf)]
        self.out = nn.Linear(c, out)

    def __call__(self, ids, wfeat, widx):
        h = self.emb(ids)
        for b in self.convs:
            h = b(h)
        w = wfeat[0][widx[0]][None]
        for f in self.wproj:
            w = f(w)
        h = h + w + self.pos(mx.arange(ids.shape[1]))[None]
        for b in self.tf:
            h = b(h)
        return self.out(h)
