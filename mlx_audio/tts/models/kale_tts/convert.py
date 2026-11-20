"""Convert the kale-tts-1 release files (Hugging Face: KaleComputer/kale-tts-1-{base,tiny,nano}) to MLX.

  tiny / nano: the float32 release graph (`kale-tts-1-{tiny,nano}.fp32.onnx`) -> model.safetensors in this package's layout + config.json,
               voices (`voices/<id>.bin` style table + `.extra.bin` voice inputs) -> voices/<name>.safetensors, and the
               release text front end (`python/kale_frontend.js`, `frontend/words/`) copied alongside.
  base:        Kokoro format (`kale-tts-1-base.pth`, PyTorch) -> a `kokoro` model directory (model.safetensors, config.json,
               voices/<lang>_<name>.safetensors); needs torch for reading the .pth.

`load_model("KaleComputer/kale-tts-1-nano")` runs this once through `prepare_model_path` and caches the result.

    python -m mlx_audio.tts.models.kale_tts.convert --hf-path KaleComputer/kale-tts-1-nano --mlx-path ./kale-tts-1-nano-mlx
"""

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx
import numpy as np

ONNX_GLOB = "*.fp32.onnx"  # kale-tts-1-nano.fp32.onnx / kale-tts-1-tiny.fp32.onnx
BASE_FILE = "kale-tts-1-base.pth"
SOURCE_PATTERNS = [
    "*.json",
    ONNX_GLOB,
    BASE_FILE,
    "voices/*",
    "python/kale_frontend.js",
    "frontend/words/*.tsv",
    "frontend/words/*.bin",
    "frontend/words/en-gb/*",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
]
_CONV_WN = re.compile(
    r"(\.(conv1|conv2|conv1x1|pool)|^text_encoder\.cnn\.\d+\.0|^decoder\.(F0_conv|N_conv|asr_res\.0))\.weight$"
)
_CONV_PLAIN = re.compile(
    r"(F0_proj|N_proj|head\.inp|head\.blocks\.\d+\.dw|student\.convs\.\d+\.conv)\.weight$"
)
_LSTM_GATES = [0, 2, 3, 1]
_ALBERT_PARTS = {
    "query",
    "key",
    "value",
    "dense",
    "ffn",
    "ffn_output",
}  # ONNX i, o, f, c -> torch / MLX i, f, g(c), o


_NAMED = {
    "ettin": [
        ("et.", "ettin."),
        ("st.", "student."),
        ("pred.", "predictor."),
        ("te.", "text_encoder."),
        ("dec.", "decoder."),
    ],
    "albert": [
        ("rt.predictor.", "predictor."),
        ("rt.bert_encoder.", "bert_encoder."),
        ("rt.bert.", "bert."),
        ("ra.te.", "text_encoder."),
        ("ra.dec.", "decoder."),
    ],
}
# module paths of graph nodes (anonymous MatMul / LSTM weights): exact names first, then prefixes
_NODE_EXACT = {
    "lstm": "predictor.lstm",
    "shared": "predictor.shared",
    "lstm_1": "text_encoder.lstm",
    "bert_encoder": "bert_encoder",
}
_NODE_PREFIX = [
    ("lstms.", "predictor.text_encoder.lstms."),
    ("duration_proj.", "predictor.duration_proj."),
    ("head.", "decoder.head."),
    ("bert.", "bert."),
    ("et.", "ettin."),
    ("st.", "student."),
]


def _rename_named(name: str, text_side: str) -> Optional[str]:
    for src, dst in _NAMED[text_side]:
        if name.startswith(src):
            return dst + name[len(src) :]
    return None


def _rename_node(path: str) -> Optional[str]:
    if path in _NODE_EXACT:
        return _NODE_EXACT[path]
    for src, dst in _NODE_PREFIX:
        if path.startswith(src):
            return dst + path[len(src) :]
    return None


def _consumer_paths(graph):
    """anonymous initializer -> module path of the node that reads it (through Identity / Cast / Transpose)"""
    cons = {}
    for n in graph.node:
        for k, x in enumerate(n.input):
            cons.setdefault(x, []).append((n, k))

    def follow(x, depth=0):
        for n, k in cons.get(x, []):
            if n.op_type in ("Identity", "Cast", "Transpose") and depth < 3:
                yield from follow(n.output[0], depth + 1)
            else:
                yield n, k

    out = {}
    for init in graph.initializer:
        if not init.name.startswith("onnx::"):
            continue
        hits = list(follow(init.name))
        if not hits:
            continue
        n, k = hits[0]
        parts = n.name.removeprefix("/g").strip("/").split("/")[:-1]
        # ALBERT's shared layer runs 6 times: ".../attention/query_5/MatMul" is the same weight as ".../query/MatMul"
        parts = [
            re.sub(r"_\d+$", "", p) if re.sub(r"_\d+$", "", p) in _ALBERT_PARTS else p
            for p in parts
            if p
        ]
        # nn.Sequential children export as "/wproj/wproj.0": keep the child only
        parts = [
            p
            for i, p in enumerate(parts)
            if not (i + 1 < len(parts) and parts[i + 1].startswith(p + "."))
        ]
        out[init.name] = (".".join(parts), n.op_type, k)
    return out


def _wn(w: np.ndarray):
    """plain conv weight [out, in, k] -> ConvWeighted's (weight_v [out, k, in], weight_g [out, 1, 1])"""
    v = w.transpose(0, 2, 1)
    return v, np.sqrt(
        (v.astype(np.float64) ** 2).sum(axis=(1, 2), keepdims=True)
    ).astype(np.float32)


def _onnx_weights(graph, text_side: str) -> Dict[str, np.ndarray]:
    from onnx import numpy_helper

    anon = _consumer_paths(graph)
    out: Dict[str, np.ndarray] = {}
    for init in graph.initializer:
        a = numpy_helper.to_array(init)
        if a.dtype != np.float32:
            continue
        if init.name in anon:
            mod, op, k = anon[init.name]
            mod = _rename_node(mod)
            if mod is None:
                continue
            if op == "MatMul":
                out[f"{mod}.weight"] = a.T
            elif op == "LSTM":
                H = (
                    a.shape[-1]
                    if k == 2
                    else a.shape[1] // 4 if k == 1 else a.shape[1] // 8
                )
                for d, side in enumerate(("forward", "backward")):
                    if k in (1, 2):
                        w = a[d].reshape(4, H, -1)[_LSTM_GATES].reshape(4 * H, -1)
                        out[f"{mod}.{'Wx' if k == 1 else 'Wh'}_{side}"] = w
                    else:
                        b = a[d].reshape(2, 4, H)[:, _LSTM_GATES].reshape(2, 4 * H)
                        out[f"{mod}.bias_ih_{side}"], out[f"{mod}.bias_hh_{side}"] = (
                            b[0],
                            b[1],
                        )
            continue
        name = _rename_named(init.name, text_side)
        if name is None or "/" in init.name:
            continue
        name = re.sub(r"\.gamma$", ".weight", name) if ".cnn." in name else name
        name = re.sub(r"\.beta$", ".bias", name)
        if _CONV_WN.search(name):
            v, g = _wn(a)
            (
                out[name[: -len("weight")] + "weight_v"],
                out[name[: -len("weight")] + "weight_g"],
            ) = (v, g)
        elif _CONV_PLAIN.search(name):
            out[name] = a.transpose(0, 2, 1)
        else:
            out[name] = a
    return out


def _config_from_weights(
    w: Dict[str, np.ndarray], text_side: str, vj: dict, vocab: dict
) -> dict:
    hd = w["predictor.duration_proj.linear_layer.weight"].shape[1]
    sd = w["predictor.text_encoder.lstms.1.fc.weight"].shape[1]
    cfg = {
        "model_type": "kale_tts",
        "text_side": text_side,
        "n_token": w["text_encoder.embedding.weight"].shape[0],
        "hidden_dim": hd,
        "style_dim": sd,
        "n_layer": len(
            {k.split(".")[2] for k in w if k.startswith("text_encoder.cnn.")}
        ),
        "max_dur": w["predictor.duration_proj.linear_layer.weight"].shape[0],
        "text_encoder_kernel_size": w["text_encoder.cnn.0.0.weight_v"].shape[1],
        "decoder": {
            "width": w["decoder.encode.conv1.weight_v"].shape[0],
            "up_channels": w["decoder.decode.3.conv1.weight_v"].shape[0],
            "style_dim": w["decoder.encode.norm1.fc.weight"].shape[1],
            "head": {
                "dim": w["decoder.head.inp.weight"].shape[0],
                "blocks": len(
                    {k.split(".")[3] for k in w if k.startswith("decoder.head.blocks.")}
                ),
                "burst": "decoder.head.out_b.weight" in w,
            },
        },
        "vocab": vocab,
        "phonemes": vj.get("phonemes", "ipa"),
        "voices": [
            {k: v[k] for k in ("id", "name", "accent", "description") if k in v}
            for v in vj["voices"]
        ],
        "sample_rate": 24000,
    }
    if text_side == "ettin":
        cfg["variant"] = "tiny"
        cfg["ettin"] = {
            "vocab_size": w["ettin.embeddings.tok_embeddings.weight"].shape[0],
            "hidden_size": w["ettin.embeddings.tok_embeddings.weight"].shape[1],
            "intermediate_size": w["ettin.layers.0.mlp.Wo.weight"].shape[1],
            "num_hidden_layers": len(
                {k.split(".")[2] for k in w if k.startswith("ettin.layers.")}
            ),
        }
        cfg["student"] = {
            "c": w["student.emb.weight"].shape[1],
            "ntf": len({k.split(".")[2] for k in w if k.startswith("student.tf.")}),
        }
    else:
        cfg["variant"] = "nano"
        h = w["bert.encoder.embedding_hidden_mapping_in.weight"]
        cfg["plbert"] = {
            "num_hidden_layers": 6,
            "num_attention_heads": 4,
            "hidden_size": h.shape[0],
            "intermediate_size": w[
                "bert.encoder.albert_layer_groups.0.albert_layers.0.ffn.weight"
            ].shape[0],
            "max_position_embeddings": w[
                "bert.embeddings.position_embeddings.weight"
            ].shape[0],
            "embedding_size": h.shape[1],
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
        }
    return cfg


def _albert_passes(graph) -> int:
    return len(
        {
            m.group(1)
            for n in graph.node
            if (m := re.search(r"attention/query(_\d+)?/MatMul", n.name))
        }
    )


def _albert_heads(graph, hidden: int) -> int:
    from onnx import numpy_helper

    for n in graph.node:
        if "attention" in n.name and n.op_type == "Reshape" and "/g/bert/" in n.name:
            for x in graph.node:
                if x.op_type == "Constant" and x.output[0] == n.input[1]:
                    s = numpy_helper.to_array(x.attribute[0].t)
                    if s.size == 4 and s[-1] > 0 and hidden % s[-1] == 0:
                        return hidden // int(s[-1])
    return 4


def _onnx_file(src: Path) -> Optional[Path]:
    """the float32 release graph in a release folder, if there is one"""
    found = sorted(Path(src).glob(ONNX_GLOB))
    return found[0] if found else None


def convert_onnx_model(src: Path, dst: Path) -> Path:
    """tiny / nano release folder -> MLX folder"""
    import onnx

    src, dst = Path(src), Path(dst)
    graph = onnx.load(str(_onnx_file(src))).graph
    text_side = "ettin" if any(i.name == "ettin_ids" for i in graph.input) else "albert"
    w = _onnx_weights(graph, text_side)
    vj = json.loads((src / "voices.json").read_text())
    vocab = json.loads((src / "tokenizer.json").read_text())["model"]["vocab"]
    cfg = _config_from_weights(w, text_side, vj, vocab)
    if text_side == "albert":
        cfg["plbert"]["num_hidden_layers"] = _albert_passes(graph) or 6
        cfg["plbert"]["num_attention_heads"] = _albert_heads(
            graph, cfg["plbert"]["hidden_size"]
        )
    dst.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(dst / "model.safetensors"),
        {k: mx.array(v) for k, v in w.items()},
        metadata={"format": "mlx"},
    )
    (dst / "config.json").write_text(json.dumps(cfg, indent=1, ensure_ascii=False))
    # voices: style table + the graph's voice inputs, by name
    (dst / "voices").mkdir(exist_ok=True)
    for v in vj["voices"]:
        style = np.fromfile(src / f"voices/{v['id']}.bin", np.float32).reshape(-1, 256)
        extra, o, arrs = (
            np.fromfile(src / f"voices/{v['id']}.extra.bin", np.float32),
            0,
            {"style": style},
        )
        for vi in vj["voice_inputs"]:
            n = int(np.prod(vi["dims"]))
            key = vi["name"].removeprefix("voice_")
            if key != "prosody_col":
                arrs[key] = extra[o : o + n].copy()
            o += n
        mx.save_safetensors(
            str(dst / f"voices/{v['name'].lower()}.safetensors"),
            {k: mx.array(a) for k, a in arrs.items()},
        )
    # the release text front end
    fe = dst / "frontend"
    (fe / "words").mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "python/kale_frontend.js", fe / "kale_frontend.js")
    for sub in ("", "en-gb"):
        d = src / "frontend/words" / sub
        if not d.is_dir():
            continue
        (fe / "words" / sub).mkdir(exist_ok=True)
        for f in ("lexicon.tsv", "override.public.tsv", "terms.tsv", "g2p.bin"):
            if (d / f).exists():
                shutil.copy(d / f, fe / "words" / sub / f)
    for f in ("ettin-tokenizer.json", "LICENSE", "THIRD_PARTY_NOTICES.md"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)
    return dst


def convert_base(src: Path, dst: Path) -> Path:
    """kale-tts-1-base (Kokoro format) -> a `kokoro` MLX folder"""
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "converting kale-tts-1-base reads its PyTorch checkpoint: pip install torch"
        ) from e
    src, dst = Path(src), Path(dst)
    sd = torch.load(src / BASE_FILE, map_location="cpu", weights_only=True)
    flat = {}
    for top, part in sd.items():
        for k, v in part.items():
            k = k.removeprefix("module.")
            # older exports used torch.nn.utils.parametrizations.weight_norm names
            k = k.replace("parametrizations.weight.original0", "weight_g").replace(
                "parametrizations.weight.original1", "weight_v"
            )
            flat[f"{top}.{k}"] = mx.array(v.float().numpy())
    dst.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(dst / "model.safetensors"), flat, metadata={"format": "pt"})
    cfg = json.loads((src / "config.json").read_text())
    cfg["model_type"] = "kokoro"
    (dst / "config.json").write_text(json.dumps(cfg, indent=1, ensure_ascii=False))
    (dst / "voices").mkdir(exist_ok=True)
    voices = json.loads((src / "voices.json").read_text())
    for v in voices:
        pack = (
            torch.load(src / v["file"], map_location="cpu", weights_only=True)
            .float()
            .numpy()
        )
        mx.save_safetensors(
            str(dst / f"voices/{v['lang_code']}_{v['name'].lower()}.safetensors"),
            {"voice": mx.array(pack)},
        )
    (dst / "voices.json").write_text(json.dumps(voices, indent=1, ensure_ascii=False))
    for f in ("LICENSE", "README.md"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)
    return dst


def convert(src: Path, dst: Path) -> Path:
    src = Path(src)
    if _onnx_file(src) is not None:
        return convert_onnx_model(src, dst)
    if (src / BASE_FILE).exists():
        return convert_base(src, dst)
    raise FileNotFoundError(f"no {ONNX_GLOB} or {BASE_FILE} in {src}")


def _cache_root() -> Path:
    return (
        Path(os.environ.get("MLX_AUDIO_CACHE", Path.home() / ".cache" / "mlx_audio"))
        / "kale_tts"
    )


def prepare_model_path(
    model_path: Path,
    *,
    source: Optional[str] = None,
    revision: Optional[str] = None,
    force_download: bool = False,
) -> Path:
    """An MLX folder is returned as is; a release folder / repo (ONNX or Kokoro .pth) is converted once into the cache."""
    model_path = Path(model_path)
    if any(model_path.glob("*.safetensors")):
        return model_path
    if source is not None:
        # the generic download skips files the converter needs (voices, front end), so fetch the release set
        from huggingface_hub import snapshot_download

        model_path = Path(
            snapshot_download(
                source,
                revision=revision,
                allow_patterns=SOURCE_PATTERNS,
                force_download=force_download,
            )
        )
    elif _onnx_file(model_path) is None and not (model_path / BASE_FILE).exists():
        raise FileNotFoundError(
            f"{model_path} holds neither MLX weights nor kale-tts-1 release files"
        )
    key = hashlib.sha256(str(model_path.resolve()).encode()).hexdigest()[:16]
    dst = _cache_root() / f"{model_path.name}-{key}"
    if not (dst / "model.safetensors").exists() or force_download:
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        convert(model_path, tmp)
        shutil.rmtree(dst, ignore_errors=True)
        tmp.rename(dst)
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--hf-path",
        required=True,
        help="KaleComputer/kale-tts-1-{base,tiny,nano} or a local release folder",
    )
    ap.add_argument("--mlx-path", required=True)
    ap.add_argument("--revision")
    a = ap.parse_args()
    src = Path(a.hf_path)
    if not src.exists():
        from huggingface_hub import snapshot_download

        src = Path(
            snapshot_download(
                a.hf_path, revision=a.revision, allow_patterns=SOURCE_PATTERNS
            )
        )
    print(convert(src, Path(a.mlx_path)))


if __name__ == "__main__":
    main()
