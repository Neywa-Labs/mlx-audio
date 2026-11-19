from pathlib import Path

import mlx.core as mx

from mlx_audio import convert


def test_load_weights_ignores_tokenizer_in_parent_dirs(tmp_path, monkeypatch):
    model_dir = tmp_path / "tokenizer-experiments" / "whisper"
    model_dir.mkdir(parents=True)
    mx.save_safetensors(
        str(model_dir / "model.safetensors"), {"weight": mx.zeros((2, 2))}
    )
    mx.save_safetensors(
        str(model_dir / "tokenizer.safetensors"), {"tok": mx.ones((3,))}
    )

    loaded_files = []
    real_load = mx.load

    def spy(path):
        loaded_files.append(Path(path).name)
        return real_load(path)

    monkeypatch.setattr(convert.mx, "load", spy)

    weights = convert.load_weights(model_dir)

    assert loaded_files == ["model.safetensors"]
    assert sorted(weights.keys()) == ["weight"]
