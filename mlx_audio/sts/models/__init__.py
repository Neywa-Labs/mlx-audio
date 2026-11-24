# Copyright (c) 2025 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import importlib

_LAZY_EXPORTS = {
    "SAMAudio": "sam_audio",
    "SAMAudioProcessor": "sam_audio",
    "SeparationResult": "sam_audio",
    "Batch": "sam_audio",
    "save_audio": "sam_audio",
    "SAMAudioConfig": "sam_audio",
    # DeepFilterNet
    "DeepFilterNetModel": "deepfilternet",
    "DeepFilterNetConfig": "deepfilternet",
    "DeepFilterNet2Config": "deepfilternet",
    "DeepFilterNet3Config": "deepfilternet",
    "DeepFilterNetStreamer": "deepfilternet",
    "DeepFilterNetStreamingConfig": "deepfilternet",
    # MossFormer2 SE
    "MossFormer2SE": "mossformer2_se",
    "MossFormer2SEConfig": "mossformer2_se",
    "MossFormer2SEModel": "mossformer2_se",
    # LFM2.5-Audio
    "LFM2AudioModel": "lfm_audio",
    "LFM2AudioProcessor": "lfm_audio",
    "LFM2AudioConfig": "lfm_audio",
    "LFMModality": "lfm_audio",
    "ChatState": "lfm_audio",
    "GenerationConfig": "lfm_audio",
    # Moshi
    "MoshiConfig": "moshi",
    "MoshiSTSModel": "moshi",
    # Mel-Band-RoFormer
    "MelRoFormer": "mel_roformer",
    "MelRoFormerConfig": "mel_roformer",
    "MelRoFormerResult": "mel_roformer",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{_LAZY_EXPORTS[name]}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
