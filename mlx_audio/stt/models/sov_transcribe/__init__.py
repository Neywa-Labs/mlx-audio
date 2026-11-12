from .config import ModelConfig
from .sov_transcribe import Model

DETECTION_HINTS = {
    "model_type_aliases": ["sovereignai_transcribe", "sov_transcribe"],
    "architectures": ["SovereignAITranscribeForConditionalGeneration"],
}

__all__ = ["Model", "ModelConfig"]
