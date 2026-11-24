"""Test that lazy imports work correctly for modular installation.

Uses subprocess isolation because pytest imports all test files during collection,
which pollutes sys.modules before tests run (e.g., test_server.py imports miniaudio).
"""

import subprocess
import sys


def test_stt_utils_no_eager_imports():
    """Importing stt.utils should not import miniaudio or scipy."""
    code = """
import sys
import mlx_audio.stt.utils
assert "miniaudio" not in sys.modules, f"miniaudio was eagerly imported"
assert "scipy" not in sys.modules, f"scipy was eagerly imported"
print("OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"STT lazy import failed: {result.stderr}"


def test_tts_utils_no_eager_imports():
    """Importing tts.utils should not import transformers or mlx_lm."""
    code = """
import sys
import mlx_audio.tts.utils
assert "transformers" not in sys.modules, f"transformers was eagerly imported"
assert "mlx_lm" not in sys.modules, f"mlx_lm was eagerly imported"
print("OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"TTS lazy import failed: {result.stderr}"


def test_sts_utils_no_eager_imports():
    """Importing sts.utils should not import transformers or mlx_lm."""
    code = """
import sys
import mlx_audio.sts.utils
assert "transformers" not in sys.modules, f"transformers was eagerly imported"
assert "mlx_lm" not in sys.modules, f"mlx_lm was eagerly imported"
print("OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"STS lazy import failed: {result.stderr}"


def test_sts_model_no_eager_imports():
    """Importing one STS model should not import the other STS models."""
    code = """
import sys
import mlx_audio.sts.models as models

siblings = {"lfm_audio", "mel_roformer", "moshi", "mossformer2_se", "sam_audio"}
subpackages = siblings | {"deepfilternet"}
assert subpackages.issubset(dir(models))
assert all(f"{models.__name__}.{name}" not in sys.modules for name in subpackages)

deepfilternet = models.deepfilternet
from mlx_audio.sts.models import DeepFilterNetModel
assert deepfilternet.DeepFilterNetModel is DeepFilterNetModel
for name in siblings:
    assert f"{models.__name__}.{name}" not in sys.modules, f"{name} was eagerly imported"
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"STS model lazy import failed: {result.stderr}"


def test_codec_no_eager_imports():
    """Importing codec should not import miniaudio."""
    code = """
import sys
import mlx_audio.codec
assert "miniaudio" not in sys.modules, f"miniaudio was eagerly imported"
print("OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Codec lazy import failed: {result.stderr}"


def test_vendored_lm_no_eager_imports():
    code = """
import sys
import mlx_audio.lm.models.llama
assert "mlx_lm" not in sys.modules, "mlx_lm was eagerly imported"
assert "transformers" not in sys.modules, "transformers was eagerly imported"
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Vendored LM lazy import failed: {result.stderr}"
