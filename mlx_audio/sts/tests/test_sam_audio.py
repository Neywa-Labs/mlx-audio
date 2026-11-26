"""Tests for SAM-Audio model."""

import unittest
from unittest.mock import patch

import mlx.core as mx


def _tiny_dacvae_config():
    from mlx_audio.codec.models.dacvae import DACVAEConfig

    return DACVAEConfig(
        encoder_dim=8,
        encoder_rates=[2, 2],
        latent_dim=16,
        decoder_dim=32,
        decoder_rates=[2, 2],
        n_codebooks=2,
        codebook_size=16,
        codebook_dim=8,
    )


def _tiny_sam_audio_config():
    from mlx_audio.sts.models.sam_audio.config import (
        SAMAudioConfig,
        T5EncoderConfig,
        TransformerConfig,
    )

    codebook_dim = 8
    return SAMAudioConfig(
        in_channels=6 * codebook_dim,
        audio_codec=_tiny_dacvae_config(),
        text_encoder=T5EncoderConfig(name="tiny-t5", dim=16, max_length=8),
        transformer=TransformerConfig(
            dim=32,
            n_heads=4,
            n_layers=1,
            context_dim=32,
            out_channels=2 * codebook_dim,
            frequency_embedding_dim=16,
            multiple_of=8,
            max_positions=64,
        ),
        anchor_embedding_dim=8,
    )


class TestSAMAudioConfig(unittest.TestCase):
    """Tests for SAM-Audio configuration classes."""

    def test_dacvae_config_defaults(self):
        """Test DACVAEConfig default values."""
        from mlx_audio.sts.models.sam_audio.config import DACVAEConfig

        config = DACVAEConfig()
        self.assertEqual(config.sample_rate, 48000)
        self.assertEqual(config.latent_dim, 1024)
        self.assertEqual(config.codebook_dim, 128)
        self.assertEqual(config.encoder_rates, [2, 8, 10, 12])
        self.assertEqual(config.decoder_rates, [12, 10, 8, 2])
        self.assertEqual(config.encoder_dim, 64)
        self.assertEqual(config.decoder_dim, 1536)

    def test_dacvae_config_hop_length(self):
        """Test DACVAEConfig hop_length calculation."""
        from mlx_audio.sts.models.sam_audio.config import DACVAEConfig

        config = DACVAEConfig()
        # hop_length = product of encoder_rates = 2*8*10*12 = 1920
        expected_hop = 2 * 8 * 10 * 12
        self.assertEqual(config.hop_length, expected_hop)

    def test_t5_encoder_config_defaults(self):
        """Test T5EncoderConfig default values."""
        from mlx_audio.sts.models.sam_audio.config import T5EncoderConfig

        config = T5EncoderConfig()
        self.assertEqual(config.name, "t5-base")
        self.assertEqual(config.dim, 768)
        self.assertEqual(config.max_length, 512)

    def test_transformer_config_defaults(self):
        """Test TransformerConfig default values."""
        from mlx_audio.sts.models.sam_audio.config import TransformerConfig

        config = TransformerConfig()
        self.assertEqual(config.dim, 2816)
        self.assertEqual(config.n_layers, 22)
        self.assertEqual(config.n_heads, 22)
        self.assertEqual(config.out_channels, 256)

    def test_sam_audio_config_defaults(self):
        """Test SAMAudioConfig default values."""
        from mlx_audio.sts.models.sam_audio.config import SAMAudioConfig

        config = SAMAudioConfig()
        self.assertEqual(config.in_channels, 768)
        self.assertIsNotNone(config.audio_codec)
        self.assertIsNotNone(config.text_encoder)
        self.assertIsNotNone(config.transformer)

    def test_sam_audio_config_from_dict(self):
        """Test SAMAudioConfig.from_dict method."""
        from mlx_audio.sts.models.sam_audio.config import SAMAudioConfig

        config_dict = {
            "in_channels": 768,
            "audio_codec": {"sample_rate": 44100},
            "text_encoder": {"name": "t5-small"},
        }

        config = SAMAudioConfig.from_dict(config_dict)
        self.assertEqual(config.in_channels, 768)
        self.assertEqual(config.audio_codec.sample_rate, 44100)
        self.assertEqual(config.text_encoder.name, "t5-small")


class TestSeparationResult(unittest.TestCase):
    """Tests for SeparationResult dataclass."""

    def test_separation_result_creation(self):
        """Test SeparationResult dataclass creation."""
        from mlx_audio.sts.models.sam_audio.model import SeparationResult

        target = [mx.zeros((1000, 1))]
        residual = [mx.zeros((1000, 1))]
        noise = mx.zeros((1, 10, 256))

        result = SeparationResult(
            target=target, residual=residual, noise=noise, peak_memory=1.5
        )

        self.assertEqual(len(result.target), 1)
        self.assertEqual(len(result.residual), 1)
        self.assertEqual(result.noise.shape, (1, 10, 256))
        self.assertEqual(result.peak_memory, 1.5)


class TestDACVAECodec(unittest.TestCase):
    """Tests for DACVAE codec."""

    def setUp(self):
        """Set up test fixtures."""
        from mlx_audio.codec.models.dacvae import DACVAE

        self.config = _tiny_dacvae_config()
        self.codec = DACVAE(self.config)

    def test_codec_initialization(self):
        """Test DACVAE initialization."""
        self.assertIsNotNone(self.codec.encoder)
        self.assertIsNotNone(self.codec.decoder)
        self.assertIsNotNone(self.codec.quantizer_in_proj)
        self.assertIsNotNone(self.codec.quantizer_out_proj)
        self.assertEqual(self.codec.sample_rate, 48000)

    def test_codec_hop_length(self):
        """Test DACVAE hop_length property."""
        expected_hop = 4
        self.assertEqual(self.codec.hop_length, expected_hop)

    def test_codec_encode_shape(self):
        """Test DACVAE encode output shape."""
        # Create test audio: (batch, 1, samples)
        batch_size = 1
        samples = self.codec.hop_length * 1  # 1 frames
        audio = mx.zeros((batch_size, 1, samples))

        # Encode
        encoded = self.codec(audio)
        mx.eval(encoded)

        # Expected shape: (batch, codebook_dim, frames)
        expected_frames = samples // self.codec.hop_length
        self.assertEqual(encoded.shape[0], batch_size)
        self.assertEqual(encoded.shape[1], self.config.codebook_dim)
        self.assertEqual(encoded.shape[2], expected_frames)

    def test_codec_decode_shape(self):
        """Test DACVAE decode output shape."""
        batch_size = 1
        codebook_dim = self.config.codebook_dim
        frames = 1

        # Create encoded features
        encoded = mx.zeros((batch_size, codebook_dim, frames))

        # Decode
        decoded = self.codec.decode(encoded)
        mx.eval(decoded)

        # Expected shape: (batch, samples, 1)
        self.assertEqual(decoded.shape[0], batch_size)
        self.assertEqual(decoded.shape[2], 1)

    def test_codec_chunked_decode(self):
        """Test DACVAE chunked decode."""
        batch_size = 1
        codebook_dim = self.config.codebook_dim
        frames = 5

        encoded = mx.zeros((batch_size, codebook_dim, frames))

        # Decode with chunking
        decoded = self.codec.decode(encoded, chunk_size=50)
        mx.eval(decoded)

        # Should produce output
        self.assertEqual(decoded.shape[0], batch_size)
        self.assertEqual(decoded.shape[2], 1)

    def test_wav_idx_to_feature_idx(self):
        """Test waveform to feature index conversion."""
        # 1920 samples should be 1 frame
        wav_idx = self.codec.hop_length
        feature_idx = self.codec.wav_idx_to_feature_idx(wav_idx)
        self.assertEqual(feature_idx, 1)

    def test_feature_idx_to_wav_idx(self):
        """Test feature to waveform index conversion."""
        feature_idx = 1
        wav_idx = self.codec.feature_idx_to_wav_idx(feature_idx)
        self.assertEqual(wav_idx, self.codec.hop_length)


class TestSAMAudioModel(unittest.TestCase):
    """Tests for SAMAudio model."""

    def setUp(self):
        """Set up test fixtures."""
        from mlx_audio.sts.models.sam_audio.model import SAMAudio

        self.config = _tiny_sam_audio_config()
        self.model = SAMAudio(self.config)
        self.model.eval()

    def test_model_initialization(self):
        """Test SAMAudio initialization."""
        self.assertIsNotNone(self.model.audio_codec)
        self.assertIsNotNone(self.model.text_encoder)
        self.assertIsNotNone(self.model.transformer)
        self.assertIsNotNone(self.model.proj)
        self.assertIsNotNone(self.model.align_masked_video)
        self.assertIsNotNone(self.model.memory_proj)

    def test_model_sample_rate(self):
        """Test SAMAudio sample_rate property."""
        self.assertEqual(self.model.sample_rate, 48000)

    def test_sanitize_weights(self):
        """Test SAMAudio sanitize method."""
        # Create mock weights with keys that should be filtered.
        # sanitize() removes text_encoder., span_predictor., visual_ranker.,
        # text_ranker., vision_encoder., align_masked_video.conv.weight, and
        # keys with wm_rates.
        dim = self.config.transformer.dim
        weights = {
            "transformer.layers.0.weight": mx.zeros((10, 10)),
            "text_encoder.some_param": mx.zeros((10,)),  # Should be removed
            "span_predictor.weight": mx.zeros((10,)),  # Should be removed
            "some.wm_rates.param": mx.zeros((10,)),  # Should be removed
            "audio_codec.encoder.weight": mx.zeros((10, 10)),
            "align_masked_video.conv.weight": mx.zeros((dim, 16, 1)),  # Unused
            "align_masked_video.conv.bias": mx.ones((dim,)),
            "align_masked_video.layer_norm.weight": mx.ones((dim,)),
            "align_masked_video.layer_norm.bias": mx.zeros((dim,)),
            "align_masked_video.gate": mx.array([-0.1665]),
        }

        sanitized = self.model.sanitize(weights)

        # Filtered prefixes should be removed
        self.assertNotIn("text_encoder.some_param", sanitized)
        self.assertNotIn("span_predictor.weight", sanitized)
        self.assertNotIn("some.wm_rates.param", sanitized)
        self.assertNotIn("align_masked_video.conv.weight", sanitized)
        self.assertNotIn("align_masked_video.conv.bias", sanitized)
        # Other weights and zero-video alignment params should remain
        self.assertIn("transformer.layers.0.weight", sanitized)
        self.assertIn("audio_codec.encoder.weight", sanitized)
        self.assertIn("align_masked_video.conv_bias", sanitized)
        self.assertIn("align_masked_video.layer_norm.weight", sanitized)
        self.assertIn("align_masked_video.layer_norm.bias", sanitized)
        self.assertIn("align_masked_video.gate", sanitized)

        # Sanitize should be idempotent on already-converted MLX keys
        resanitized = self.model.sanitize(sanitized)
        self.assertEqual(set(resanitized.keys()), set(sanitized.keys()))

    def test_get_audio_features_shape(self):
        """Test _get_audio_features output shape."""
        batch_size = 1
        samples = self.model.audio_codec.hop_length * 10
        audio = mx.zeros((batch_size, 1, samples))

        features = self.model._get_audio_features(audio)
        mx.eval(features)

        # Output shape: (batch, frames, 2*codebook_dim)
        expected_frames = samples // self.model.audio_codec.hop_length
        self.assertEqual(features.shape[0], batch_size)
        self.assertEqual(features.shape[1], expected_frames)
        self.assertEqual(features.shape[2], 2 * self.config.audio_codec.codebook_dim)


class TestSAMAudioProcessor(unittest.TestCase):
    """Tests for SAMAudioProcessor."""

    def test_processor_initialization(self):
        """Test SAMAudioProcessor initialization."""
        from mlx_audio.sts.models.sam_audio.processor import SAMAudioProcessor

        processor = SAMAudioProcessor(
            audio_hop_length=1920,
            audio_sampling_rate=48000,
        )
        self.assertEqual(processor.audio_hop_length, 1920)
        self.assertEqual(processor.audio_sampling_rate, 48000)

    def test_batch_dataclass(self):
        """Test Batch dataclass."""
        from mlx_audio.sts.models.sam_audio.processor import Batch

        batch = Batch(
            descriptions=["speech", "music"],
            audios=mx.zeros((2, 1, 48000)),
            sizes=mx.array([25, 25]),
            wav_sizes=mx.array([48000, 48000]),
            anchor_ids=mx.zeros((2, 3), dtype=mx.int32),
            anchor_alignment=mx.zeros((2, 25), dtype=mx.int32),
        )

        self.assertEqual(len(batch.descriptions), 2)
        self.assertEqual(batch.audios.shape[0], 2)

    def test_wav_to_feature_idx(self):
        """Test wav_to_feature_idx conversion."""
        from mlx_audio.sts.models.sam_audio.processor import SAMAudioProcessor

        processor = SAMAudioProcessor(
            audio_hop_length=192,
            audio_sampling_rate=48000,
        )

        # 1920 samples = 1 frame
        result = processor.wav_to_feature_idx(192)
        self.assertEqual(result, 1)

        # 3840 samples = 2 frames
        result = processor.wav_to_feature_idx(384)
        self.assertEqual(result, 2)


class TestODEStepFunctions(unittest.TestCase):
    """Tests for ODE step functions."""

    def setUp(self):
        """Set up test fixtures."""
        from mlx_audio.sts.models.sam_audio.model import SAMAudio

        self.config = _tiny_sam_audio_config()
        self.model = SAMAudio(self.config)
        self.model.eval()

    def test_ode_step_euler_shape(self):
        """Test Euler ODE step output shape."""
        batch_size = 1
        seq_len = 2
        channels = 2 * self.config.audio_codec.codebook_dim

        noisy_audio = mx.zeros((batch_size, seq_len, channels))
        audio_features = mx.zeros((batch_size, seq_len, channels))
        text_features = mx.zeros((batch_size, 5, self.config.text_encoder.dim))

        result = self.model._ode_step_euler(
            t=0.0,
            dt=0.0625,
            noisy_audio=noisy_audio,
            audio_features=audio_features,
            text_features=text_features,
            text_mask=None,
            anchor_ids=None,
            anchor_alignment=None,
            audio_pad_mask=None,
        )
        mx.eval(result)

        self.assertEqual(result.shape, noisy_audio.shape)

    def test_ode_step_midpoint_shape(self):
        """Test Midpoint ODE step output shape."""
        batch_size = 1
        seq_len = 2
        channels = 2 * self.config.audio_codec.codebook_dim

        noisy_audio = mx.zeros((batch_size, seq_len, channels))
        audio_features = mx.zeros((batch_size, seq_len, channels))
        text_features = mx.zeros((batch_size, 5, self.config.text_encoder.dim))

        result = self.model._ode_step_midpoint(
            t=0.0,
            dt=0.0625,
            noisy_audio=noisy_audio,
            audio_features=audio_features,
            text_features=text_features,
            text_mask=None,
            anchor_ids=None,
            anchor_alignment=None,
            audio_pad_mask=None,
        )
        mx.eval(result)

        self.assertEqual(result.shape, noisy_audio.shape)


class TestAlignMaskedVideo(unittest.TestCase):
    """Tests for zero-video align_masked_video parity and backward compatibility."""

    def test_align_zeros_matches_full_align_modalities(self):
        """align_zeros() matches full AlignModalities(anchor, zeros) across dtypes."""
        from mlx_audio.sts.models.sam_audio.align import AlignModalities

        vision_dim = 16
        dim = 32
        batch_size, seq_len = 2, 5

        for dtype in (mx.float32, mx.float16, mx.bfloat16):
            mx.random.seed(0)
            conv_weight = mx.random.normal((dim, 1, vision_dim)).astype(dtype)
            conv_bias = mx.random.normal((dim,)).astype(dtype)
            ln_weight = (mx.ones((dim,)) + 0.1 * mx.random.normal((dim,))).astype(dtype)
            ln_bias = mx.random.normal((dim,)).astype(dtype)
            gate = mx.array([-0.1665], dtype=dtype)
            anchor = mx.random.normal((batch_size, seq_len, dim)).astype(dtype)

            full_align = AlignModalities(vision_dim, dim)
            full_align.conv_weight = conv_weight
            full_align.conv_bias = conv_bias
            full_align.layer_norm.weight = ln_weight
            full_align.layer_norm.bias = ln_bias
            full_align.gate = gate

            zero_align = AlignModalities(None, dim)
            zero_align.conv_bias = conv_bias
            zero_align.layer_norm.weight = ln_weight
            zero_align.layer_norm.bias = ln_bias
            zero_align.gate = gate

            ref = full_align(
                anchor, mx.zeros((batch_size, vision_dim, seq_len), dtype=dtype)
            )
            out = zero_align.align_zeros(anchor)
            mx.eval(ref, out)

            self.assertEqual(out.dtype, dtype)
            self.assertTrue(mx.allclose(out, ref, atol=1e-6, rtol=1e-6).item())

    def test_align_without_in_channels_semantics(self):
        """Test AlignModalities(None, dim) for None vs tensor target."""
        from mlx_audio.sts.models.sam_audio.align import AlignModalities

        align = AlignModalities(None, 32)
        anchor = mx.ones((1, 4, 32))
        self.assertTrue(mx.array_equal(align(anchor, None), anchor).item())
        with self.assertRaises(ValueError):
            align(anchor, mx.zeros((1, 16, 4)))

    def test_align_inputs_matches_full_align_and_runs_before_anchors(self):
        """SAMAudio.align_inputs applies align_masked_video before embed_anchors."""
        from mlx.utils import tree_flatten

        from mlx_audio.sts.models.sam_audio.align import AlignModalities
        from mlx_audio.sts.models.sam_audio.model import SAMAudio

        config = _tiny_sam_audio_config()
        dim = config.transformer.dim
        channels = 2 * config.audio_codec.codebook_dim
        vision_dim = 16
        batch_size, seq_len = 2, 4

        for dtype in (mx.float32, mx.float16, mx.bfloat16):
            mx.random.seed(1)
            model = SAMAudio(config)
            model.eval()

            # Build full checkpoint in target dtype with upstream-style align keys
            base_weights = {
                k: v.astype(dtype) if mx.issubdtype(v.dtype, mx.floating) else v
                for k, v in tree_flatten(model.parameters())
                if not k.startswith("align_masked_video.")
            }
            conv_weight = mx.random.normal((dim, vision_dim, 1)).astype(dtype)
            conv_bias = mx.random.normal((dim,)).astype(dtype)
            ln_weight = (mx.ones((dim,)) + 0.1 * mx.random.normal((dim,))).astype(dtype)
            ln_bias = mx.random.normal((dim,)).astype(dtype)
            gate = mx.array([-0.1665], dtype=dtype)

            upstream_weights = {
                **base_weights,
                "align_masked_video.conv.weight": conv_weight,
                "align_masked_video.conv.bias": conv_bias,
                "align_masked_video.layer_norm.weight": ln_weight,
                "align_masked_video.layer_norm.bias": ln_bias,
                "align_masked_video.gate": gate,
            }
            sanitized = model.sanitize(upstream_weights)
            model.load_weights(list(sanitized.items()), strict=True)

            # Set non-zero anchor gate so embed_anchors actually contributes
            model.embed_anchors.gate = mx.array([0.25], dtype=dtype)

            noisy_audio = mx.random.normal((batch_size, seq_len, channels)).astype(
                dtype
            )
            audio_features = mx.random.normal((batch_size, seq_len, channels)).astype(
                dtype
            )
            anchor_ids = mx.array([[0, 1, 2], [2, 0, 1]], dtype=mx.int32)
            anchor_alignment = mx.array([[0, 1, 2, 1], [1, 0, 2, 0]], dtype=mx.int32)

            # Reference using full AlignModalities with zero video features
            full_align = AlignModalities(vision_dim, dim)
            full_align.conv_weight = mx.transpose(conv_weight, (0, 2, 1))
            full_align.conv_bias = conv_bias
            full_align.layer_norm.weight = ln_weight
            full_align.layer_norm.bias = ln_bias
            full_align.gate = gate

            x = mx.concatenate(
                [noisy_audio, mx.zeros_like(audio_features), audio_features],
                axis=2,
            )
            projected = model.proj(x)
            ref_before_anchors = full_align(
                projected, mx.zeros((batch_size, vision_dim, seq_len), dtype=dtype)
            )
            ref = model.embed_anchors(ref_before_anchors, anchor_ids, anchor_alignment)

            # Verify align_masked_video runs before embed_anchors
            seen_by_anchors = []
            orig_embed_anchors = model.embed_anchors

            def spy_embed_anchors(
                inp,
                ids=None,
                alignment=None,
                *,
                seen=seen_by_anchors,
                orig=orig_embed_anchors,
            ):
                seen.append(inp)
                return orig(inp, ids, alignment)

            with patch.object(model, "embed_anchors", side_effect=spy_embed_anchors):
                out = model.align_inputs(
                    noisy_audio,
                    audio_features,
                    anchor_ids=anchor_ids,
                    anchor_alignment=anchor_alignment,
                )

            mx.eval(ref_before_anchors, ref, out, seen_by_anchors[0])
            self.assertEqual(out.dtype, dtype)
            self.assertTrue(
                mx.allclose(
                    seen_by_anchors[0], ref_before_anchors, atol=1e-6, rtol=1e-6
                ).item()
            )
            self.assertTrue(mx.allclose(out, ref, atol=1e-6, rtol=1e-6).item())
            # Confirm the zero-video bias is genuinely non-zero
            self.assertFalse(
                mx.allclose(seen_by_anchors[0], projected, atol=1e-3, rtol=1e-3).item()
            )

    def test_backward_compatible_without_align_masked_video_weights(self):
        """Legacy checkpoints without align_masked_video load as an exact no-op."""
        import warnings

        from mlx.utils import tree_flatten

        from mlx_audio.sts.models.sam_audio.model import SAMAudio

        config = _tiny_sam_audio_config()
        channels = 2 * config.audio_codec.codebook_dim
        batch_size, seq_len = 1, 4

        for dtype in (mx.float32, mx.float16, mx.bfloat16):
            mx.random.seed(2)
            model = SAMAudio(config)
            model.eval()

            legacy_weights = {
                k: v.astype(dtype) if mx.issubdtype(v.dtype, mx.floating) else v
                for k, v in tree_flatten(model.parameters())
                if not k.startswith("align_masked_video.")
            }

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.load_weights(list(legacy_weights.items()), strict=False)

            noisy_audio = mx.random.normal((batch_size, seq_len, channels)).astype(
                dtype
            )
            audio_features = mx.random.normal((batch_size, seq_len, channels)).astype(
                dtype
            )

            out = model.align_inputs(noisy_audio, audio_features)
            x = mx.concatenate(
                [noisy_audio, mx.zeros_like(audio_features), audio_features],
                axis=2,
            )
            expected = model.proj(x)
            mx.eval(out, expected)

            self.assertEqual(out.dtype, expected.dtype)
            self.assertTrue(mx.array_equal(out, expected).item())

    def test_align_masked_video_not_quantized(self):
        """Quantizing SAMAudio leaves align_masked_video parameters unquantized."""
        from mlx.utils import tree_flatten

        from mlx_audio.convert import build_quant_predicate
        from mlx_audio.lm.convert import quantize_model
        from mlx_audio.sts.models.sam_audio.config import (
            SAMAudioConfig,
            T5EncoderConfig,
            TransformerConfig,
        )
        from mlx_audio.sts.models.sam_audio.model import SAMAudio

        codebook_dim = 8
        config = SAMAudioConfig(
            in_channels=6 * codebook_dim,
            audio_codec=_tiny_dacvae_config(),
            text_encoder=T5EncoderConfig(name="tiny-t5", dim=64, max_length=8),
            transformer=TransformerConfig(
                dim=64,
                n_heads=4,
                n_layers=1,
                context_dim=64,
                out_channels=2 * codebook_dim,
                frequency_embedding_dim=16,
                multiple_of=64,
                max_positions=64,
            ),
            anchor_embedding_dim=64,
        )
        model = SAMAudio(config)
        model.align_masked_video.conv_bias = mx.ones((64,))
        model.align_masked_video.gate = mx.array([-0.1665])

        predicate = build_quant_predicate(model)
        quantize_model(model, {}, group_size=64, bits=4, quant_predicate=predicate)

        params = dict(tree_flatten(model.parameters()))
        self.assertIn("align_masked_video.conv_bias", params)
        self.assertIn("align_masked_video.gate", params)
        self.assertIn("align_masked_video.layer_norm.weight", params)
        self.assertIn("align_masked_video.layer_norm.bias", params)
        self.assertNotIn("align_masked_video.scales", params)
        self.assertNotIn("align_masked_video.layer_norm.scales", params)
        self.assertEqual(params["align_masked_video.conv_bias"].dtype, mx.float32)
        self.assertEqual(params["align_masked_video.gate"].dtype, mx.float32)
