"""
Tests for speech processing functionality using Kyutai STT models.
"""

import pytest
import numpy as np
import torch
import io
import wave
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from digipal.ai.speech_processor import (
    SpeechProcessor,
    AudioValidator,
    SpeechProcessingResult,
    AudioValidationResult
)


class TestAudioValidator:
    """Test cases for AudioValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = AudioValidator(target_sample_rate=24000, min_duration=0.1, max_duration=30.0)
    
    def test_init(self):
        """Test AudioValidator initialization."""
        assert self.validator.target_sample_rate == 24000
        assert self.validator.min_duration == 0.1
        assert self.validator.max_duration == 30.0
    
    def test_validate_audio_valid_numpy_array(self):
        """Test validation with valid numpy array."""
        # Create valid audio data (1 second at 24kHz)
        audio_array = np.random.uniform(-0.5, 0.5, 24000).astype(np.float32)
        
        result = self.validator.validate_audio(audio_array, sample_rate=24000)
        
        assert result.is_valid
        assert result.sample_rate == 24000
        assert abs(result.duration - 1.0) < 0.01  # Should be approximately 1 second
        assert result.channels == 1
        assert len(result.issues) == 0
    
    def test_validate_audio_too_short(self):
        """Test validation with audio that's too short."""
        # Create audio shorter than minimum duration
        audio_array = np.random.uniform(-0.5, 0.5, 1000).astype(np.float32)  # ~0.04 seconds
        
        result = self.validator.validate_audio(audio_array, sample_rate=24000)
        
        assert not result.is_valid
        assert any("too short" in issue for issue in result.issues)
    
    def test_validate_audio_too_long(self):
        """Test validation with audio that's too long."""
        # Create audio longer than maximum duration
        audio_array = np.random.uniform(-0.5, 0.5, 24000 * 35).astype(np.float32)  # 35 seconds
        
        result = self.validator.validate_audio(audio_array, sample_rate=24000)
        
        assert not result.is_valid
        assert any("too long" in issue for issue in result.issues)
    
    def test_validate_audio_silent(self):
        """Test validation with silent audio."""
        # Create very quiet audio
        audio_array = np.random.uniform(-0.005, 0.005, 24000).astype(np.float32)
        
        result = self.validator.validate_audio(audio_array, sample_rate=24000)
        
        assert not result.is_valid
        assert any("silent" in issue for issue in result.issues)
    
    def test_validate_audio_clipped(self):
        """Test validation with clipped audio."""
        # Create audio with clipping
        audio_array = np.random.uniform(-1.0, 1.0, 24000).astype(np.float32)
        
        result = self.validator.validate_audio(audio_array, sample_rate=24000)
        
        assert not result.is_valid
        assert any("clipped" in issue for issue in result.issues)
    
    def test_validate_audio_wrong_sample_rate(self):
        """Test validation with wrong sample rate."""
        audio_array = np.random.uniform(-0.5, 0.5, 16000).astype(np.float32)
        
        result = self.validator.validate_audio(audio_array, sample_rate=16000)
        
        assert not result.is_valid
        assert any("Sample rate mismatch" in issue for issue in result.issues)
    
    def test_validate_audio_stereo_to_mono(self):
        """Test validation with stereo audio conversion."""
        # Create stereo audio (2 channels)
        stereo_audio = np.random.uniform(-0.5, 0.5, (24000, 2)).astype(np.float32)
        
        result = self.validator.validate_audio(stereo_audio, sample_rate=24000)
        
        assert result.is_valid
        assert result.channels == 1  # Should be converted to mono
    
    def test_bytes_to_array_wav(self):
        """Test conversion of WAV bytes to numpy array."""
        # Create a simple WAV file in memory
        sample_rate = 24000
        duration = 1.0
        audio_data = np.random.uniform(-0.5, 0.5, int(sample_rate * duration)).astype(np.float32)
        
        # Convert to 16-bit PCM
        audio_16bit = (audio_data * 32767).astype(np.int16)
        
        # Create WAV bytes
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_16bit.tobytes())
        
        wav_bytes = wav_buffer.getvalue()
        
        # Test conversion
        result_array, result_sample_rate = self.validator._bytes_to_array(wav_bytes)
        
        assert result_array is not None
        assert result_sample_rate == sample_rate
        assert len(result_array) == len(audio_data)
        assert np.allclose(result_array, audio_data, atol=1/16384)  # Account for quantization
    
    def test_preprocess_audio(self):
        """Test audio preprocessing."""
        # Create test audio
        audio_array = np.random.uniform(-0.5, 0.5, 24000).astype(np.float32)
        
        # Test preprocessing
        processed = self.validator.preprocess_audio(audio_array, 24000)
        
        assert processed is not None
        assert len(processed) == len(audio_array)
        assert processed.dtype == np.float32
    
    def test_resample_audio(self):
        """Test audio resampling."""
        # Create audio at 16kHz
        original_rate = 16000
        target_rate = 24000
        audio_array = np.random.uniform(-0.5, 0.5, original_rate).astype(np.float32)
        
        # Test resampling
        resampled = self.validator._resample_audio(audio_array, original_rate, target_rate)
        
        expected_length = int(len(audio_array) * target_rate / original_rate)
        assert len(resampled) == expected_length
    
    def test_normalize_audio(self):
        """Test audio normalization."""
        # Create audio with high amplitude
        audio_array = np.random.uniform(-2.0, 2.0, 1000).astype(np.float32)
        
        normalized = self.validator._normalize_audio(audio_array)
        
        # Should be normalized to 70% of max
        assert np.max(np.abs(normalized)) <= 0.7
    
    def test_apply_noise_reduction(self):
        """Test noise reduction filter."""
        # Create audio with some low-frequency content
        audio_array = np.random.uniform(-0.5, 0.5, 1000).astype(np.float32)
        
        filtered = self.validator._apply_noise_reduction(audio_array)
        
        assert len(filtered) == len(audio_array)
        assert filtered.dtype == np.float32


class TestSpeechProcessor:
    """Test cases for SpeechProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = SpeechProcessor(model_id="kyutai/stt-2.6b-en_fr-trfs")
    
    def test_init(self):
        """Test SpeechProcessor initialization."""
        assert self.processor.model_id == "kyutai/stt-2.6b-en_fr-trfs"
        assert self.processor.device in ["cuda", "cpu"]
        assert not self.processor.is_model_loaded()
    
    def test_init_with_custom_device(self):
        """Test initialization with custom device."""
        processor = SpeechProcessor(device="cpu")
        assert processor.device == "cpu"
    
    @patch('digipal.ai.speech_processor.KyutaiSpeechToTextProcessor')
    @patch('digipal.ai.speech_processor.KyutaiSpeechToTextForConditionalGeneration')
    def test_load_model_success(self, mock_model_class, mock_processor_class):
        """Test successful model loading."""
        # Mock the processor and model
        mock_processor = Mock()
        mock_model = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        
        result = self.processor.load_model()
        
        assert result is True
        assert self.processor.is_model_loaded()
        mock_processor_class.from_pretrained.assert_called_once_with(self.processor.model_id)
        mock_model_class.from_pretrained.assert_called_once()
    
    @patch('digipal.ai.speech_processor.KyutaiSpeechToTextProcessor')
    def test_load_model_failure(self, mock_processor_class):
        """Test model loading failure."""
        # Mock processor loading to raise an exception
        mock_processor_class.from_pretrained.side_effect = Exception("Model not found")
        
        result = self.processor.load_model()
        
        assert result is False
        assert not self.processor.is_model_loaded()
    
    @patch('digipal.ai.speech_processor.KyutaiSpeechToTextProcessor')
    @patch('digipal.ai.speech_processor.KyutaiSpeechToTextForConditionalGeneration')
    def test_process_speech_success(self, mock_model_class, mock_processor_class):
        """Test successful speech processing."""
        # Mock the processor and model
        mock_processor = Mock()
        mock_model = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Mock processor methods
        mock_inputs = {
            'input_features': torch.randn(1, 80, 3000),
            'attention_mask': torch.ones(1, 3000)
        }
        mock_inputs_obj = Mock()
        mock_inputs_obj.to.return_value = mock_inputs_obj
        mock_inputs_obj.keys.return_value = mock_inputs.keys()
        mock_inputs_obj.__getitem__ = lambda self, key: mock_inputs[key]
        mock_inputs_obj.__iter__ = lambda self: iter(mock_inputs)
        mock_processor.return_value = mock_inputs_obj
        
        # Mock model generation
        mock_tokens = torch.tensor([[1, 2, 3, 4]])
        mock_model.generate.return_value = mock_tokens
        
        # Mock decoding
        mock_processor.batch_decode.return_value = ["hello world"]
        
        # Load model first
        self.processor.load_model()
        
        # Create test audio
        audio_array = np.random.uniform(-0.5, 0.5, 24000).astype(np.float32)
        
        result = self.processor.process_speech(audio_array, sample_rate=24000)
        
        assert result.success
        assert result.transcribed_text == "hello world"
        assert result.confidence > 0.0
        assert result.processing_time > 0.0
        assert result.error_message is None
    
    def test_process_speech_model_not_loaded(self):
        """Test speech processing when model is not loaded."""
        # Create test audio
        audio_array = np.random.uniform(-0.5, 0.5, 24000).astype(np.float32)
        
        with patch.object(self.processor, 'load_model', return_value=False):
            result = self.processor.process_speech(audio_array, sample_rate=24000)
        
        assert not result.success
        assert result.transcribed_text == ""
        assert "Failed to load" in result.error_message
    
    def test_process_speech_invalid_audio(self):
        """Test speech processing with invalid audio."""
        # Create invalid audio (too short)
        audio_array = np.random.uniform(-0.5, 0.5, 100).astype(np.float32)
        
        with patch.object(self.processor, 'is_model_loaded', return_value=True):
            result = self.processor.process_speech(audio_array, sample_rate=24000)
        
        assert not result.success
        assert "Audio validation failed" in result.error_message
    
    @patch('digipal.ai.speech_processor.KyutaiSpeechToTextProcessor')
    @patch('digipal.ai.speech_processor.KyutaiSpeechToTextForConditionalGeneration')
    def test_process_speech_processing_error(self, mock_model_class, mock_processor_class):
        """Test speech processing with processing error."""
        # Mock the processor and model
        mock_processor = Mock()
        mock_model = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Mock processor to raise an exception
        mock_processor.side_effect = Exception("Processing error")
        
        # Load model first
        self.processor.load_model()
        
        # Create test audio
        audio_array = np.random.uniform(-0.5, 0.5, 24000).astype(np.float32)
        
        result = self.processor.process_speech(audio_array, sample_rate=24000)
        
        assert not result.success
        assert "Speech processing error" in result.error_message
    
    def test_clean_transcription(self):
        """Test transcription cleaning."""
        # Test various cleaning scenarios
        test_cases = [
            ("  hello   world  ", "hello world"),
            ("hello [NOISE] world", "hello world"),
            ("test [SILENCE] text", "test text"),
            ("", ""),
            ("   ", ""),
            ("hello  world  [NOISE]  ", "hello world")
        ]
        
        for input_text, expected in test_cases:
            result = self.processor._clean_transcription(input_text)
            assert result == expected
    
    def test_estimate_confidence(self):
        """Test confidence estimation."""
        # Create mock validation result
        validation_result = AudioValidationResult(
            is_valid=True,
            sample_rate=24000,
            duration=2.0,
            channels=1,
            issues=[]
        )
        
        # Test with good transcription
        confidence = self.processor._estimate_confidence("hello world", validation_result)
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be relatively high
        
        # Test with empty transcription
        confidence = self.processor._estimate_confidence("", validation_result)
        assert confidence <= 0.2  # Should be very low
        
        # Test with validation issues
        validation_result.issues = ["Audio too quiet", "Sample rate mismatch"]
        confidence = self.processor._estimate_confidence("hello", validation_result)
        assert confidence <= 0.5  # Should be lower due to issues
    
    def test_get_model_info(self):
        """Test getting model information."""
        info = self.processor.get_model_info()
        
        assert 'model_id' in info
        assert 'device' in info
        assert 'loaded' in info
        assert 'target_sample_rate' in info
        assert 'supported_languages' in info
        
        assert info['model_id'] == "kyutai/stt-2.6b-en_fr-trfs"
        assert info['loaded'] is False  # Model not loaded in test
        assert info['target_sample_rate'] == 24000
        assert 'en' in info['supported_languages']
        assert 'fr' in info['supported_languages']
    
    @patch('digipal.ai.speech_processor.KyutaiSpeechToTextProcessor')
    @patch('digipal.ai.speech_processor.KyutaiSpeechToTextForConditionalGeneration')
    def test_unload_model(self, mock_model_class, mock_processor_class):
        """Test model unloading."""
        # Mock the processor and model
        mock_processor = Mock()
        mock_model = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Load model first
        self.processor.load_model()
        assert self.processor.is_model_loaded()
        
        # Unload model
        self.processor.unload_model()
        assert not self.processor.is_model_loaded()
        assert self.processor.model is None
        assert self.processor.processor is None


class TestSpeechProcessingResult:
    """Test cases for SpeechProcessingResult dataclass."""
    
    def test_success_result(self):
        """Test successful result creation."""
        result = SpeechProcessingResult(
            success=True,
            transcribed_text="hello world",
            confidence=0.95,
            processing_time=1.5
        )
        
        assert result.success
        assert result.transcribed_text == "hello world"
        assert result.confidence == 0.95
        assert result.processing_time == 1.5
        assert result.error_message is None
    
    def test_failure_result(self):
        """Test failure result creation."""
        result = SpeechProcessingResult(
            success=False,
            transcribed_text="",
            confidence=0.0,
            processing_time=0.1,
            error_message="Processing failed"
        )
        
        assert not result.success
        assert result.transcribed_text == ""
        assert result.confidence == 0.0
        assert result.processing_time == 0.1
        assert result.error_message == "Processing failed"


class TestAudioValidationResult:
    """Test cases for AudioValidationResult dataclass."""
    
    def test_valid_result(self):
        """Test valid audio result."""
        result = AudioValidationResult(
            is_valid=True,
            sample_rate=24000,
            duration=2.5,
            channels=1,
            issues=[]
        )
        
        assert result.is_valid
        assert result.sample_rate == 24000
        assert result.duration == 2.5
        assert result.channels == 1
        assert len(result.issues) == 0
    
    def test_invalid_result(self):
        """Test invalid audio result."""
        issues = ["Audio too short", "Sample rate mismatch"]
        result = AudioValidationResult(
            is_valid=False,
            sample_rate=16000,
            duration=0.05,
            channels=2,
            issues=issues
        )
        
        assert not result.is_valid
        assert result.sample_rate == 16000
        assert result.duration == 0.05
        assert result.channels == 2
        assert result.issues == issues


# Integration tests
class TestSpeechProcessingIntegration:
    """Integration tests for speech processing components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = SpeechProcessor()
        self.validator = AudioValidator()
    
    def test_end_to_end_processing_mock(self):
        """Test end-to-end processing with mocked models."""
        with patch('digipal.ai.speech_processor.KyutaiSpeechToTextProcessor') as mock_proc_class, \
             patch('digipal.ai.speech_processor.KyutaiSpeechToTextForConditionalGeneration') as mock_model_class:
            
            # Mock the processor and model
            mock_processor = Mock()
            mock_model = Mock()
            mock_proc_class.from_pretrained.return_value = mock_processor
            mock_model_class.from_pretrained.return_value = mock_model
            
            # Mock processing pipeline
            mock_inputs = {
                'input_features': torch.randn(1, 80, 3000),
                'attention_mask': torch.ones(1, 3000)
            }
            mock_inputs_obj = Mock()
            mock_inputs_obj.to.return_value = mock_inputs_obj
            mock_inputs_obj.keys.return_value = mock_inputs.keys()
            mock_inputs_obj.__getitem__ = lambda self, key: mock_inputs[key]
            mock_inputs_obj.__iter__ = lambda self: iter(mock_inputs)
            mock_processor.return_value = mock_inputs_obj
            
            mock_tokens = torch.tensor([[1, 2, 3]])
            mock_model.generate.return_value = mock_tokens
            mock_processor.batch_decode.return_value = ["test transcription"]
            
            # Create valid test audio
            audio_array = np.random.uniform(-0.3, 0.3, 48000).astype(np.float32)  # 2 seconds
            
            # Process speech
            result = self.processor.process_speech(audio_array, sample_rate=24000)
            
            assert result.success
            assert result.transcribed_text == "test transcription"
            assert result.confidence > 0.0
    
    def test_audio_validation_integration(self):
        """Test integration between audio validation and processing."""
        # Test with various audio conditions
        test_cases = [
            # (audio_length, sample_rate, should_be_valid)
            (24000, 24000, True),   # 1 second, correct rate - valid
            (1000, 24000, False),   # Too short - invalid
            (24000 * 35, 24000, False),  # Too long - invalid
            (16000, 16000, False),  # Wrong sample rate - invalid
        ]
        
        for audio_length, sample_rate, should_be_valid in test_cases:
            audio_array = np.random.uniform(-0.3, 0.3, audio_length).astype(np.float32)
            
            validation_result = self.validator.validate_audio(audio_array, sample_rate)
            assert validation_result.is_valid == should_be_valid