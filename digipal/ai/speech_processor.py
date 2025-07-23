"""
Speech processing module using Kyutai speech-to-text models.

This module provides speech-to-text functionality using Kyutai's STT models
with audio validation, preprocessing, and error handling.
"""

import torch
import numpy as np
import logging
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass
import io
import wave
from transformers import KyutaiSpeechToTextProcessor, KyutaiSpeechToTextForConditionalGeneration

logger = logging.getLogger(__name__)


@dataclass
class AudioValidationResult:
    """Result of audio validation checks."""
    is_valid: bool
    sample_rate: int
    duration: float
    channels: int
    issues: List[str]


@dataclass
class SpeechProcessingResult:
    """Result of speech processing operation."""
    success: bool
    transcribed_text: str
    confidence: float
    processing_time: float
    error_message: Optional[str] = None


class AudioValidator:
    """Validates and preprocesses audio input for speech recognition."""
    
    def __init__(self, target_sample_rate: int = 24000, min_duration: float = 0.1, max_duration: float = 30.0):
        """
        Initialize audio validator.
        
        Args:
            target_sample_rate: Target sample rate for processing (24kHz for Kyutai)
            min_duration: Minimum audio duration in seconds
            max_duration: Maximum audio duration in seconds
        """
        self.target_sample_rate = target_sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
    
    def validate_audio(self, audio_data: Union[bytes, np.ndarray], sample_rate: Optional[int] = None) -> AudioValidationResult:
        """
        Validate audio data for speech processing.
        
        Args:
            audio_data: Raw audio data as bytes or numpy array
            sample_rate: Sample rate of the audio data
            
        Returns:
            AudioValidationResult with validation details
        """
        issues = []
        
        try:
            # Convert bytes to numpy array if needed
            if isinstance(audio_data, bytes):
                audio_array, detected_sample_rate = self._bytes_to_array(audio_data)
                if sample_rate is None:
                    sample_rate = detected_sample_rate
            else:
                audio_array = audio_data
                if sample_rate is None:
                    sample_rate = self.target_sample_rate
            
            # Check if audio array is valid
            if audio_array is None or len(audio_array) == 0:
                issues.append("Empty or invalid audio data")
                return AudioValidationResult(False, 0, 0.0, 0, issues)
            
            # Calculate duration
            duration = len(audio_array) / sample_rate
            
            # Detect number of channels
            if audio_array.ndim == 1:
                channels = 1
            else:
                channels = audio_array.shape[1] if audio_array.ndim == 2 else 1
                # Convert to mono if stereo
                if channels > 1:
                    audio_array = np.mean(audio_array, axis=1)
                    channels = 1
            
            # Validate duration
            if duration < self.min_duration:
                issues.append(f"Audio too short: {duration:.2f}s (minimum: {self.min_duration}s)")
            
            if duration > self.max_duration:
                issues.append(f"Audio too long: {duration:.2f}s (maximum: {self.max_duration}s)")
            
            # Check sample rate
            if sample_rate != self.target_sample_rate:
                issues.append(f"Sample rate mismatch: {sample_rate}Hz (expected: {self.target_sample_rate}Hz)")
            
            # Check for silence (very low amplitude)
            if np.max(np.abs(audio_array)) < 0.01:
                issues.append("Audio appears to be silent or very quiet")
            
            # Check for clipping
            if np.max(np.abs(audio_array)) > 0.95:
                issues.append("Audio may be clipped (too loud)")
            
            is_valid = len(issues) == 0
            
            return AudioValidationResult(
                is_valid=is_valid,
                sample_rate=sample_rate,
                duration=duration,
                channels=channels,
                issues=issues
            )
            
        except Exception as e:
            logger.error(f"Error validating audio: {e}")
            issues.append(f"Validation error: {str(e)}")
            return AudioValidationResult(False, 0, 0.0, 0, issues)
    
    def _bytes_to_array(self, audio_bytes: bytes) -> tuple[Optional[np.ndarray], int]:
        """
        Convert audio bytes to numpy array.
        
        Args:
            audio_bytes: Raw audio bytes
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            # Try to parse as WAV file
            with io.BytesIO(audio_bytes) as audio_io:
                with wave.open(audio_io, 'rb') as wav_file:
                    sample_rate = wav_file.getframerate()
                    channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    frames = wav_file.readframes(-1)
                    
                    # Convert to numpy array
                    if sample_width == 1:
                        audio_array = np.frombuffer(frames, dtype=np.uint8)
                        audio_array = (audio_array.astype(np.float32) - 128) / 128.0
                    elif sample_width == 2:
                        audio_array = np.frombuffer(frames, dtype=np.int16)
                        audio_array = audio_array.astype(np.float32) / 32768.0
                    elif sample_width == 4:
                        audio_array = np.frombuffer(frames, dtype=np.int32)
                        audio_array = audio_array.astype(np.float32) / 2147483648.0
                    else:
                        raise ValueError(f"Unsupported sample width: {sample_width}")
                    
                    # Handle stereo to mono conversion
                    if channels == 2:
                        audio_array = audio_array.reshape(-1, 2)
                        audio_array = np.mean(audio_array, axis=1)
                    
                    return audio_array, sample_rate
                    
        except Exception as e:
            logger.warning(f"Failed to parse as WAV: {e}")
            
        # Fallback: assume raw 16-bit PCM at target sample rate
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
            return audio_array, self.target_sample_rate
        except Exception as e:
            logger.error(f"Failed to convert audio bytes: {e}")
            return None, 0
    
    def preprocess_audio(self, audio_array: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Preprocess audio for optimal speech recognition.
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Current sample rate
            
        Returns:
            Preprocessed audio array
        """
        try:
            # Resample if needed
            if sample_rate != self.target_sample_rate:
                audio_array = self._resample_audio(audio_array, sample_rate, self.target_sample_rate)
            
            # Apply noise reduction (simple high-pass filter)
            audio_array = self._apply_noise_reduction(audio_array)
            
            # Normalize audio
            audio_array = self._normalize_audio(audio_array)
            
            return audio_array
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            return audio_array
    
    def _resample_audio(self, audio_array: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if from_rate == to_rate:
            return audio_array
        
        # Simple linear interpolation resampling
        # For production, consider using scipy.signal.resample or librosa
        ratio = to_rate / from_rate
        new_length = int(len(audio_array) * ratio)
        
        # Create new time indices
        old_indices = np.arange(len(audio_array))
        new_indices = np.linspace(0, len(audio_array) - 1, new_length)
        
        # Interpolate
        resampled = np.interp(new_indices, old_indices, audio_array)
        
        return resampled
    
    def _apply_noise_reduction(self, audio_array: np.ndarray) -> np.ndarray:
        """Apply basic noise reduction (high-pass filter)."""
        # Simple high-pass filter to remove low-frequency noise
        # This is a basic implementation; for production, use proper DSP libraries
        
        if len(audio_array) < 3:
            return audio_array
        
        # Simple first-order high-pass filter
        alpha = 0.95
        filtered = np.zeros_like(audio_array)
        filtered[0] = audio_array[0]
        
        for i in range(1, len(audio_array)):
            filtered[i] = alpha * (filtered[i-1] + audio_array[i] - audio_array[i-1])
        
        return filtered
    
    def _normalize_audio(self, audio_array: np.ndarray) -> np.ndarray:
        """Normalize audio amplitude."""
        max_val = np.max(np.abs(audio_array))
        if max_val > 0:
            # Normalize to 70% of maximum to avoid clipping
            return audio_array * (0.7 / max_val)
        return audio_array


class SpeechProcessor:
    """
    Main speech processing class using Kyutai speech-to-text models.
    """
    
    def __init__(self, model_id: str = "kyutai/stt-2.6b-en_fr-trfs", device: Optional[str] = None):
        """
        Initialize speech processor with Kyutai model.
        
        Args:
            model_id: HuggingFace model identifier for Kyutai STT
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.processor = None
        self.model = None
        self.audio_validator = AudioValidator()
        self._model_loaded = False
        
        logger.info(f"SpeechProcessor initialized with model: {model_id}")
        logger.info(f"Using device: {self.device}")
    
    def load_model(self) -> bool:
        """
        Load the Kyutai speech-to-text model and processor.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading Kyutai model: {self.model_id}")
            
            # Load processor
            self.processor = KyutaiSpeechToTextProcessor.from_pretrained(self.model_id)
            logger.info("Processor loaded successfully")
            
            # Load model
            self.model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(
                self.model_id,
                device_map=self.device,
                torch_dtype="auto"
            )
            logger.info("Model loaded successfully")
            
            self._model_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Kyutai model: {e}")
            self._model_loaded = False
            return False
    
    def is_model_loaded(self) -> bool:
        """
        Check if the model is loaded and ready.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self._model_loaded and self.processor is not None and self.model is not None
    
    def process_speech(self, audio_data: Union[bytes, np.ndarray], sample_rate: Optional[int] = None) -> SpeechProcessingResult:
        """
        Process speech audio and convert to text.
        
        Args:
            audio_data: Raw audio data as bytes or numpy array
            sample_rate: Sample rate of the audio data
            
        Returns:
            SpeechProcessingResult with transcription and metadata
        """
        import time
        start_time = time.time()
        
        try:
            # Ensure model is loaded
            if not self.is_model_loaded():
                if not self.load_model():
                    return SpeechProcessingResult(
                        success=False,
                        transcribed_text="",
                        confidence=0.0,
                        processing_time=time.time() - start_time,
                        error_message="Failed to load speech recognition model"
                    )
            
            # Validate audio
            validation_result = self.audio_validator.validate_audio(audio_data, sample_rate)
            
            if not validation_result.is_valid:
                error_msg = f"Audio validation failed: {', '.join(validation_result.issues)}"
                logger.warning(error_msg)
                return SpeechProcessingResult(
                    success=False,
                    transcribed_text="",
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    error_message=error_msg
                )
            
            # Convert to numpy array if needed
            if isinstance(audio_data, bytes):
                audio_array, detected_sample_rate = self.audio_validator._bytes_to_array(audio_data)
                if sample_rate is None:
                    sample_rate = detected_sample_rate
            else:
                audio_array = audio_data
                if sample_rate is None:
                    sample_rate = validation_result.sample_rate
            
            # Preprocess audio
            processed_audio = self.audio_validator.preprocess_audio(audio_array, sample_rate)
            
            # Prepare model inputs
            inputs = self.processor(processed_audio)
            inputs = inputs.to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                output_tokens = self.model.generate(**inputs)
            
            # Decode the generated tokens
            transcribed_text = self.processor.batch_decode(output_tokens, skip_special_tokens=True)[0]
            
            # Clean up transcription
            transcribed_text = self._clean_transcription(transcribed_text)
            
            processing_time = time.time() - start_time
            
            # Calculate confidence (placeholder - Kyutai doesn't provide confidence scores directly)
            confidence = self._estimate_confidence(transcribed_text, validation_result)
            
            logger.info(f"Speech processed successfully in {processing_time:.2f}s: '{transcribed_text}'")
            
            return SpeechProcessingResult(
                success=True,
                transcribed_text=transcribed_text,
                confidence=confidence,
                processing_time=processing_time
            )
            
        except Exception as e:
            error_msg = f"Speech processing error: {str(e)}"
            logger.error(error_msg)
            
            return SpeechProcessingResult(
                success=False,
                transcribed_text="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                error_message=error_msg
            )
    
    def _clean_transcription(self, text: str) -> str:
        """
        Clean and normalize transcribed text.
        
        Args:
            text: Raw transcribed text
            
        Returns:
            Cleaned transcription
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove common transcription artifacts
        text = text.replace("[NOISE]", "").replace("[SILENCE]", "")
        text = text.replace("  ", " ").strip()
        
        return text
    
    def _estimate_confidence(self, transcribed_text: str, validation_result: AudioValidationResult) -> float:
        """
        Estimate confidence score for transcription.
        
        Args:
            transcribed_text: Transcribed text
            validation_result: Audio validation result
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # This is a simple heuristic-based confidence estimation
        # In production, you might want to use model-specific confidence measures
        
        confidence = 0.5  # Base confidence
        
        # Adjust based on audio quality
        if len(validation_result.issues) == 0:
            confidence += 0.3
        else:
            confidence -= 0.1 * len(validation_result.issues)
        
        # Adjust based on transcription length and content
        if transcribed_text:
            if len(transcribed_text.split()) >= 2:  # Multiple words
                confidence += 0.2
            if any(char.isalpha() for char in transcribed_text):  # Contains letters
                confidence += 0.1
        else:
            confidence = 0.1  # Very low confidence for empty transcription
        
        # Adjust based on audio duration
        if validation_result.duration > 1.0:  # Longer audio generally more reliable
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_id': self.model_id,
            'device': self.device,
            'loaded': self.is_model_loaded(),
            'target_sample_rate': self.audio_validator.target_sample_rate,
            'supported_languages': ['en', 'fr']  # Kyutai STT supports English and French
        }
    
    def unload_model(self) -> None:
        """
        Unload the model to free memory.
        """
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        self._model_loaded = False
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Speech model unloaded")