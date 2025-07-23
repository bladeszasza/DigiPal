#!/usr/bin/env python3
"""
Demo script for Kyutai speech processing integration.

This script demonstrates the SpeechProcessor functionality with mock audio data.
"""

import numpy as np
import sys
import os

# Add the parent directory to the path so we can import digipal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from digipal.ai.speech_processor import SpeechProcessor, AudioValidator


def create_mock_audio(duration: float = 2.0, sample_rate: int = 24000) -> np.ndarray:
    """
    Create mock audio data for testing.
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        Mock audio array
    """
    # Create a simple sine wave with some noise
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # A4 note
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    # Add some noise to make it more realistic
    noise = 0.05 * np.random.normal(0, 1, len(audio))
    audio = audio + noise
    
    return audio.astype(np.float32)


def demo_audio_validation():
    """Demonstrate audio validation functionality."""
    print("=== Audio Validation Demo ===")
    
    validator = AudioValidator()
    
    # Test with valid audio
    print("\n1. Testing with valid audio:")
    valid_audio = create_mock_audio(duration=2.0, sample_rate=24000)
    result = validator.validate_audio(valid_audio, sample_rate=24000)
    
    print(f"   Valid: {result.is_valid}")
    print(f"   Duration: {result.duration:.2f}s")
    print(f"   Sample Rate: {result.sample_rate}Hz")
    print(f"   Channels: {result.channels}")
    print(f"   Issues: {result.issues}")
    
    # Test with invalid audio (too short)
    print("\n2. Testing with invalid audio (too short):")
    short_audio = create_mock_audio(duration=0.05, sample_rate=24000)
    result = validator.validate_audio(short_audio, sample_rate=24000)
    
    print(f"   Valid: {result.is_valid}")
    print(f"   Duration: {result.duration:.2f}s")
    print(f"   Issues: {result.issues}")
    
    # Test with wrong sample rate
    print("\n3. Testing with wrong sample rate:")
    wrong_rate_audio = create_mock_audio(duration=1.0, sample_rate=16000)
    result = validator.validate_audio(wrong_rate_audio, sample_rate=16000)
    
    print(f"   Valid: {result.is_valid}")
    print(f"   Sample Rate: {result.sample_rate}Hz")
    print(f"   Issues: {result.issues}")


def demo_speech_processor():
    """Demonstrate speech processor functionality."""
    print("\n=== Speech Processor Demo ===")
    
    processor = SpeechProcessor()
    
    # Get model info
    print("\n1. Model Information:")
    info = processor.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Note: We won't actually load the model in this demo since it requires
    # downloading large model files. In a real scenario, you would:
    
    print("\n2. Model Loading (simulated):")
    print("   In a real scenario, you would call:")
    print("   success = processor.load_model()")
    print("   This would download and load the Kyutai model.")
    
    print("\n3. Speech Processing (simulated):")
    print("   To process real audio, you would call:")
    print("   result = processor.process_speech(audio_data, sample_rate=24000)")
    print("   This would return a SpeechProcessingResult with:")
    print("   - success: bool")
    print("   - transcribed_text: str")
    print("   - confidence: float")
    print("   - processing_time: float")
    print("   - error_message: Optional[str]")
    
    # Demonstrate preprocessing
    print("\n4. Audio Preprocessing:")
    audio = create_mock_audio(duration=1.0, sample_rate=16000)
    print(f"   Original audio: {len(audio)} samples at 16kHz")
    
    preprocessed = processor.audio_validator.preprocess_audio(audio, 16000)
    print(f"   Preprocessed audio: {len(preprocessed)} samples")
    print("   - Resampled to 24kHz")
    print("   - Noise reduction applied")
    print("   - Normalized amplitude")


def demo_error_handling():
    """Demonstrate error handling capabilities."""
    print("\n=== Error Handling Demo ===")
    
    processor = SpeechProcessor()
    
    # Test with invalid audio
    print("\n1. Processing invalid audio:")
    invalid_audio = np.array([])  # Empty audio
    
    # This would normally call the actual processing, but we'll simulate
    print("   Attempting to process empty audio array...")
    print("   Expected result: SpeechProcessingResult with success=False")
    print("   Error message would indicate audio validation failure")
    
    # Test with model not loaded
    print("\n2. Processing without loaded model:")
    print("   Attempting to process audio without loading model...")
    print("   Expected result: SpeechProcessingResult with success=False")
    print("   Error message would indicate model loading failure")


def main():
    """Run all demos."""
    print("Kyutai Speech Processing Integration Demo")
    print("=" * 50)
    
    try:
        demo_audio_validation()
        demo_speech_processor()
        demo_error_handling()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("\nTo use the speech processor in production:")
        print("1. Install required dependencies: transformers, torch, numpy")
        print("2. Create a SpeechProcessor instance")
        print("3. Load the model with processor.load_model()")
        print("4. Process audio with processor.process_speech(audio_data)")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())