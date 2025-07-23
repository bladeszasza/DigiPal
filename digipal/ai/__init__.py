"""
AI communication layer including speech processing and language models.
"""

from .communication import (
    AICommunication,
    CommandInterpreter,
    ResponseGenerator,
    ConversationMemoryManager
)
from .speech_processor import (
    SpeechProcessor,
    AudioValidator,
    SpeechProcessingResult,
    AudioValidationResult
)

__all__ = [
    'AICommunication',
    'CommandInterpreter', 
    'ResponseGenerator',
    'ConversationMemoryManager',
    'SpeechProcessor',
    'AudioValidator',
    'SpeechProcessingResult',
    'AudioValidationResult'
]