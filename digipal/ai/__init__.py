"""
AI communication layer including speech processing and language models.
"""

from .communication import (
    AICommunication,
    CommandInterpreter,
    ResponseGenerator,
    ConversationMemoryManager
)

__all__ = [
    'AICommunication',
    'CommandInterpreter', 
    'ResponseGenerator',
    'ConversationMemoryManager'
]