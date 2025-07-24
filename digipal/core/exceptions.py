"""
Custom exception classes for DigiPal application.

This module defines all custom exceptions used throughout the DigiPal system
for better error handling and user experience.
"""

from typing import Optional, Dict, Any
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels for categorizing exceptions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better error classification."""
    AUTHENTICATION = "authentication"
    STORAGE = "storage"
    AI_MODEL = "ai_model"
    SPEECH_PROCESSING = "speech_processing"
    IMAGE_GENERATION = "image_generation"
    PET_LIFECYCLE = "pet_lifecycle"
    MCP_PROTOCOL = "mcp_protocol"
    NETWORK = "network"
    VALIDATION = "validation"
    SYSTEM = "system"


class DigiPalException(Exception):
    """Base exception class for all DigiPal-specific errors."""
    
    def __init__(
        self, 
        message: str, 
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        user_message: Optional[str] = None,
        recovery_suggestions: Optional[list] = None,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize DigiPal exception.
        
        Args:
            message: Technical error message for logging
            category: Error category for classification
            severity: Error severity level
            user_message: User-friendly error message
            recovery_suggestions: List of recovery suggestions for users
            error_code: Unique error code for tracking
            context: Additional context information
        """
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.user_message = user_message or self._generate_user_message()
        self.recovery_suggestions = recovery_suggestions or []
        self.error_code = error_code
        self.context = context or {}
    
    def _generate_user_message(self) -> str:
        """Generate a user-friendly message based on the category."""
        category_messages = {
            ErrorCategory.AUTHENTICATION: "There was a problem with authentication. Please try logging in again.",
            ErrorCategory.STORAGE: "There was a problem saving or loading your DigiPal data.",
            ErrorCategory.AI_MODEL: "The AI system is having trouble responding. Please try again.",
            ErrorCategory.SPEECH_PROCESSING: "I couldn't understand your speech. Please try speaking again.",
            ErrorCategory.IMAGE_GENERATION: "There was a problem generating your DigiPal's image.",
            ErrorCategory.PET_LIFECYCLE: "There was a problem with your DigiPal's lifecycle management.",
            ErrorCategory.MCP_PROTOCOL: "There was a problem with external system communication.",
            ErrorCategory.NETWORK: "There was a network connection problem. Please check your internet.",
            ErrorCategory.VALIDATION: "The provided information is not valid. Please check and try again.",
            ErrorCategory.SYSTEM: "A system error occurred. Please try again."
        }
        return category_messages.get(self.category, "An unexpected error occurred.")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging and serialization."""
        return {
            'message': str(self),
            'category': self.category.value,
            'severity': self.severity.value,
            'user_message': self.user_message,
            'recovery_suggestions': self.recovery_suggestions,
            'error_code': self.error_code,
            'context': self.context
        }


class AuthenticationError(DigiPalException):
    """Exception raised for authentication-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            recovery_suggestions=[
                "Check your HuggingFace credentials",
                "Ensure you have a stable internet connection",
                "Try refreshing the page and logging in again"
            ],
            **kwargs
        )


class StorageError(DigiPalException):
    """Exception raised for storage and database-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.STORAGE,
            severity=ErrorSeverity.HIGH,
            recovery_suggestions=[
                "Check if you have sufficient disk space",
                "Try restarting the application",
                "Contact support if the problem persists"
            ],
            **kwargs
        )


class AIModelError(DigiPalException):
    """Exception raised for AI model-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.AI_MODEL,
            severity=ErrorSeverity.MEDIUM,
            recovery_suggestions=[
                "Try your request again in a moment",
                "Use simpler language if speaking to your DigiPal",
                "Check your internet connection"
            ],
            **kwargs
        )


class SpeechProcessingError(DigiPalException):
    """Exception raised for speech processing errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.SPEECH_PROCESSING,
            severity=ErrorSeverity.LOW,
            recovery_suggestions=[
                "Speak more clearly and slowly",
                "Check your microphone permissions",
                "Try using text input instead",
                "Reduce background noise"
            ],
            **kwargs
        )


class ImageGenerationError(DigiPalException):
    """Exception raised for image generation errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.IMAGE_GENERATION,
            severity=ErrorSeverity.LOW,
            recovery_suggestions=[
                "A default image will be used instead",
                "Try again later when the service is available",
                "Check your internet connection"
            ],
            **kwargs
        )


class PetLifecycleError(DigiPalException):
    """Exception raised for pet lifecycle management errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.PET_LIFECYCLE,
            severity=ErrorSeverity.HIGH,
            recovery_suggestions=[
                "Try reloading your DigiPal",
                "Check if your DigiPal data is corrupted",
                "Contact support for data recovery"
            ],
            **kwargs
        )


class MCPProtocolError(DigiPalException):
    """Exception raised for MCP protocol-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.MCP_PROTOCOL,
            severity=ErrorSeverity.MEDIUM,
            recovery_suggestions=[
                "Check the MCP client configuration",
                "Verify authentication credentials",
                "Try reconnecting to the MCP server"
            ],
            **kwargs
        )


class NetworkError(DigiPalException):
    """Exception raised for network-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            recovery_suggestions=[
                "Check your internet connection",
                "Try again in a few moments",
                "Switch to offline mode if available"
            ],
            **kwargs
        )


class ValidationError(DigiPalException):
    """Exception raised for data validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            recovery_suggestions=[
                "Check the format of your input",
                "Ensure all required fields are filled",
                "Try with different values"
            ],
            **kwargs
        )
        if field:
            self.context['field'] = field


class SystemError(DigiPalException):
    """Exception raised for general system errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.HIGH,
            recovery_suggestions=[
                "Try restarting the application",
                "Check system resources (memory, disk space)",
                "Contact support if the problem persists"
            ],
            **kwargs
        )


class RecoveryError(DigiPalException):
    """Exception raised when recovery operations fail."""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            recovery_suggestions=[
                "Manual intervention may be required",
                "Contact support immediately",
                "Do not attempt further operations"
            ],
            **kwargs
        )
        if original_error:
            self.context['original_error'] = str(original_error)