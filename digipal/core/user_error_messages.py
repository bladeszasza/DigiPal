"""
User-friendly error messages and recovery guidance for DigiPal.

This module provides comprehensive user-facing error messages and
step-by-step recovery instructions for different error scenarios.
"""

import logging
from typing import Dict, List, Optional, Any
from enum import Enum

from .exceptions import DigiPalException, ErrorCategory, ErrorSeverity

logger = logging.getLogger(__name__)


class MessageTone(Enum):
    """Tone for error messages."""
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    EMPATHETIC = "empathetic"


class UserErrorMessageGenerator:
    """Generates user-friendly error messages and recovery guidance."""
    
    def __init__(self, default_tone: MessageTone = MessageTone.FRIENDLY):
        """Initialize message generator."""
        self.default_tone = default_tone
        self.message_templates = self._initialize_message_templates()
        self.recovery_guides = self._initialize_recovery_guides()
        self.contextual_messages = self._initialize_contextual_messages()
    
    def _initialize_message_templates(self) -> Dict[str, Dict[MessageTone, str]]:
        """Initialize message templates for different tones."""
        return {
            'authentication_failed': {
                MessageTone.FRIENDLY: "Oops! We couldn't log you in. Let's get this sorted out! 🔐",
                MessageTone.PROFESSIONAL: "Authentication failed. Please verify your credentials.",
                MessageTone.CASUAL: "Login didn't work. Let's try again!",
                MessageTone.EMPATHETIC: "We understand login issues can be frustrating. Let's help you get back in."
            },
            'storage_error': {
                MessageTone.FRIENDLY: "We're having trouble saving your DigiPal's data. Don't worry, we'll fix this! 💾",
                MessageTone.PROFESSIONAL: "A storage error occurred. Your data may not have been saved properly.",
                MessageTone.CASUAL: "Couldn't save your stuff. Let's figure this out.",
                MessageTone.EMPATHETIC: "We know how important your DigiPal's progress is. Let's recover your data."
            },
            'ai_model_error': {
                MessageTone.FRIENDLY: "Your DigiPal is having trouble understanding right now. Give us a moment! 🤖",
                MessageTone.PROFESSIONAL: "AI service is temporarily unavailable. Please try again shortly.",
                MessageTone.CASUAL: "The AI is being a bit wonky. Hang tight!",
                MessageTone.EMPATHETIC: "We know you want to chat with your DigiPal. We're working on getting them back online."
            },
            'speech_processing_error': {
                MessageTone.FRIENDLY: "I couldn't quite catch what you said. Could you try speaking again? 🎤",
                MessageTone.PROFESSIONAL: "Speech recognition failed. Please ensure clear audio input.",
                MessageTone.CASUAL: "Didn't catch that. Say it again?",
                MessageTone.EMPATHETIC: "Sometimes speech recognition can be tricky. Let's try a different approach."
            },
            'image_generation_error': {
                MessageTone.FRIENDLY: "We're having trouble creating your DigiPal's image, but they're still there! 🎨",
                MessageTone.PROFESSIONAL: "Image generation service is unavailable. Default images will be used.",
                MessageTone.CASUAL: "Can't make the picture right now, but your DigiPal is fine!",
                MessageTone.EMPATHETIC: "Your DigiPal is beautiful even without a custom image. We'll try again later."
            },
            'network_error': {
                MessageTone.FRIENDLY: "Looks like there's a connection hiccup. Let's try to reconnect! 🌐",
                MessageTone.PROFESSIONAL: "Network connectivity issue detected. Please check your connection.",
                MessageTone.CASUAL: "Internet's acting up. Check your connection?",
                MessageTone.EMPATHETIC: "Connection problems can be annoying. Let's get you back online."
            },
            'pet_lifecycle_error': {
                MessageTone.FRIENDLY: "Something went wrong with your DigiPal's growth. Let's help them! 🐣",
                MessageTone.PROFESSIONAL: "Pet lifecycle error occurred. Data integrity may be compromised.",
                MessageTone.CASUAL: "Your DigiPal hit a snag. Let's fix them up!",
                MessageTone.EMPATHETIC: "We understand your DigiPal means a lot to you. Let's restore them safely."
            },
            'system_error': {
                MessageTone.FRIENDLY: "Something unexpected happened, but we're on it! 🔧",
                MessageTone.PROFESSIONAL: "A system error occurred. Please try restarting the application.",
                MessageTone.CASUAL: "Things got weird. Maybe restart?",
                MessageTone.EMPATHETIC: "Technical issues can be frustrating. We're here to help you through this."
            }
        }
    
    def _initialize_recovery_guides(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize step-by-step recovery guides."""
        return {
            'authentication': [
                {
                    'step': 1,
                    'title': 'Check Your Credentials',
                    'description': 'Make sure your HuggingFace username and token are correct',
                    'action': 'verify_credentials',
                    'difficulty': 'easy'
                },
                {
                    'step': 2,
                    'title': 'Test Your Connection',
                    'description': 'Ensure you have a stable internet connection',
                    'action': 'test_connection',
                    'difficulty': 'easy'
                },
                {
                    'step': 3,
                    'title': 'Try Offline Mode',
                    'description': 'Use the application in offline mode with limited features',
                    'action': 'enable_offline_mode',
                    'difficulty': 'easy'
                },
                {
                    'step': 4,
                    'title': 'Clear Browser Data',
                    'description': 'Clear cookies and cached data, then try logging in again',
                    'action': 'clear_browser_data',
                    'difficulty': 'medium'
                }
            ],
            'storage': [
                {
                    'step': 1,
                    'title': 'Check Disk Space',
                    'description': 'Make sure you have at least 100MB of free disk space',
                    'action': 'check_disk_space',
                    'difficulty': 'easy'
                },
                {
                    'step': 2,
                    'title': 'Restart Application',
                    'description': 'Close and reopen DigiPal to reset the storage connection',
                    'action': 'restart_application',
                    'difficulty': 'easy'
                },
                {
                    'step': 3,
                    'title': 'Restore from Backup',
                    'description': 'Use an automatic backup to restore your DigiPal data',
                    'action': 'restore_backup',
                    'difficulty': 'medium'
                },
                {
                    'step': 4,
                    'title': 'Check File Permissions',
                    'description': 'Ensure DigiPal has permission to write to its data directory',
                    'action': 'check_permissions',
                    'difficulty': 'hard'
                }
            ],
            'ai_model': [
                {
                    'step': 1,
                    'title': 'Wait and Retry',
                    'description': 'AI services may be temporarily busy. Wait 30 seconds and try again',
                    'action': 'wait_and_retry',
                    'difficulty': 'easy'
                },
                {
                    'step': 2,
                    'title': 'Use Simple Commands',
                    'description': 'Try using shorter, simpler phrases when talking to your DigiPal',
                    'action': 'use_simple_commands',
                    'difficulty': 'easy'
                },
                {
                    'step': 3,
                    'title': 'Switch to Text Mode',
                    'description': 'Use text input instead of speech if available',
                    'action': 'switch_to_text',
                    'difficulty': 'easy'
                },
                {
                    'step': 4,
                    'title': 'Restart Application',
                    'description': 'Close and reopen DigiPal to reload the AI models',
                    'action': 'restart_application',
                    'difficulty': 'medium'
                }
            ],
            'speech_processing': [
                {
                    'step': 1,
                    'title': 'Check Microphone',
                    'description': 'Make sure your microphone is connected and working',
                    'action': 'check_microphone',
                    'difficulty': 'easy'
                },
                {
                    'step': 2,
                    'title': 'Reduce Background Noise',
                    'description': 'Find a quieter location or reduce background noise',
                    'action': 'reduce_noise',
                    'difficulty': 'easy'
                },
                {
                    'step': 3,
                    'title': 'Speak Clearly',
                    'description': 'Speak slowly and clearly, facing the microphone',
                    'action': 'speak_clearly',
                    'difficulty': 'easy'
                },
                {
                    'step': 4,
                    'title': 'Use Text Input',
                    'description': 'Switch to typing your messages instead of speaking',
                    'action': 'use_text_input',
                    'difficulty': 'easy'
                }
            ],
            'network': [
                {
                    'step': 1,
                    'title': 'Check Internet Connection',
                    'description': 'Make sure you\'re connected to the internet',
                    'action': 'check_internet',
                    'difficulty': 'easy'
                },
                {
                    'step': 2,
                    'title': 'Try Different Network',
                    'description': 'Switch to a different WiFi network or use mobile data',
                    'action': 'switch_network',
                    'difficulty': 'easy'
                },
                {
                    'step': 3,
                    'title': 'Use Offline Mode',
                    'description': 'Continue using DigiPal with cached data and limited features',
                    'action': 'enable_offline_mode',
                    'difficulty': 'easy'
                },
                {
                    'step': 4,
                    'title': 'Check Firewall Settings',
                    'description': 'Ensure DigiPal is allowed through your firewall',
                    'action': 'check_firewall',
                    'difficulty': 'hard'
                }
            ],
            'pet_lifecycle': [
                {
                    'step': 1,
                    'title': 'Reload Your DigiPal',
                    'description': 'Try refreshing the page or restarting the application',
                    'action': 'reload_pet',
                    'difficulty': 'easy'
                },
                {
                    'step': 2,
                    'title': 'Check Recent Backups',
                    'description': 'Look for recent automatic backups of your DigiPal',
                    'action': 'check_backups',
                    'difficulty': 'medium'
                },
                {
                    'step': 3,
                    'title': 'Restore from Backup',
                    'description': 'Restore your DigiPal from the most recent backup',
                    'action': 'restore_from_backup',
                    'difficulty': 'medium'
                },
                {
                    'step': 4,
                    'title': 'Contact Support',
                    'description': 'If all else fails, contact support for manual data recovery',
                    'action': 'contact_support',
                    'difficulty': 'easy'
                }
            ]
        }
    
    def _initialize_contextual_messages(self) -> Dict[str, Dict[str, str]]:
        """Initialize contextual messages based on user state."""
        return {
            'first_time_user': {
                'authentication': "Welcome to DigiPal! Let's get you set up with your HuggingFace account.",
                'storage': "We're setting up your DigiPal's home. This might take a moment.",
                'ai_model': "Your DigiPal is learning to talk! This may take a few minutes on first startup."
            },
            'returning_user': {
                'authentication': "Welcome back! Let's get you reconnected to your DigiPal.",
                'storage': "Loading your DigiPal's data. They've missed you!",
                'ai_model': "Your DigiPal is waking up and getting ready to chat!"
            },
            'during_evolution': {
                'pet_lifecycle': "Your DigiPal is in the middle of evolving! Let's make sure this goes smoothly.",
                'storage': "We're saving your DigiPal's evolution progress. This is important!"
            },
            'during_interaction': {
                'speech_processing': "Your DigiPal is listening carefully. Let's make sure they can hear you clearly.",
                'ai_model': "Your DigiPal is thinking about what to say. Sometimes they need a moment!"
            }
        }
    
    def generate_user_message(
        self, 
        error: DigiPalException, 
        tone: Optional[MessageTone] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a user-friendly error message.
        
        Args:
            error: The error that occurred
            tone: Message tone to use
            user_context: Additional user context
            
        Returns:
            User-friendly error message
        """
        tone = tone or self.default_tone
        user_context = user_context or {}
        
        # Get base message template
        error_type = self._get_error_type_key(error)
        base_message = self.message_templates.get(error_type, {}).get(
            tone, 
            "Something unexpected happened, but we're working on it!"
        )
        
        # Add contextual information
        contextual_key = user_context.get('user_state', 'general')
        if contextual_key in self.contextual_messages:
            contextual_msg = self.contextual_messages[contextual_key].get(error.category.value)
            if contextual_msg:
                base_message = f"{contextual_msg} {base_message}"
        
        # Add severity-specific information
        if error.severity == ErrorSeverity.CRITICAL:
            base_message += " This is a critical issue that needs immediate attention."
        elif error.severity == ErrorSeverity.HIGH:
            base_message += " This is important to fix to ensure your DigiPal works properly."
        
        return base_message
    
    def get_recovery_guide(
        self, 
        error: DigiPalException,
        max_steps: int = 4,
        difficulty_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get step-by-step recovery guide for an error.
        
        Args:
            error: The error that occurred
            max_steps: Maximum number of steps to return
            difficulty_filter: Filter by difficulty ('easy', 'medium', 'hard')
            
        Returns:
            List of recovery steps
        """
        category_key = error.category.value
        guide_steps = self.recovery_guides.get(category_key, [])
        
        # Filter by difficulty if specified
        if difficulty_filter:
            guide_steps = [step for step in guide_steps if step['difficulty'] == difficulty_filter]
        
        # Limit number of steps
        guide_steps = guide_steps[:max_steps]
        
        # Add error-specific context to steps
        for step in guide_steps:
            step['error_code'] = error.error_code
            step['error_category'] = error.category.value
            
            # Add specific instructions based on error context
            if error.context:
                step['context'] = error.context
        
        return guide_steps
    
    def _get_error_type_key(self, error: DigiPalException) -> str:
        """Get the error type key for message templates."""
        category_mapping = {
            ErrorCategory.AUTHENTICATION: 'authentication_failed',
            ErrorCategory.STORAGE: 'storage_error',
            ErrorCategory.AI_MODEL: 'ai_model_error',
            ErrorCategory.SPEECH_PROCESSING: 'speech_processing_error',
            ErrorCategory.IMAGE_GENERATION: 'image_generation_error',
            ErrorCategory.NETWORK: 'network_error',
            ErrorCategory.PET_LIFECYCLE: 'pet_lifecycle_error',
            ErrorCategory.SYSTEM: 'system_error'
        }
        
        return category_mapping.get(error.category, 'system_error')
    
    def generate_progress_message(self, recovery_step: Dict[str, Any]) -> str:
        """
        Generate a progress message for a recovery step.
        
        Args:
            recovery_step: Recovery step information
            
        Returns:
            Progress message
        """
        step_num = recovery_step.get('step', 1)
        title = recovery_step.get('title', 'Recovery Step')
        
        progress_messages = [
            f"Step {step_num}: {title}",
            f"Working on: {title}",
            f"Now trying: {title}",
            f"Attempting: {title}"
        ]
        
        # Choose message based on step number
        message_index = (step_num - 1) % len(progress_messages)
        return progress_messages[message_index]
    
    def generate_success_message(self, error_category: str, recovery_method: str) -> str:
        """
        Generate a success message after recovery.
        
        Args:
            error_category: Category of error that was recovered
            recovery_method: Method used for recovery
            
        Returns:
            Success message
        """
        success_messages = {
            'authentication': "Great! You're logged in and ready to go! 🎉",
            'storage': "Perfect! Your DigiPal's data is safe and sound! 💾",
            'ai_model': "Awesome! Your DigiPal is ready to chat again! 🤖",
            'speech_processing': "Excellent! Your DigiPal can hear you clearly now! 🎤",
            'network': "Wonderful! You're back online! 🌐",
            'pet_lifecycle': "Amazing! Your DigiPal is healthy and happy! 🐣"
        }
        
        base_message = success_messages.get(error_category, "Great! The issue has been resolved! ✅")
        
        if recovery_method:
            base_message += f" (Fixed using: {recovery_method})"
        
        return base_message


# Global message generator instance
user_message_generator = UserErrorMessageGenerator()


def get_user_friendly_error_message(
    error: DigiPalException,
    tone: Optional[MessageTone] = None,
    user_context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Get a user-friendly error message.
    
    Args:
        error: The error that occurred
        tone: Message tone to use
        user_context: Additional user context
        
    Returns:
        User-friendly error message
    """
    return user_message_generator.generate_user_message(error, tone, user_context)


def get_recovery_guide(
    error: DigiPalException,
    max_steps: int = 4,
    difficulty_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get recovery guide for an error.
    
    Args:
        error: The error that occurred
        max_steps: Maximum number of steps
        difficulty_filter: Filter by difficulty
        
    Returns:
        List of recovery steps
    """
    return user_message_generator.get_recovery_guide(error, max_steps, difficulty_filter)