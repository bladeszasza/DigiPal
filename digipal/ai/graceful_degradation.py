"""
Graceful degradation system for AI models in DigiPal.

This module provides fallback mechanisms when AI models fail,
ensuring the application continues to function with reduced capabilities.
"""

import logging
import random
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum

from ..core.models import DigiPal, Interaction
from ..core.enums import LifeStage, InteractionResult
from ..core.exceptions import AIModelError, DigiPalException
from ..core.error_handler import with_error_handling, CircuitBreaker, CircuitBreakerConfig

logger = logging.getLogger(__name__)


class DegradationLevel(Enum):
    """Levels of service degradation."""
    FULL_SERVICE = "full_service"
    REDUCED_FEATURES = "reduced_features"
    BASIC_RESPONSES = "basic_responses"
    MINIMAL_FUNCTION = "minimal_function"
    EMERGENCY_MODE = "emergency_mode"


class FallbackResponseGenerator:
    """Generates fallback responses when AI models are unavailable."""
    
    def __init__(self):
        """Initialize fallback response generator."""
        self.response_templates = self._initialize_response_templates()
        self.command_responses = self._initialize_command_responses()
        self.personality_modifiers = self._initialize_personality_modifiers()
    
    def _initialize_response_templates(self) -> Dict[LifeStage, List[str]]:
        """Initialize response templates for each life stage."""
        return {
            LifeStage.EGG: [
                "*The egg glows softly*",
                "*The egg trembles slightly*",
                "*The egg remains warm and quiet*",
                "*You sense movement inside the egg*"
            ],
            LifeStage.BABY: [
                "*baby sounds*",
                "Goo goo!",
                "*giggles*",
                "Mama?",
                "*curious baby noises*",
                "Baba!",
                "*happy gurgling*"
            ],
            LifeStage.CHILD: [
                "I'm having fun!",
                "What's that?",
                "Can we play?",
                "I'm learning!",
                "That's cool!",
                "I want to explore!",
                "Tell me more!"
            ],
            LifeStage.TEEN: [
                "That's interesting...",
                "I guess that's okay.",
                "Whatever you say.",
                "I'm figuring things out.",
                "That's pretty cool, I suppose.",
                "I'm growing up fast!",
                "Things are changing..."
            ],
            LifeStage.YOUNG_ADULT: [
                "I understand what you mean.",
                "That makes sense to me.",
                "I'm ready for anything!",
                "Let's tackle this together.",
                "I feel confident about this.",
                "I'm at my peak right now!",
                "What's our next adventure?"
            ],
            LifeStage.ADULT: [
                "I've learned a lot over the years.",
                "That's a wise perspective.",
                "Let me share my experience with you.",
                "I understand the deeper meaning.",
                "Maturity brings clarity.",
                "I'm here to guide you.",
                "Experience has taught me much."
            ],
            LifeStage.ELDERLY: [
                "Ah, yes... I remember...",
                "In my long life, I've seen...",
                "Time passes so quickly...",
                "Let me tell you about the old days...",
                "Wisdom comes with age...",
                "I cherish these moments with you.",
                "My memories are precious to me."
            ]
        }
    
    def _initialize_command_responses(self) -> Dict[str, Dict[LifeStage, List[str]]]:
        """Initialize responses for specific commands."""
        return {
            'eat': {
                LifeStage.BABY: ["*nom nom*", "Yummy!", "*happy eating sounds*"],
                LifeStage.CHILD: ["This tastes good!", "I'm hungry!", "Thank you for feeding me!"],
                LifeStage.TEEN: ["Thanks, I needed that.", "Food is fuel, right?", "Not bad."],
                LifeStage.YOUNG_ADULT: ["Perfect timing, I was getting hungry.", "This will give me energy!", "Thanks for taking care of me."],
                LifeStage.ADULT: ["I appreciate you looking after my needs.", "This nourishment is welcome.", "Thank you for your care."],
                LifeStage.ELDERLY: ["Ah, you still take such good care of me...", "Food tastes different now, but I'm grateful.", "Thank you, dear friend."]
            },
            'sleep': {
                LifeStage.BABY: ["*yawn*", "Sleepy time...", "*closes eyes*"],
                LifeStage.CHILD: ["I'm getting tired!", "Can I take a nap?", "Sleep sounds good!"],
                LifeStage.TEEN: ["I could use some rest.", "Sleep is important, I guess.", "Fine, I'll rest."],
                LifeStage.YOUNG_ADULT: ["Rest will help me perform better.", "Good idea, I need to recharge.", "Sleep is essential for peak performance."],
                LifeStage.ADULT: ["Rest is wisdom.", "I'll take this time to reflect.", "Sleep brings clarity."],
                LifeStage.ELDERLY: ["Rest comes easier now...", "I dream of old times...", "Sleep is peaceful at my age."]
            },
            'good': {
                LifeStage.BABY: ["*happy baby sounds*", "Goo!", "*giggles with joy*"],
                LifeStage.CHILD: ["Yay! I did good!", "I'm happy!", "Thank you!"],
                LifeStage.TEEN: ["Thanks, I try.", "That means something.", "Cool, thanks."],
                LifeStage.YOUNG_ADULT: ["I appreciate the recognition!", "That motivates me!", "Thanks for the positive feedback!"],
                LifeStage.ADULT: ["Your approval means a lot to me.", "I strive to do my best.", "Thank you for acknowledging my efforts."],
                LifeStage.ELDERLY: ["Your kind words warm my heart...", "After all these years, praise still matters...", "Thank you, my dear friend."]
            },
            'bad': {
                LifeStage.BABY: ["*sad baby sounds*", "Waaah!", "*confused crying*"],
                LifeStage.CHILD: ["I'm sorry!", "I didn't mean to!", "I'll try better!"],
                LifeStage.TEEN: ["Whatever.", "I don't care.", "Fine, I get it."],
                LifeStage.YOUNG_ADULT: ["I understand. I'll do better.", "Point taken.", "I'll learn from this."],
                LifeStage.ADULT: ["I accept your criticism.", "I'll reflect on this.", "Thank you for your honesty."],
                LifeStage.ELDERLY: ["I'm sorry to disappoint you...", "Even at my age, I can still learn...", "I understand your concern."]
            },
            'play': {
                LifeStage.BABY: ["*excited baby sounds*", "Play! Play!", "*happy wiggling*"],
                LifeStage.CHILD: ["Yes! Let's play!", "This is fun!", "I love playing!"],
                LifeStage.TEEN: ["I guess playing is okay.", "Sure, why not.", "Playing can be fun sometimes."],
                LifeStage.YOUNG_ADULT: ["Great idea! Let's have some fun!", "Play is important for balance!", "I'm ready to play!"],
                LifeStage.ADULT: ["Play keeps the spirit young.", "I enjoy our time together.", "Even adults need to play."],
                LifeStage.ELDERLY: ["Playing brings back memories...", "I may be slow, but I still enjoy fun...", "These moments are precious."]
            },
            'train': {
                LifeStage.CHILD: ["I want to get stronger!", "Training is hard but fun!", "I'm learning!"],
                LifeStage.TEEN: ["Training is important, I guess.", "I'll get stronger.", "This is challenging."],
                LifeStage.YOUNG_ADULT: ["Let's push my limits!", "Training makes me stronger!", "I'm ready for the challenge!"],
                LifeStage.ADULT: ["Discipline and training build character.", "I'll give my best effort.", "Training is a lifelong journey."],
                LifeStage.ELDERLY: ["I may be old, but I can still try...", "Training keeps me active...", "My body may be slower, but my spirit is strong."]
            }
        }
    
    def _initialize_personality_modifiers(self) -> Dict[str, List[str]]:
        """Initialize personality-based response modifiers."""
        return {
            'friendly': [" *smiles warmly*", " *friendly gesture*", " *welcoming tone*"],
            'shy': [" *looks down shyly*", " *quiet voice*", " *hesitant*"],
            'playful': [" *bounces excitedly*", " *playful grin*", " *mischievous look*"],
            'serious': [" *thoughtful expression*", " *serious tone*", " *focused*"],
            'curious': [" *tilts head curiously*", " *eyes light up*", " *interested*"],
            'calm': [" *peaceful demeanor*", " *serene*", " *tranquil*"]
        }
    
    def generate_fallback_response(
        self, 
        user_input: str, 
        pet: DigiPal, 
        command: Optional[str] = None,
        degradation_level: DegradationLevel = DegradationLevel.BASIC_RESPONSES
    ) -> str:
        """
        Generate a fallback response when AI models are unavailable.
        
        Args:
            user_input: User's input text
            pet: DigiPal instance
            command: Interpreted command (if any)
            degradation_level: Level of service degradation
            
        Returns:
            Fallback response string
        """
        try:
            # Handle different degradation levels
            if degradation_level == DegradationLevel.EMERGENCY_MODE:
                return self._generate_emergency_response(pet)
            
            # Try command-specific responses first
            if command and command in self.command_responses:
                command_templates = self.command_responses[command].get(pet.life_stage, [])
                if command_templates:
                    response = random.choice(command_templates)
                    return self._apply_personality_modifier(response, pet)
            
            # Fall back to general responses
            general_templates = self.response_templates.get(pet.life_stage, [])
            if general_templates:
                response = random.choice(general_templates)
                return self._apply_personality_modifier(response, pet)
            
            # Ultimate fallback
            return self._generate_emergency_response(pet)
            
        except Exception as e:
            logger.error(f"Fallback response generation failed: {e}")
            return "*DigiPal is resting*"
    
    def _apply_personality_modifier(self, response: str, pet: DigiPal) -> str:
        """Apply personality-based modifiers to response."""
        try:
            if not pet.personality_traits:
                return response
            
            # Find dominant personality trait
            dominant_trait = max(pet.personality_traits.items(), key=lambda x: x[1])
            trait_name, trait_value = dominant_trait
            
            # Apply modifier if trait is strong enough
            if trait_value > 0.7 and trait_name in self.personality_modifiers:
                modifier = random.choice(self.personality_modifiers[trait_name])
                return response + modifier
            
            return response
            
        except Exception:
            return response
    
    def _generate_emergency_response(self, pet: DigiPal) -> str:
        """Generate minimal emergency response."""
        emergency_responses = {
            LifeStage.EGG: "*egg*",
            LifeStage.BABY: "*baby*",
            LifeStage.CHILD: "Hi!",
            LifeStage.TEEN: "Hey.",
            LifeStage.YOUNG_ADULT: "Hello!",
            LifeStage.ADULT: "Greetings.",
            LifeStage.ELDERLY: "Hello, friend."
        }
        
        return emergency_responses.get(pet.life_stage, "*DigiPal*")


class AIServiceManager:
    """Manages AI service availability and degradation."""
    
    def __init__(self):
        """Initialize AI service manager."""
        self.service_status: Dict[str, bool] = {
            'language_model': True,
            'speech_processing': True,
            'image_generation': True
        }
        
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.fallback_generator = FallbackResponseGenerator()
        self.current_degradation_level = DegradationLevel.FULL_SERVICE
        
        # Initialize circuit breakers
        self._initialize_circuit_breakers()
    
    def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for AI services."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=300.0,  # 5 minutes
            expected_exception=AIModelError
        )
        
        for service in self.service_status.keys():
            self.circuit_breakers[service] = CircuitBreaker(config)
    
    def call_ai_service(
        self, 
        service_name: str, 
        func: Callable, 
        fallback_func: Optional[Callable] = None,
        *args, 
        **kwargs
    ) -> Any:
        """
        Call an AI service with circuit breaker protection.
        
        Args:
            service_name: Name of the AI service
            func: Function to call
            fallback_func: Fallback function if service fails
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Service result or fallback result
        """
        try:
            circuit_breaker = self.circuit_breakers.get(service_name)
            if circuit_breaker:
                result = circuit_breaker.call(func, *args, **kwargs)
                self.service_status[service_name] = True
                self._update_degradation_level()
                return result
            else:
                return func(*args, **kwargs)
                
        except Exception as e:
            logger.warning(f"AI service {service_name} failed: {e}")
            self.service_status[service_name] = False
            self._update_degradation_level()
            
            if fallback_func:
                try:
                    return fallback_func(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"Fallback for {service_name} also failed: {fallback_error}")
            
            raise AIModelError(f"AI service {service_name} unavailable: {str(e)}")
    
    def _update_degradation_level(self):
        """Update current degradation level based on service status."""
        available_services = sum(1 for status in self.service_status.values() if status)
        total_services = len(self.service_status)
        
        if available_services == total_services:
            self.current_degradation_level = DegradationLevel.FULL_SERVICE
        elif available_services >= total_services * 0.75:
            self.current_degradation_level = DegradationLevel.REDUCED_FEATURES
        elif available_services >= total_services * 0.5:
            self.current_degradation_level = DegradationLevel.BASIC_RESPONSES
        elif available_services > 0:
            self.current_degradation_level = DegradationLevel.MINIMAL_FUNCTION
        else:
            self.current_degradation_level = DegradationLevel.EMERGENCY_MODE
        
        logger.info(f"Degradation level updated to: {self.current_degradation_level.value}")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status and degradation level."""
        return {
            'services': dict(self.service_status),
            'degradation_level': self.current_degradation_level.value,
            'circuit_breakers': {
                name: {
                    'state': cb.state,
                    'failure_count': cb.failure_count,
                    'last_failure': cb.last_failure_time.isoformat() if cb.last_failure_time else None
                }
                for name, cb in self.circuit_breakers.items()
            }
        }
    
    def force_service_recovery(self, service_name: str):
        """Force recovery attempt for a specific service."""
        if service_name in self.circuit_breakers:
            circuit_breaker = self.circuit_breakers[service_name]
            circuit_breaker.state = "half-open"
            circuit_breaker.failure_count = 0
            logger.info(f"Forced recovery attempt for service: {service_name}")
    
    def generate_degraded_response(
        self, 
        user_input: str, 
        pet: DigiPal, 
        command: Optional[str] = None
    ) -> Interaction:
        """
        Generate a response using degraded AI capabilities.
        
        Args:
            user_input: User's input text
            pet: DigiPal instance
            command: Interpreted command (if any)
            
        Returns:
            Interaction with fallback response
        """
        try:
            response = self.fallback_generator.generate_fallback_response(
                user_input, pet, command, self.current_degradation_level
            )
            
            # Create interaction
            interaction = Interaction(
                timestamp=datetime.now(),
                user_input=user_input,
                interpreted_command=command or "",
                pet_response=response,
                attribute_changes={},
                success=True,
                result=InteractionResult.SUCCESS
            )
            
            # Add degradation notice for non-emergency modes
            if self.current_degradation_level != DegradationLevel.FULL_SERVICE:
                if self.current_degradation_level != DegradationLevel.EMERGENCY_MODE:
                    interaction.pet_response += " (AI services are currently limited)"
            
            return interaction
            
        except Exception as e:
            logger.error(f"Degraded response generation failed: {e}")
            
            # Ultimate fallback
            return Interaction(
                timestamp=datetime.now(),
                user_input=user_input,
                interpreted_command="",
                pet_response="*DigiPal is resting*",
                attribute_changes={},
                success=False,
                result=InteractionResult.FAILURE
            )


# Global AI service manager instance
ai_service_manager = AIServiceManager()


def with_ai_fallback(service_name: str, fallback_response: Optional[str] = None):
    """
    Decorator for AI service calls with automatic fallback.
    
    Args:
        service_name: Name of the AI service
        fallback_response: Default fallback response
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def fallback_func(*args, **kwargs):
                if fallback_response:
                    return fallback_response
                # Try to extract pet from arguments for context-aware fallback
                pet = None
                for arg in args:
                    if isinstance(arg, DigiPal):
                        pet = arg
                        break
                
                if pet:
                    return ai_service_manager.fallback_generator.generate_fallback_response(
                        "", pet, None, ai_service_manager.current_degradation_level
                    )
                
                return "Service temporarily unavailable"
            
            return ai_service_manager.call_ai_service(
                service_name, func, fallback_func, *args, **kwargs
            )
        
        return wrapper
    return decorator