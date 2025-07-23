"""
AI Communication layer for DigiPal application.

This module handles speech processing, natural language generation,
command interpretation, and conversation memory management.
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
import torch

from ..core.models import DigiPal, Interaction, Command
from ..core.enums import LifeStage, CommandType, InteractionResult
from .language_model import LanguageModel


logger = logging.getLogger(__name__)


class AICommunication:
    """
    Main AI communication class that orchestrates speech processing,
    language model interactions, and conversation management.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", quantization: bool = True, kyutai_config: Optional[Dict] = None):
        """
        Initialize AI communication system.
        
        Args:
            model_name: HuggingFace model identifier for Qwen3-0.6B
            quantization: Whether to use quantization for memory optimization
            kyutai_config: Configuration for Kyutai speech processing (placeholder for now)
        """
        self.model_name = model_name
        self.quantization = quantization
        self.kyutai_config = kyutai_config or {}
        
        # Initialize components
        self.command_interpreter = CommandInterpreter()
        self.response_generator = ResponseGenerator()
        self.memory_manager = ConversationMemoryManager()
        
        # Initialize language model
        self.language_model = LanguageModel(model_name, quantization)
        self._model_loaded = False
        
        logger.info(f"AICommunication initialized with model: {model_name}")
        logger.info(f"Quantization enabled: {quantization}")
    
    def process_speech(self, audio_data: bytes) -> str:
        """
        Process speech audio data and convert to text.
        
        Args:
            audio_data: Raw audio bytes from user input
            
        Returns:
            Transcribed text from speech
            
        Note: This is a placeholder implementation. In the full version,
        this would integrate with Kyutai speech-to-text processing.
        """
        # Placeholder implementation
        logger.info("Processing speech audio (placeholder)")
        
        # In real implementation, this would:
        # 1. Validate audio quality
        # 2. Apply noise reduction
        # 3. Use Kyutai model for speech-to-text
        # 4. Handle recognition errors
        
        # For now, return a placeholder response
        return "placeholder_speech_text"
    
    def generate_response(self, input_text: str, pet: DigiPal) -> str:
        """
        Generate contextual response using Qwen3-0.6B language model.
        
        Args:
            input_text: User input text
            pet: Current DigiPal instance for context
            
        Returns:
            Generated response text
        """
        logger.info(f"Generating response for input: {input_text}")
        
        # Ensure model is loaded
        if not self._model_loaded:
            self.load_model()
        
        # Use language model if available, otherwise fallback to template responses
        if self.language_model.is_loaded():
            return self.language_model.generate_response(input_text, pet)
        else:
            logger.warning("Language model not available, using fallback response generator")
            return self.response_generator.generate_response(input_text, pet)
    
    def interpret_command(self, text: str, pet: DigiPal) -> Command:
        """
        Interpret user text input into actionable commands.
        
        Args:
            text: User input text
            pet: Current DigiPal instance for context
            
        Returns:
            Parsed Command object
        """
        return self.command_interpreter.parse_command(text, pet.life_stage)
    
    def process_interaction(self, input_text: str, pet: DigiPal) -> Interaction:
        """
        Process a complete user interaction with the DigiPal.
        
        Args:
            input_text: User input text
            pet: Current DigiPal instance
            
        Returns:
            Complete Interaction object with results
        """
        # Parse the command
        command = self.interpret_command(input_text, pet)
        
        # Generate response
        response = self.generate_response(input_text, pet)
        
        # Create interaction record
        interaction = Interaction(
            timestamp=datetime.now(),
            user_input=input_text,
            interpreted_command=command.action,
            pet_response=response,
            success=command.stage_appropriate,
            result=InteractionResult.SUCCESS if command.stage_appropriate else InteractionResult.STAGE_INAPPROPRIATE
        )
        
        # Update conversation memory
        self.update_conversation_memory(interaction, pet)
        
        return interaction
    
    def update_conversation_memory(self, interaction: Interaction, pet: DigiPal) -> None:
        """
        Update conversation memory with new interaction.
        
        Args:
            interaction: New interaction to add to memory
            pet: DigiPal instance to update
        """
        self.memory_manager.add_interaction(interaction, pet)
    
    def load_model(self) -> bool:
        """
        Load the Qwen3-0.6B language model.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        logger.info("Loading Qwen3-0.6B language model...")
        
        try:
            success = self.language_model.load_model()
            self._model_loaded = success
            
            if success:
                logger.info("Language model loaded successfully")
            else:
                logger.warning("Failed to load language model, will use fallback responses")
            
            return success
            
        except Exception as e:
            logger.error(f"Error loading language model: {e}")
            self._model_loaded = False
            return False
    
    def is_model_loaded(self) -> bool:
        """
        Check if the language model is loaded and ready.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self._model_loaded and self.language_model.is_loaded()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded language model.
        
        Returns:
            Dictionary with model information
        """
        base_info = {
            'model_name': self.model_name,
            'quantization': self.quantization,
            'loaded': self.is_model_loaded()
        }
        
        if self.language_model:
            base_info.update(self.language_model.get_model_info())
        
        return base_info
    
    def unload_model(self) -> None:
        """
        Unload the language model to free memory.
        """
        if self.language_model:
            # Clear model references to free memory
            self.language_model.model = None
            self.language_model.tokenizer = None
            self._model_loaded = False
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Language model unloaded")


class CommandInterpreter:
    """
    Interprets user text input into structured commands based on DigiPal's life stage.
    """
    
    def __init__(self):
        """Initialize command interpreter with command patterns."""
        self.command_patterns = self._initialize_command_patterns()
        self.stage_commands = self._initialize_stage_commands()
    
    def _initialize_command_patterns(self) -> Dict[CommandType, List[str]]:
        """Initialize regex patterns for command recognition."""
        return {
            CommandType.EAT: [
                r'\b(eat|feed|food|hungry|meal)\b',
                r'\b(give.*food|want.*food)\b'
            ],
            CommandType.SLEEP: [
                r'\b(sleep|rest|tired|nap|bed)\b',
                r'\b(go.*sleep|time.*sleep)\b'
            ],
            CommandType.GOOD: [
                r'\b(good|great|excellent|well done|nice)\b',
                r'\b(praise|proud|amazing)\b'
            ],
            CommandType.BAD: [
                r'\b(bad|no|stop|wrong|naughty)\b',
                r'\b(scold|discipline|behave)\b'
            ],
            CommandType.TRAIN: [
                r'\b(train|exercise|workout|practice|training)\b',
                r'\b(let\'s train|training time|work on|time for.*training)\b'
            ],
            CommandType.PLAY: [
                r'\b(play|fun|game|toy)\b',
                r'\b(let\'s play|play time)\b'
            ],
            CommandType.STATUS: [
                r'\b(status|how.*you|feeling|health|show)\b',
                r'\b(check.*stats|show.*attributes|show.*status)\b'
            ]
        }
    
    def _initialize_stage_commands(self) -> Dict[LifeStage, List[CommandType]]:
        """Initialize available commands for each life stage."""
        return {
            LifeStage.EGG: [],
            LifeStage.BABY: [CommandType.EAT, CommandType.SLEEP, CommandType.GOOD, CommandType.BAD],
            LifeStage.CHILD: [CommandType.EAT, CommandType.SLEEP, CommandType.GOOD, CommandType.BAD, 
                             CommandType.PLAY, CommandType.TRAIN],
            LifeStage.TEEN: [CommandType.EAT, CommandType.SLEEP, CommandType.GOOD, CommandType.BAD,
                            CommandType.PLAY, CommandType.TRAIN, CommandType.STATUS],
            LifeStage.YOUNG_ADULT: [CommandType.EAT, CommandType.SLEEP, CommandType.GOOD, CommandType.BAD,
                                   CommandType.PLAY, CommandType.TRAIN, CommandType.STATUS],
            LifeStage.ADULT: [CommandType.EAT, CommandType.SLEEP, CommandType.GOOD, CommandType.BAD,
                             CommandType.PLAY, CommandType.TRAIN, CommandType.STATUS],
            LifeStage.ELDERLY: [CommandType.EAT, CommandType.SLEEP, CommandType.GOOD, CommandType.BAD,
                               CommandType.PLAY, CommandType.TRAIN, CommandType.STATUS]
        }
    
    def parse_command(self, text: str, life_stage: LifeStage) -> Command:
        """
        Parse user text into a structured command.
        
        Args:
            text: User input text
            life_stage: Current DigiPal life stage
            
        Returns:
            Parsed Command object
        """
        text_lower = text.lower().strip()
        
        # Check each command type for pattern matches
        for command_type, patterns in self.command_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    # Check if command is appropriate for current life stage
                    stage_appropriate = command_type in self.stage_commands.get(life_stage, [])
                    
                    return Command(
                        action=command_type.value,
                        command_type=command_type,
                        stage_appropriate=stage_appropriate,
                        energy_required=self._get_energy_requirement(command_type),
                        parameters=self._extract_parameters(text_lower, command_type)
                    )
        
        # If no pattern matches, return unknown command
        return Command(
            action="unknown",
            command_type=CommandType.UNKNOWN,
            stage_appropriate=False,
            energy_required=0,
            parameters={"original_text": text}
        )
    
    def _get_energy_requirement(self, command_type: CommandType) -> int:
        """Get energy requirement for command type."""
        energy_requirements = {
            CommandType.EAT: 0,
            CommandType.SLEEP: 0,
            CommandType.GOOD: 0,
            CommandType.BAD: 0,
            CommandType.TRAIN: 20,
            CommandType.PLAY: 10,
            CommandType.STATUS: 0,
            CommandType.UNKNOWN: 0
        }
        return energy_requirements.get(command_type, 0)
    
    def _extract_parameters(self, text: str, command_type: CommandType) -> Dict[str, Any]:
        """Extract parameters from command text."""
        parameters = {}
        
        # Add command-specific parameter extraction logic
        if command_type == CommandType.TRAIN:
            # Look for specific training types
            if 'strength' in text or 'attack' in text:
                parameters['training_type'] = 'strength'
            elif 'defense' in text or 'guard' in text:
                parameters['training_type'] = 'defense'
            elif 'speed' in text or 'agility' in text:
                parameters['training_type'] = 'speed'
            elif 'brain' in text or 'intelligence' in text:
                parameters['training_type'] = 'brains'
            else:
                parameters['training_type'] = 'general'
        
        elif command_type == CommandType.EAT:
            # Look for food types (placeholder for future expansion)
            parameters['food_type'] = 'standard'
        
        return parameters


class ResponseGenerator:
    """
    Generates contextual responses based on DigiPal state and user input.
    """
    
    def __init__(self):
        """Initialize response generator with templates."""
        self.response_templates = self._initialize_response_templates()
    
    def _initialize_response_templates(self) -> Dict[LifeStage, Dict[str, List[str]]]:
        """Initialize response templates for each life stage and situation."""
        return {
            LifeStage.EGG: {
                'default': ["*The egg remains silent*", "*The egg seems to be listening*"],
                'speech_detected': ["*The egg trembles slightly*", "*Something stirs within the egg*"]
            },
            LifeStage.BABY: {
                'eat': ["*happy baby sounds*", "Goo goo!", "*contentedly munches*"],
                'sleep': ["*yawns sleepily*", "Zzz...", "*curls up peacefully*"],
                'good': ["*giggles happily*", "Goo!", "*bounces with joy*"],
                'bad': ["*whimpers*", "*looks sad*", "*hides behind hands*"],
                'unknown': ["*tilts head curiously*", "*makes confused baby sounds*", "Goo?"],
                'default': ["*baby babbling*", "Goo goo ga ga!", "*looks at you with big eyes*"]
            },
            LifeStage.CHILD: {
                'eat': ["Yummy! Thank you!", "*munches happily*", "This tastes good!"],
                'sleep': ["I'm getting sleepy...", "*yawns*", "Nap time!"],
                'good': ["Really? Thank you!", "*beams with pride*", "I did good!"],
                'bad': ["Sorry... I'll be better", "*looks down sadly*", "I didn't mean to..."],
                'train': ["Let's get stronger!", "*pumps tiny fists*", "Training is fun!"],
                'play': ["Yay! Let's play!", "*jumps excitedly*", "This is so much fun!"],
                'unknown': ["I don't understand...", "*looks confused*", "What does that mean?"],
                'default': ["Hi there!", "*waves enthusiastically*", "What should we do?"]
            },
            LifeStage.TEEN: {
                'eat': ["Thanks, I was getting hungry", "*eats with good appetite*", "This hits the spot!"],
                'sleep': ["Yeah, I could use some rest", "*stretches*", "Sleep sounds good right now"],
                'good': ["Thanks! I've been working hard", "*smiles proudly*", "That means a lot!"],
                'bad': ["Okay, okay, I get it", "*sighs*", "I'll try to do better"],
                'train': ["Alright, let's do this!", "*gets into stance*", "I'm ready to train!"],
                'play': ["Sure, let's have some fun!", "*grins*", "I could use a break anyway"],
                'status': ["I'm feeling pretty good overall", "*flexes*", "Want to know something specific?"],
                'unknown': ["Hmm, not sure what you mean", "*scratches head*", "Could you be more specific?"],
                'default': ["Hey! What's up?", "*looks attentive*", "Ready for whatever!"]
            },
            LifeStage.YOUNG_ADULT: {
                'eat': ["Perfect timing, thanks!", "*eats with appreciation*", "Just what I needed"],
                'sleep': ["Good idea, I should rest up", "*settles down comfortably*", "Rest is important for growth"],
                'good': ["I appreciate the encouragement!", "*stands tall with confidence*", "Your support means everything"],
                'bad': ["You're right, I need to focus more", "*nods seriously*", "I'll be more careful"],
                'train': ["Let's push our limits!", "*determined expression*", "Every session makes us stronger!"],
                'play': ["A good balance of work and play!", "*laughs*", "Let's enjoy ourselves!"],
                'status': ["I'm in my prime right now!", "*shows off confidently*", "Want the full rundown?"],
                'unknown': ["I'm not quite sure what you're asking", "*thinks carefully*", "Can you elaborate?"],
                'default': ["Good to see you!", "*confident smile*", "What's on the agenda today?"]
            },
            LifeStage.ADULT: {
                'eat': ["Thank you for the meal", "*eats thoughtfully*", "Proper nutrition is key"],
                'sleep': ["Rest is wisdom", "*settles down peacefully*", "A clear mind needs good rest"],
                'good': ["Your words honor me", "*bows respectfully*", "I strive to be worthy of your praise"],
                'bad': ["I understand your concern", "*reflects seriously*", "I will consider your words carefully"],
                'train': ["Discipline shapes the spirit", "*begins training with focus*", "Let us grow stronger together"],
                'play': ["Joy has its place in life", "*smiles warmly*", "Even adults need moments of lightness"],
                'status': ["I am at my peak capabilities", "*stands with dignity*", "How may I serve?"],
                'unknown': ["Your meaning escapes me", "*listens intently*", "Please help me understand"],
                'default': ["Greetings, my friend", "*respectful nod*", "How may we spend our time together?"]
            },
            LifeStage.ELDERLY: {
                'eat': ["Ah, sustenance for these old bones", "*eats slowly and deliberately*", "Simple pleasures matter most"],
                'sleep': ["Rest comes easier now", "*settles down with a sigh*", "Dreams of younger days..."],
                'good': ["Your kindness warms an old heart", "*smiles gently*", "I have lived well with you"],
                'bad': ["At my age, mistakes are lessons", "*chuckles softly*", "I am still learning, it seems"],
                'train': ["These old muscles remember", "*moves carefully but determined*", "Wisdom guides where strength once led"],
                'play': ["Play keeps the spirit young", "*laughs with delight*", "Age is just a number!"],
                'status': ["I have seen much in my time", "*gazes thoughtfully*", "Each day is a gift now"],
                'unknown': ["My hearing isn't what it was", "*cups ear*", "Could you repeat that, dear?"],
                'default': ["Hello, old friend", "*warm, weathered smile*", "Another day together..."]
            }
        }
    
    def generate_response(self, input_text: str, pet: DigiPal) -> str:
        """
        Generate contextual response based on input and pet state.
        
        Args:
            input_text: User input text
            pet: Current DigiPal instance
            
        Returns:
            Generated response string
        """
        # Parse command to determine response type
        command_interpreter = CommandInterpreter()
        command = command_interpreter.parse_command(input_text, pet.life_stage)
        
        # Get appropriate response template
        stage_templates = self.response_templates.get(pet.life_stage, {})
        
        # Select response based on command type
        if command.stage_appropriate and command.command_type != CommandType.UNKNOWN:
            response_key = command.command_type.value
        elif command.command_type == CommandType.UNKNOWN or not command.stage_appropriate:
            # For stage-inappropriate commands or unknown commands, use unknown response
            response_key = 'unknown'
        else:
            response_key = 'default'
        
        # Get responses for the key, fallback to default
        responses = stage_templates.get(response_key, stage_templates.get('default', ["*confused sounds*"]))
        
        # Select response based on pet's personality or randomly
        # For now, use simple selection based on happiness
        if pet.happiness > 70:
            response_index = 0  # Use first (most positive) response
        elif pet.happiness > 30:
            response_index = min(1, len(responses) - 1)  # Use middle response
        else:
            response_index = len(responses) - 1  # Use last (least positive) response
        
        return responses[response_index]


class ConversationMemoryManager:
    """
    Manages conversation history and memory for DigiPal interactions.
    """
    
    def __init__(self, max_memory_size: int = 100):
        """
        Initialize memory manager.
        
        Args:
            max_memory_size: Maximum number of interactions to keep in memory
        """
        self.max_memory_size = max_memory_size
    
    def add_interaction(self, interaction: Interaction, pet: DigiPal) -> None:
        """
        Add new interaction to pet's conversation history.
        
        Args:
            interaction: New interaction to add
            pet: DigiPal instance to update
        """
        # Add interaction to pet's history
        pet.conversation_history.append(interaction)
        
        # Update last interaction time
        pet.last_interaction = interaction.timestamp
        
        # Learn new commands if successful
        if interaction.success and interaction.interpreted_command:
            pet.learned_commands.add(interaction.interpreted_command)
        
        # Manage memory size
        self._manage_memory_size(pet)
        
        # Update personality traits based on interaction
        self._update_personality_traits(interaction, pet)
    
    def _manage_memory_size(self, pet: DigiPal) -> None:
        """
        Manage conversation history size to prevent memory bloat.
        
        Args:
            pet: DigiPal instance to manage
        """
        if len(pet.conversation_history) > self.max_memory_size:
            # Keep most recent interactions
            pet.conversation_history = pet.conversation_history[-self.max_memory_size:]
    
    def _update_personality_traits(self, interaction: Interaction, pet: DigiPal) -> None:
        """
        Update pet's personality traits based on interaction patterns.
        
        Args:
            interaction: Recent interaction
            pet: DigiPal instance to update
        """
        # Initialize personality traits if not present
        if not pet.personality_traits:
            pet.personality_traits = {
                'friendliness': 0.5,
                'playfulness': 0.5,
                'obedience': 0.5,
                'curiosity': 0.5
            }
        
        # Update traits based on interaction type
        if interaction.interpreted_command == 'good':
            pet.personality_traits['obedience'] = min(1.0, pet.personality_traits['obedience'] + 0.1)
        elif interaction.interpreted_command == 'bad':
            pet.personality_traits['obedience'] = max(0.0, pet.personality_traits['obedience'] - 0.05)
        elif interaction.interpreted_command == 'play':
            pet.personality_traits['playfulness'] = min(1.0, pet.personality_traits['playfulness'] + 0.1)
        elif interaction.success:
            pet.personality_traits['friendliness'] = min(1.0, pet.personality_traits['friendliness'] + 0.05)
        
        # Increase curiosity for unknown commands (shows engagement)
        if interaction.interpreted_command == 'unknown':
            pet.personality_traits['curiosity'] = min(1.0, pet.personality_traits['curiosity'] + 0.02)
    
    def get_recent_interactions(self, pet: DigiPal, count: int = 10) -> List[Interaction]:
        """
        Get recent interactions from pet's memory.
        
        Args:
            pet: DigiPal instance
            count: Number of recent interactions to retrieve
            
        Returns:
            List of recent interactions
        """
        return pet.conversation_history[-count:] if pet.conversation_history else []
    
    def get_interaction_summary(self, pet: DigiPal) -> Dict[str, Any]:
        """
        Get summary statistics of pet's interaction history.
        
        Args:
            pet: DigiPal instance
            
        Returns:
            Dictionary with interaction statistics
        """
        if not pet.conversation_history:
            return {
                'total_interactions': 0,
                'successful_interactions': 0,
                'success_rate': 0.0,
                'most_common_commands': [],
                'last_interaction': None
            }
        
        total = len(pet.conversation_history)
        successful = sum(1 for i in pet.conversation_history if i.success)
        
        # Count command frequency
        command_counts = {}
        for interaction in pet.conversation_history:
            cmd = interaction.interpreted_command
            command_counts[cmd] = command_counts.get(cmd, 0) + 1
        
        # Sort commands by frequency
        most_common = sorted(command_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_interactions': total,
            'successful_interactions': successful,
            'success_rate': successful / total if total > 0 else 0.0,
            'most_common_commands': most_common,
            'last_interaction': pet.conversation_history[-1].timestamp if pet.conversation_history else None
        }