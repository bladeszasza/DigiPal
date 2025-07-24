"""
DigiPal Core Engine - Central orchestrator for all DigiPal functionality.

This module provides the main DigiPalCore class that coordinates pet creation,
loading, state management, interaction processing, and real-time updates.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import threading
import time

from .models import DigiPal, Interaction, Command
from .enums import EggType, LifeStage, InteractionResult, AttributeType
from .attribute_engine import AttributeEngine
from .evolution_controller import EvolutionController, EvolutionResult
from ..storage.storage_manager import StorageManager
from ..ai.communication import AICommunication

logger = logging.getLogger(__name__)


class PetState:
    """Represents the current state of a DigiPal for external systems."""
    
    def __init__(self, pet: DigiPal):
        """Initialize PetState from DigiPal instance."""
        self.id = pet.id
        self.user_id = pet.user_id
        self.name = pet.name
        self.life_stage = pet.life_stage
        self.generation = pet.generation
        
        # Attributes
        self.hp = pet.hp
        self.mp = pet.mp
        self.offense = pet.offense
        self.defense = pet.defense
        self.speed = pet.speed
        self.brains = pet.brains
        self.discipline = pet.discipline
        self.happiness = pet.happiness
        self.weight = pet.weight
        self.care_mistakes = pet.care_mistakes
        self.energy = pet.energy
        
        # Status
        self.age_hours = pet.get_age_hours()
        self.last_interaction = pet.last_interaction
        self.current_image_path = pet.current_image_path
        
        # Derived status
        self.needs_attention = self._calculate_needs_attention(pet)
        self.evolution_ready = False  # Will be set by core engine
        self.status_summary = self._generate_status_summary(pet)
    
    def _calculate_needs_attention(self, pet: DigiPal) -> bool:
        """Calculate if pet needs immediate attention."""
        return (
            pet.energy < 40 or 
            pet.happiness < 30 or 
            pet.weight < 10 or 
            pet.weight > 80 or
            (datetime.now() - pet.last_interaction).total_seconds() > 3600  # 1 hour
        )
    
    def _generate_status_summary(self, pet: DigiPal) -> str:
        """Generate a brief status summary."""
        if pet.energy < 20:
            return "Very tired"
        elif pet.happiness < 30:
            return "Unhappy"
        elif pet.energy < 40:
            return "Getting tired"
        elif pet.happiness > 80:
            return "Very happy"
        else:
            return "Doing well"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert PetState to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'name': self.name,
            'life_stage': self.life_stage.value,
            'generation': self.generation,
            'attributes': {
                'hp': self.hp,
                'mp': self.mp,
                'offense': self.offense,
                'defense': self.defense,
                'speed': self.speed,
                'brains': self.brains,
                'discipline': self.discipline,
                'happiness': self.happiness,
                'weight': self.weight,
                'care_mistakes': self.care_mistakes,
                'energy': self.energy
            },
            'status': {
                'age_hours': self.age_hours,
                'last_interaction': self.last_interaction.isoformat(),
                'current_image_path': self.current_image_path,
                'needs_attention': self.needs_attention,
                'evolution_ready': self.evolution_ready,
                'status_summary': self.status_summary
            }
        }


class InteractionProcessor:
    """Processes user interactions with DigiPal through the AI communication layer."""
    
    def __init__(self, ai_communication: AICommunication, attribute_engine: AttributeEngine):
        """Initialize interaction processor."""
        self.ai_communication = ai_communication
        self.attribute_engine = attribute_engine
    
    def process_text_interaction(self, text: str, pet: DigiPal) -> Interaction:
        """
        Process a text-based interaction with the DigiPal.
        
        Args:
            text: User input text
            pet: DigiPal instance
            
        Returns:
            Interaction result
        """
        logger.info(f"Processing text interaction: '{text}' for pet {pet.id}")
        
        # Use AI communication to process the interaction
        interaction = self.ai_communication.process_interaction(text, pet)
        
        # Apply any care actions based on the interpreted command
        if interaction.interpreted_command and interaction.success:
            self._apply_command_effects(interaction.interpreted_command, pet, interaction)
        
        return interaction
    
    def process_speech_interaction(self, audio_data: bytes, pet: DigiPal, sample_rate: Optional[int] = None) -> Interaction:
        """
        Process a speech-based interaction with the DigiPal.
        
        Args:
            audio_data: Raw audio bytes
            pet: DigiPal instance
            sample_rate: Audio sample rate (optional)
            
        Returns:
            Interaction result
        """
        logger.info(f"Processing speech interaction for pet {pet.id}")
        
        # Convert speech to text
        transcribed_text = self.ai_communication.process_speech(audio_data, sample_rate)
        
        if not transcribed_text:
            # Speech processing failed
            interaction = Interaction(
                user_input="[Speech not recognized]",
                interpreted_command="",
                pet_response="I couldn't understand what you said. Could you try again?",
                success=False,
                result=InteractionResult.FAILURE
            )
            pet.conversation_history.append(interaction)
            return interaction
        
        # Process the transcribed text as a normal text interaction
        interaction = self.process_text_interaction(transcribed_text, pet)
        
        # Special handling for first speech interaction (egg hatching)
        if pet.life_stage == LifeStage.EGG and transcribed_text:
            interaction.pet_response = "The egg begins to crack and glow! Your DigiPal is hatching!"
            interaction.attribute_changes["hatching"] = 1
        
        return interaction
    
    def _apply_command_effects(self, command: str, pet: DigiPal, interaction: Interaction):
        """Apply the effects of a successful command to the pet."""
        attribute_changes = {}
        
        # Map commands to care actions
        command_to_action = {
            'eat': 'meat',  # Default food
            'sleep': 'rest',
            'good': 'praise',
            'bad': 'scold',
            'train': 'strength_training',  # Default training
            'play': 'play'
        }
        
        action_name = command_to_action.get(command)
        if action_name:
            success, care_interaction = self.attribute_engine.apply_care_action(pet, action_name)
            if success:
                attribute_changes.update(care_interaction.attribute_changes)
        
        # Update interaction with attribute changes
        interaction.attribute_changes.update(attribute_changes)


class DigiPalCore:
    """
    Central orchestrator for DigiPal functionality.
    
    Manages pet lifecycle, coordinates all components, and provides
    the main interface for DigiPal operations.
    """
    
    def __init__(self, storage_manager: StorageManager, ai_communication: AICommunication):
        """
        Initialize DigiPal Core Engine.
        
        Args:
            storage_manager: Storage manager for data persistence
            ai_communication: AI communication layer
        """
        self.storage_manager = storage_manager
        self.ai_communication = ai_communication
        
        # Initialize core components
        self.attribute_engine = AttributeEngine()
        self.evolution_controller = EvolutionController()
        self.interaction_processor = InteractionProcessor(ai_communication, self.attribute_engine)
        
        # Active pets cache (user_id -> DigiPal)
        self.active_pets: Dict[str, DigiPal] = {}
        
        # Background update system
        self._update_thread = None
        self._stop_updates = False
        self._update_interval = 60  # Update every minute
        
        logger.info("DigiPalCore initialized successfully")
    
    def create_new_pet(self, egg_type: EggType, user_id: str, name: str = "DigiPal") -> DigiPal:
        """
        Create a new DigiPal with specified egg type and user.
        
        Args:
            egg_type: Type of egg to create
            user_id: User ID who owns the pet
            name: Name for the new DigiPal
            
        Returns:
            Newly created DigiPal instance
        """
        logger.info(f"Creating new DigiPal for user {user_id} with egg type {egg_type.value}")
        
        # Create new DigiPal instance
        pet = DigiPal(
            user_id=user_id,
            name=name,
            egg_type=egg_type,
            life_stage=LifeStage.EGG
        )
        
        # Initialize egg-specific attributes (already done in DigiPal.__post_init__)
        
        # Save to storage
        if self.storage_manager.save_pet(pet):
            # Add to active pets cache
            self.active_pets[user_id] = pet
            logger.info(f"Successfully created pet {pet.id} for user {user_id}")
            return pet
        else:
            logger.error(f"Failed to save new pet for user {user_id}")
            raise RuntimeError("Failed to save new DigiPal to storage")
    
    def load_existing_pet(self, user_id: str) -> Optional[DigiPal]:
        """
        Load an existing DigiPal for a user.
        
        Args:
            user_id: User ID to load pet for
            
        Returns:
            Loaded DigiPal instance or None if not found
        """
        logger.info(f"Loading existing DigiPal for user {user_id}")
        
        # Check cache first
        if user_id in self.active_pets:
            logger.info(f"Found pet in cache for user {user_id}")
            return self.active_pets[user_id]
        
        # Load from storage
        pet = self.storage_manager.load_pet(user_id)
        if pet:
            # Add to cache
            self.active_pets[user_id] = pet
            logger.info(f"Successfully loaded pet {pet.id} for user {user_id}")
            
            # Apply time-based updates since last interaction
            self._apply_time_based_updates(pet)
            
            return pet
        else:
            logger.info(f"No existing pet found for user {user_id}")
            return None
    
    def get_pet_state(self, user_id: str) -> Optional[PetState]:
        """
        Get current state of user's DigiPal.
        
        Args:
            user_id: User ID
            
        Returns:
            PetState instance or None if no pet found
        """
        pet = self.active_pets.get(user_id)
        if not pet:
            pet = self.load_existing_pet(user_id)
        
        if pet:
            state = PetState(pet)
            # Check evolution eligibility
            eligible, _, _ = self.evolution_controller.check_evolution_eligibility(pet)
            state.evolution_ready = eligible
            return state
        
        return None
    
    def process_interaction(self, user_id: str, input_text: str) -> Tuple[bool, Interaction]:
        """
        Process a text interaction with user's DigiPal.
        
        Args:
            user_id: User ID
            input_text: User input text
            
        Returns:
            Tuple of (success, interaction_result)
        """
        pet = self.active_pets.get(user_id)
        if not pet:
            pet = self.load_existing_pet(user_id)
        
        if not pet:
            logger.error(f"No pet found for user {user_id}")
            return False, Interaction(
                user_input=input_text,
                pet_response="No DigiPal found. Please create a new one first.",
                success=False,
                result=InteractionResult.FAILURE
            )
        
        # Process the interaction
        interaction = self.interaction_processor.process_text_interaction(input_text, pet)
        
        # Handle special cases
        if pet.life_stage == LifeStage.EGG and input_text:
            # First interaction with egg triggers hatching
            self._trigger_hatching(pet, interaction)
        
        # Save updated pet state
        self.storage_manager.save_pet(pet)
        
        return interaction.success, interaction
    
    def process_speech_interaction(self, user_id: str, audio_data: bytes, sample_rate: Optional[int] = None) -> Tuple[bool, Interaction]:
        """
        Process a speech interaction with user's DigiPal.
        
        Args:
            user_id: User ID
            audio_data: Raw audio bytes
            sample_rate: Audio sample rate (optional)
            
        Returns:
            Tuple of (success, interaction_result)
        """
        pet = self.active_pets.get(user_id)
        if not pet:
            pet = self.load_existing_pet(user_id)
        
        if not pet:
            logger.error(f"No pet found for user {user_id}")
            return False, Interaction(
                user_input="[Speech input]",
                pet_response="No DigiPal found. Please create a new one first.",
                success=False,
                result=InteractionResult.FAILURE
            )
        
        # Process the speech interaction
        interaction = self.interaction_processor.process_speech_interaction(audio_data, pet, sample_rate)
        
        # Handle special cases
        if pet.life_stage == LifeStage.EGG and interaction.success:
            # First speech interaction with egg triggers hatching
            self._trigger_hatching(pet, interaction)
        
        # Save updated pet state
        self.storage_manager.save_pet(pet)
        
        return interaction.success, interaction
    
    def _trigger_hatching(self, pet: DigiPal, interaction: Interaction):
        """Trigger egg hatching to baby stage."""
        if pet.life_stage == LifeStage.EGG:
            logger.info(f"Triggering hatching for pet {pet.id}")
            
            # Evolve to baby stage
            evolution_result = self.evolution_controller.trigger_evolution(pet, force=True)
            
            if evolution_result.success:
                interaction.pet_response = "Your DigiPal has hatched! Welcome to the world, little one!"
                interaction.attribute_changes.update(evolution_result.attribute_changes)
                logger.info(f"Pet {pet.id} successfully hatched to baby stage")
            else:
                logger.error(f"Failed to hatch pet {pet.id}")
    
    def update_pet_state(self, user_id: str, force_save: bool = False) -> bool:
        """
        Update pet state with time-based changes.
        
        Args:
            user_id: User ID
            force_save: Force save to storage even if no changes
            
        Returns:
            True if update was successful
        """
        pet = self.active_pets.get(user_id)
        if not pet:
            return False
        
        try:
            # Apply time-based updates
            changes_applied = self._apply_time_based_updates(pet)
            
            # Check for evolution
            evolution_occurred = self._check_and_apply_evolution(pet)
            
            # Save if changes were made or forced
            if changes_applied or evolution_occurred or force_save:
                return self.storage_manager.save_pet(pet)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating pet state for user {user_id}: {e}")
            return False
    
    def _apply_time_based_updates(self, pet: DigiPal) -> bool:
        """
        Apply time-based attribute decay and updates.
        
        Args:
            pet: DigiPal to update
            
        Returns:
            True if any changes were applied
        """
        now = datetime.now()
        time_diff = now - pet.last_interaction
        hours_passed = time_diff.total_seconds() / 3600
        
        if hours_passed < 0.01:  # Less than ~36 seconds, skip update
            return False
        
        logger.debug(f"Applying time-based updates for pet {pet.id}: {hours_passed:.2f} hours passed")
        
        # Apply attribute decay
        decay_changes = self.attribute_engine.apply_time_decay(pet, hours_passed)
        
        # Update evolution timer
        self.evolution_controller.update_evolution_timer(pet, hours_passed)
        
        # Update last interaction time to prevent repeated updates
        pet.last_interaction = now
        
        return len(decay_changes) > 0
    
    def _check_and_apply_evolution(self, pet: DigiPal) -> bool:
        """
        Check if pet should evolve and apply evolution if ready.
        
        Args:
            pet: DigiPal to check
            
        Returns:
            True if evolution occurred
        """
        # Check time-based evolution
        if self.evolution_controller.check_time_based_evolution(pet):
            logger.info(f"Time-based evolution triggered for pet {pet.id}")
            evolution_result = self.evolution_controller.trigger_evolution(pet)
            
            if evolution_result.success:
                logger.info(f"Pet {pet.id} evolved from {evolution_result.old_stage.value} to {evolution_result.new_stage.value}")
                return True
            else:
                logger.warning(f"Evolution failed for pet {pet.id}: {evolution_result.message}")
        
        # Check if pet should die (elderly stage time limit)
        if self.evolution_controller.is_death_time(pet):
            logger.info(f"Pet {pet.id} has reached end of life")
            self._handle_pet_death(pet)
            return True
        
        return False
    
    def _handle_pet_death(self, pet: DigiPal):
        """Handle pet death and prepare for next generation."""
        logger.info(f"Handling death of pet {pet.id}")
        
        # Calculate care quality for inheritance
        care_assessment = self.attribute_engine.get_care_quality_assessment(pet)
        care_quality = care_assessment.get('care_quality', 'fair')
        
        # Create DNA inheritance data
        dna = self.evolution_controller.create_inheritance_dna(pet, care_quality)
        
        # Mark pet as inactive
        self.storage_manager.delete_pet(pet.id)
        
        # Remove from active pets cache
        if pet.user_id in self.active_pets:
            del self.active_pets[pet.user_id]
        
        # Store DNA for next generation (could be stored in user session or database)
        # For now, we'll log it - in a full implementation, this would be stored
        logger.info(f"DNA inheritance prepared for user {pet.user_id}: generation {dna.generation}")
    
    def trigger_evolution(self, user_id: str, force: bool = False) -> Tuple[bool, EvolutionResult]:
        """
        Manually trigger evolution for user's DigiPal.
        
        Args:
            user_id: User ID
            force: Force evolution regardless of requirements
            
        Returns:
            Tuple of (success, evolution_result)
        """
        pet = self.active_pets.get(user_id)
        if not pet:
            pet = self.load_existing_pet(user_id)
        
        if not pet:
            return False, EvolutionResult(
                success=False,
                old_stage=LifeStage.EGG,
                new_stage=LifeStage.EGG,
                message="No DigiPal found"
            )
        
        evolution_result = self.evolution_controller.trigger_evolution(pet, force)
        
        if evolution_result.success:
            # Save updated pet
            self.storage_manager.save_pet(pet)
            logger.info(f"Manual evolution triggered for pet {pet.id}")
        
        return evolution_result.success, evolution_result
    
    def start_background_updates(self):
        """Start background thread for automatic pet state updates."""
        if self._update_thread and self._update_thread.is_alive():
            logger.warning("Background updates already running")
            return
        
        self._stop_updates = False
        self._update_thread = threading.Thread(target=self._background_update_loop, daemon=True)
        self._update_thread.start()
        logger.info("Background updates started")
    
    def stop_background_updates(self):
        """Stop background thread for automatic pet state updates."""
        self._stop_updates = True
        if self._update_thread:
            self._update_thread.join(timeout=5)
        logger.info("Background updates stopped")
    
    def _background_update_loop(self):
        """Background loop for automatic pet state updates."""
        logger.info("Background update loop started")
        
        while not self._stop_updates:
            try:
                # Update all active pets
                for user_id in list(self.active_pets.keys()):
                    self.update_pet_state(user_id)
                
                # Sleep for update interval
                time.sleep(self._update_interval)
                
            except Exception as e:
                logger.error(f"Error in background update loop: {e}")
                time.sleep(self._update_interval)
        
        logger.info("Background update loop stopped")
    
    def get_care_actions(self, user_id: str) -> List[str]:
        """
        Get available care actions for user's DigiPal.
        
        Args:
            user_id: User ID
            
        Returns:
            List of available care action names
        """
        pet = self.active_pets.get(user_id)
        if not pet:
            pet = self.load_existing_pet(user_id)
        
        if pet:
            return self.attribute_engine.get_available_actions(pet)
        
        return []
    
    def apply_care_action(self, user_id: str, action_name: str) -> Tuple[bool, Interaction]:
        """
        Apply a care action to user's DigiPal.
        
        Args:
            user_id: User ID
            action_name: Name of care action to apply
            
        Returns:
            Tuple of (success, interaction_result)
        """
        pet = self.active_pets.get(user_id)
        if not pet:
            pet = self.load_existing_pet(user_id)
        
        if not pet:
            return False, Interaction(
                user_input=action_name,
                pet_response="No DigiPal found",
                success=False,
                result=InteractionResult.FAILURE
            )
        
        # Apply care action through attribute engine
        success, interaction = self.attribute_engine.apply_care_action(pet, action_name)
        
        # Save updated pet state
        if success:
            self.storage_manager.save_pet(pet)
        
        return success, interaction
    
    def get_pet_statistics(self, user_id: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics for user's DigiPal.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary with pet statistics
        """
        pet = self.active_pets.get(user_id)
        if not pet:
            pet = self.load_existing_pet(user_id)
        
        if not pet:
            return {}
        
        # Get interaction summary from AI communication
        interaction_summary = self.ai_communication.memory_manager.get_interaction_summary(pet)
        
        # Get care quality assessment
        care_assessment = self.attribute_engine.get_care_quality_assessment(pet)
        
        # Get evolution status
        evolution_eligible, next_stage, evolution_requirements = self.evolution_controller.check_evolution_eligibility(pet)
        
        return {
            'basic_info': {
                'id': pet.id,
                'name': pet.name,
                'life_stage': pet.life_stage.value,
                'generation': pet.generation,
                'age_hours': pet.get_age_hours(),
                'egg_type': pet.egg_type.value
            },
            'attributes': {
                'hp': pet.hp,
                'mp': pet.mp,
                'offense': pet.offense,
                'defense': pet.defense,
                'speed': pet.speed,
                'brains': pet.brains,
                'discipline': pet.discipline,
                'happiness': pet.happiness,
                'weight': pet.weight,
                'care_mistakes': pet.care_mistakes,
                'energy': pet.energy
            },
            'care_assessment': care_assessment,
            'interaction_summary': interaction_summary,
            'evolution_status': {
                'eligible': evolution_eligible,
                'next_stage': next_stage.value if next_stage else None,
                'requirements_met': evolution_requirements
            },
            'personality_traits': pet.personality_traits,
            'learned_commands': list(pet.learned_commands)
        }
    
    def process_audio_interaction(self, user_id: str, audio_data) -> Tuple[bool, Interaction]:
        """
        Process audio interaction with user's DigiPal.
        
        Args:
            user_id: User ID
            audio_data: Audio data from microphone
            
        Returns:
            Tuple of (success, interaction)
        """
        pet = self.active_pets.get(user_id)
        if not pet:
            pet = self.load_existing_pet(user_id)
        
        if not pet:
            error_interaction = Interaction(
                timestamp=datetime.now(),
                user_input="",
                interpreted_command="",
                pet_response="No DigiPal found. Please create one first.",
                attribute_changes={},
                success=False
            )
            return False, error_interaction
        
        try:
            # Process audio through AI communication layer
            text_input = self.ai_communication.process_speech(audio_data)
            
            if not text_input or text_input.strip() == "":
                error_interaction = Interaction(
                    timestamp=datetime.now(),
                    user_input="",
                    interpreted_command="",
                    pet_response="I couldn't understand what you said. Please try again.",
                    attribute_changes={},
                    success=False
                )
                return False, error_interaction
            
            # Process the transcribed text as a regular interaction
            return self.process_interaction(user_id, text_input)
            
        except Exception as e:
            logger.error(f"Error processing audio interaction for user {user_id}: {e}")
            error_interaction = Interaction(
                timestamp=datetime.now(),
                user_input="",
                interpreted_command="",
                pet_response=f"Sorry, I had trouble processing your voice. Error: {str(e)}",
                attribute_changes={},
                success=False
            )
            return False, error_interaction
    
    def clear_conversation_history(self, user_id: str) -> bool:
        """
        Clear conversation history for user's DigiPal.
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful, False otherwise
        """
        pet = self.active_pets.get(user_id)
        if not pet:
            pet = self.load_existing_pet(user_id)
        
        if not pet:
            return False
        
        try:
            # Clear conversation history in pet
            pet.conversation_history.clear()
            
            # Clear memory in AI communication layer
            self.ai_communication.memory_manager.clear_conversation_memory(pet)
            
            # Save the updated pet
            self.storage_manager.save_pet(pet)
            
            logger.info(f"Cleared conversation history for pet {pet.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing conversation history for user {user_id}: {e}")
            return False
    
    def shutdown(self):
        """Shutdown the DigiPal core engine and save all active pets."""
        logger.info("Shutting down DigiPalCore")
        
        # Stop background updates
        self.stop_background_updates()
        
        # Save all active pets
        for user_id, pet in self.active_pets.items():
            try:
                self.storage_manager.save_pet(pet)
                logger.info(f"Saved pet {pet.id} for user {user_id}")
            except Exception as e:
                logger.error(f"Failed to save pet {pet.id} during shutdown: {e}")
        
        # Clear active pets cache
        self.active_pets.clear()
        
        # Unload AI models to free memory
        self.ai_communication.unload_all_models()
        
        logger.info("DigiPalCore shutdown complete")