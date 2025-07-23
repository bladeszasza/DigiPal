"""
AttributeEngine for DigiPal - Implements Digimon World 1 inspired attribute calculations.
"""

from typing import Dict, List, Optional, Tuple
import random
from datetime import datetime, timedelta

from .models import DigiPal, CareAction, AttributeModifier, Interaction
from .enums import AttributeType, CareActionType, LifeStage, InteractionResult


class AttributeEngine:
    """
    Core engine for managing DigiPal attributes and care mechanics.
    Implements Digimon World 1 inspired attribute calculations with bounds checking.
    """
    
    # Attribute bounds (min, max)
    ATTRIBUTE_BOUNDS = {
        AttributeType.HP: (1, 999),
        AttributeType.MP: (0, 999),
        AttributeType.OFFENSE: (0, 999),
        AttributeType.DEFENSE: (0, 999),
        AttributeType.SPEED: (0, 999),
        AttributeType.BRAINS: (0, 999),
        AttributeType.DISCIPLINE: (0, 100),
        AttributeType.HAPPINESS: (0, 100),
        AttributeType.WEIGHT: (1, 99),
        AttributeType.CARE_MISTAKES: (0, 999),
        AttributeType.ENERGY: (0, 100)
    }
    
    # Energy decay rate per hour
    ENERGY_DECAY_RATE = 2.0
    
    # Happiness decay rate per hour
    HAPPINESS_DECAY_RATE = 1.0
    
    def __init__(self):
        """Initialize the AttributeEngine."""
        self.care_actions = self._initialize_care_actions()
    
    def _initialize_care_actions(self) -> Dict[str, CareAction]:
        """Initialize all available care actions with their effects."""
        actions = {}
        
        # Training actions
        actions["strength_training"] = CareAction(
            name="Strength Training",
            action_type=CareActionType.TRAIN,
            energy_cost=15,
            happiness_change=-5,
            attribute_modifiers=[
                AttributeModifier(AttributeType.OFFENSE, 3),
                AttributeModifier(AttributeType.HP, 2),
                AttributeModifier(AttributeType.WEIGHT, -1)
            ],
            success_conditions=["energy >= 15"],
            failure_effects=[
                AttributeModifier(AttributeType.CARE_MISTAKES, 1)
            ]
        )
        
        actions["defense_training"] = CareAction(
            name="Defense Training",
            action_type=CareActionType.TRAIN,
            energy_cost=15,
            happiness_change=-5,
            attribute_modifiers=[
                AttributeModifier(AttributeType.DEFENSE, 3),
                AttributeModifier(AttributeType.HP, 2),
                AttributeModifier(AttributeType.WEIGHT, -1)
            ],
            success_conditions=["energy >= 15"],
            failure_effects=[
                AttributeModifier(AttributeType.CARE_MISTAKES, 1)
            ]
        )
        
        actions["speed_training"] = CareAction(
            name="Speed Training",
            action_type=CareActionType.TRAIN,
            energy_cost=12,
            happiness_change=-3,
            attribute_modifiers=[
                AttributeModifier(AttributeType.SPEED, 3),
                AttributeModifier(AttributeType.WEIGHT, -2)
            ],
            success_conditions=["energy >= 12"],
            failure_effects=[
                AttributeModifier(AttributeType.CARE_MISTAKES, 1)
            ]
        )
        
        actions["brain_training"] = CareAction(
            name="Brain Training",
            action_type=CareActionType.TRAIN,
            energy_cost=10,
            happiness_change=-2,
            attribute_modifiers=[
                AttributeModifier(AttributeType.BRAINS, 3),
                AttributeModifier(AttributeType.MP, 2)
            ],
            success_conditions=["energy >= 10"],
            failure_effects=[
                AttributeModifier(AttributeType.CARE_MISTAKES, 1)
            ]
        )
        
        # Advanced training actions
        actions["endurance_training"] = CareAction(
            name="Endurance Training",
            action_type=CareActionType.TRAIN,
            energy_cost=20,
            happiness_change=-8,
            attribute_modifiers=[
                AttributeModifier(AttributeType.HP, 4),
                AttributeModifier(AttributeType.DEFENSE, 2),
                AttributeModifier(AttributeType.WEIGHT, -2)
            ],
            success_conditions=["energy >= 20"],
            failure_effects=[
                AttributeModifier(AttributeType.CARE_MISTAKES, 1)
            ]
        )
        
        actions["agility_training"] = CareAction(
            name="Agility Training",
            action_type=CareActionType.TRAIN,
            energy_cost=18,
            happiness_change=-6,
            attribute_modifiers=[
                AttributeModifier(AttributeType.SPEED, 4),
                AttributeModifier(AttributeType.OFFENSE, 1),
                AttributeModifier(AttributeType.WEIGHT, -3)
            ],
            success_conditions=["energy >= 18"],
            failure_effects=[
                AttributeModifier(AttributeType.CARE_MISTAKES, 1)
            ]
        )
        
        # Feeding actions
        actions["meat"] = CareAction(
            name="Feed Meat",
            action_type=CareActionType.FEED,
            energy_cost=0,
            happiness_change=5,
            attribute_modifiers=[
                AttributeModifier(AttributeType.WEIGHT, 2),
                AttributeModifier(AttributeType.HP, 1),
                AttributeModifier(AttributeType.OFFENSE, 1)
            ],
            success_conditions=[],
            failure_effects=[]
        )
        
        actions["fish"] = CareAction(
            name="Feed Fish",
            action_type=CareActionType.FEED,
            energy_cost=0,
            happiness_change=3,
            attribute_modifiers=[
                AttributeModifier(AttributeType.WEIGHT, 1),
                AttributeModifier(AttributeType.BRAINS, 1),
                AttributeModifier(AttributeType.MP, 1)
            ],
            success_conditions=[],
            failure_effects=[]
        )
        
        actions["vegetables"] = CareAction(
            name="Feed Vegetables",
            action_type=CareActionType.FEED,
            energy_cost=0,
            happiness_change=2,
            attribute_modifiers=[
                AttributeModifier(AttributeType.WEIGHT, 1),
                AttributeModifier(AttributeType.DEFENSE, 1)
            ],
            success_conditions=[],
            failure_effects=[]
        )
        
        # Additional food types for variety
        actions["protein_shake"] = CareAction(
            name="Feed Protein Shake",
            action_type=CareActionType.FEED,
            energy_cost=0,
            happiness_change=1,
            attribute_modifiers=[
                AttributeModifier(AttributeType.WEIGHT, 1),
                AttributeModifier(AttributeType.OFFENSE, 2),
                AttributeModifier(AttributeType.HP, 1)
            ],
            success_conditions=[],
            failure_effects=[]
        )
        
        actions["energy_drink"] = CareAction(
            name="Feed Energy Drink",
            action_type=CareActionType.FEED,
            energy_cost=-5,  # Restores some energy
            happiness_change=3,
            attribute_modifiers=[
                AttributeModifier(AttributeType.SPEED, 1),
                AttributeModifier(AttributeType.MP, 1)
            ],
            success_conditions=[],
            failure_effects=[]
        )
        
        # Care actions
        actions["praise"] = CareAction(
            name="Praise",
            action_type=CareActionType.PRAISE,
            energy_cost=0,
            happiness_change=10,
            attribute_modifiers=[
                AttributeModifier(AttributeType.DISCIPLINE, -2)
            ],
            success_conditions=[],
            failure_effects=[]
        )
        
        actions["scold"] = CareAction(
            name="Scold",
            action_type=CareActionType.SCOLD,
            energy_cost=0,
            happiness_change=-8,
            attribute_modifiers=[
                AttributeModifier(AttributeType.DISCIPLINE, 5)
            ],
            success_conditions=[],
            failure_effects=[]
        )
        
        actions["rest"] = CareAction(
            name="Rest",
            action_type=CareActionType.REST,
            energy_cost=-30,  # Negative cost means it restores energy
            happiness_change=3,
            attribute_modifiers=[],
            success_conditions=[],
            failure_effects=[]
        )
        
        actions["play"] = CareAction(
            name="Play",
            action_type=CareActionType.PLAY,
            energy_cost=8,
            happiness_change=8,
            attribute_modifiers=[
                AttributeModifier(AttributeType.WEIGHT, -1)
            ],
            success_conditions=["energy >= 8"],
            failure_effects=[
                AttributeModifier(AttributeType.CARE_MISTAKES, 1)
            ]
        )
        
        return actions
    
    def apply_care_action(self, pet: DigiPal, action_name: str) -> Tuple[bool, Interaction]:
        """
        Apply a care action to the DigiPal and return the result.
        
        Args:
            pet: The DigiPal to apply the action to
            action_name: Name of the care action to apply
            
        Returns:
            Tuple of (success, interaction_record)
        """
        if action_name not in self.care_actions:
            interaction = Interaction(
                user_input=action_name,
                interpreted_command=action_name,
                pet_response=f"Unknown care action: {action_name}",
                success=False,
                result=InteractionResult.INVALID_COMMAND
            )
            return False, interaction
        
        action = self.care_actions[action_name]
        
        # Check success conditions
        success = self._check_action_conditions(pet, action)
        
        attribute_changes = {}
        
        if success:
            # Apply energy cost
            if action.energy_cost > 0:
                pet.modify_attribute(AttributeType.ENERGY, -action.energy_cost)
                attribute_changes["energy"] = -action.energy_cost
            elif action.energy_cost < 0:  # Rest action restores energy
                pet.modify_attribute(AttributeType.ENERGY, -action.energy_cost)
                attribute_changes["energy"] = -action.energy_cost
            
            # Apply happiness change
            pet.modify_attribute(AttributeType.HAPPINESS, action.happiness_change)
            attribute_changes["happiness"] = action.happiness_change
            
            # Apply attribute modifiers
            for modifier in action.attribute_modifiers:
                if self._check_modifier_conditions(pet, modifier):
                    pet.modify_attribute(modifier.attribute, modifier.change)
                    attribute_changes[modifier.attribute.value] = modifier.change
            
            response = f"Successfully performed {action.name}!"
            result = InteractionResult.SUCCESS
            
        else:
            # Apply failure effects
            for modifier in action.failure_effects:
                pet.modify_attribute(modifier.attribute, modifier.change)
                attribute_changes[modifier.attribute.value] = modifier.change
            
            response = f"Failed to perform {action.name} - insufficient energy or conditions not met"
            result = InteractionResult.INSUFFICIENT_ENERGY
        
        # Update last interaction time
        pet.last_interaction = datetime.now()
        
        interaction = Interaction(
            user_input=action_name,
            interpreted_command=action_name,
            pet_response=response,
            attribute_changes=attribute_changes,
            success=success,
            result=result
        )
        
        # Add to conversation history
        pet.conversation_history.append(interaction)
        
        return success, interaction
    
    def _check_action_conditions(self, pet: DigiPal, action: CareAction) -> bool:
        """Check if all conditions for an action are met."""
        for condition in action.success_conditions:
            if not self._evaluate_condition(pet, condition):
                return False
        return True
    
    def _check_modifier_conditions(self, pet: DigiPal, modifier: AttributeModifier) -> bool:
        """Check if all conditions for an attribute modifier are met."""
        for condition in modifier.conditions:
            if not self._evaluate_condition(pet, condition):
                return False
        return True
    
    def _evaluate_condition(self, pet: DigiPal, condition: str) -> bool:
        """Evaluate a condition string against the pet's current state."""
        # Simple condition parser for basic comparisons
        if ">=" in condition:
            attr_name, value = condition.split(">=")
            attr_name = attr_name.strip()
            value = int(value.strip())
            
            if attr_name == "energy":
                return pet.energy >= value
            elif attr_name == "happiness":
                return pet.happiness >= value
            elif attr_name == "discipline":
                return pet.discipline >= value
            # Add more conditions as needed
        
        return True  # Default to true for unknown conditions
    
    def apply_time_decay(self, pet: DigiPal, hours_passed: float) -> Dict[str, int]:
        """
        Apply time-based attribute decay to the DigiPal.
        
        Args:
            pet: The DigiPal to apply decay to
            hours_passed: Number of hours that have passed
            
        Returns:
            Dictionary of attribute changes applied
        """
        changes = {}
        
        # Energy decay
        energy_decay = int(hours_passed * self.ENERGY_DECAY_RATE)
        if energy_decay > 0:
            old_energy = pet.energy
            pet.modify_attribute(AttributeType.ENERGY, -energy_decay)
            changes["energy"] = pet.energy - old_energy
        
        # Happiness decay
        happiness_decay = int(hours_passed * self.HAPPINESS_DECAY_RATE)
        if happiness_decay > 0:
            old_happiness = pet.happiness
            pet.modify_attribute(AttributeType.HAPPINESS, -happiness_decay)
            changes["happiness"] = pet.happiness - old_happiness
        
        # Weight changes based on energy levels
        if pet.energy < 20:  # Very low energy causes weight loss
            weight_change = -max(1, int(hours_passed * 0.5))
            old_weight = pet.weight
            pet.modify_attribute(AttributeType.WEIGHT, weight_change)
            changes["weight"] = pet.weight - old_weight
        
        return changes
    
    def calculate_care_mistake(self, pet: DigiPal, action_type: CareActionType) -> bool:
        """
        Calculate if a care mistake should be recorded based on pet state and action.
        
        Args:
            pet: The DigiPal being cared for
            action_type: Type of care action being performed
            
        Returns:
            True if a care mistake should be recorded
        """
        mistake_probability = 0.0
        
        # Higher chance of mistakes when pet is in poor condition
        if pet.energy < 20:
            mistake_probability += 0.3
        if pet.happiness < 20:
            mistake_probability += 0.2
        if pet.weight < 10 or pet.weight > 80:
            mistake_probability += 0.2
        
        # Training when tired is more likely to cause mistakes
        if action_type == CareActionType.TRAIN and pet.energy < 30:
            mistake_probability += 0.4
        
        # Overfeeding increases mistake chance
        if action_type == CareActionType.FEED and pet.weight > 70:
            mistake_probability += 0.3
        
        # Excessive scolding increases mistake chance
        if action_type == CareActionType.SCOLD and pet.discipline > 80:
            mistake_probability += 0.25
        
        # Random chance
        return random.random() < mistake_probability
    
    def get_care_quality_assessment(self, pet: DigiPal) -> Dict[str, str]:
        """
        Assess the overall care quality based on pet's current state.
        
        Args:
            pet: The DigiPal to assess
            
        Returns:
            Dictionary with care quality metrics
        """
        assessment = {}
        
        # Energy assessment
        if pet.energy >= 80:
            assessment["energy"] = "excellent"
        elif pet.energy >= 60:
            assessment["energy"] = "good"
        elif pet.energy >= 40:
            assessment["energy"] = "fair"
        elif pet.energy >= 20:
            assessment["energy"] = "poor"
        else:
            assessment["energy"] = "critical"
        
        # Happiness assessment
        if pet.happiness >= 80:
            assessment["happiness"] = "very_happy"
        elif pet.happiness >= 60:
            assessment["happiness"] = "happy"
        elif pet.happiness >= 40:
            assessment["happiness"] = "neutral"
        elif pet.happiness >= 20:
            assessment["happiness"] = "sad"
        else:
            assessment["happiness"] = "very_sad"
        
        # Weight assessment
        if 15 <= pet.weight <= 35:
            assessment["weight"] = "healthy"
        elif 10 <= pet.weight < 15 or 35 < pet.weight <= 50:
            assessment["weight"] = "slightly_off"
        elif 5 <= pet.weight < 10 or 50 < pet.weight <= 70:
            assessment["weight"] = "concerning"
        else:
            assessment["weight"] = "unhealthy"
        
        # Discipline assessment
        if 40 <= pet.discipline <= 70:
            assessment["discipline"] = "balanced"
        elif pet.discipline < 40:
            assessment["discipline"] = "undisciplined"
        else:
            assessment["discipline"] = "over_disciplined"
        
        # Care mistakes assessment
        if pet.care_mistakes == 0:
            assessment["care_quality"] = "perfect"
        elif pet.care_mistakes <= 3:
            assessment["care_quality"] = "excellent"
        elif pet.care_mistakes <= 7:
            assessment["care_quality"] = "good"
        elif pet.care_mistakes <= 15:
            assessment["care_quality"] = "fair"
        else:
            assessment["care_quality"] = "poor"
        
        return assessment
    
    def get_attribute_bounds(self, attribute: AttributeType) -> Tuple[int, int]:
        """Get the min and max bounds for an attribute."""
        return self.ATTRIBUTE_BOUNDS.get(attribute, (0, 999))
    
    def validate_attribute_value(self, attribute: AttributeType, value: int) -> int:
        """Validate and clamp an attribute value to its bounds."""
        min_val, max_val = self.get_attribute_bounds(attribute)
        return max(min_val, min(max_val, value))
    
    def get_available_actions(self, pet: DigiPal) -> List[str]:
        """Get list of care actions available for the current pet state."""
        available = []
        
        for action_name, action in self.care_actions.items():
            # Check if pet has enough energy for the action
            if action.energy_cost > 0 and pet.energy < action.energy_cost:
                continue
            
            # Check life stage appropriateness
            if self._is_action_appropriate_for_stage(action, pet.life_stage):
                available.append(action_name)
        
        return available
    
    def _is_action_appropriate_for_stage(self, action: CareAction, stage: LifeStage) -> bool:
        """Check if an action is appropriate for the current life stage."""
        # All stages can do basic care
        basic_actions = {CareActionType.FEED, CareActionType.PRAISE, CareActionType.SCOLD, CareActionType.REST}
        
        if action.action_type in basic_actions:
            return True
        
        # Training and play require child stage or higher
        if action.action_type in {CareActionType.TRAIN, CareActionType.PLAY}:
            return stage in {LifeStage.CHILD, LifeStage.TEEN, LifeStage.YOUNG_ADULT, LifeStage.ADULT, LifeStage.ELDERLY}
        
        return True
    
    def get_care_action(self, action_name: str) -> Optional[CareAction]:
        """Get a care action by name."""
        return self.care_actions.get(action_name)
    
    def get_all_care_actions(self) -> Dict[str, CareAction]:
        """Get all available care actions."""
        return self.care_actions.copy()