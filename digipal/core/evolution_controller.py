"""
EvolutionController for DigiPal - Manages life stage progression and generational inheritance.
"""

from typing import Dict, List, Optional, Tuple, Any
import random
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from .models import DigiPal, AttributeModifier
from .enums import LifeStage, EggType, AttributeType


@dataclass
class EvolutionRequirement:
    """Represents requirements for evolution to a specific life stage."""
    min_attributes: Dict[AttributeType, int] = field(default_factory=dict)
    max_attributes: Dict[AttributeType, int] = field(default_factory=dict)
    max_care_mistakes: int = 999
    min_age_hours: float = 0.0
    max_age_hours: float = float('inf')
    required_actions: List[str] = field(default_factory=list)
    happiness_threshold: int = 0
    discipline_range: Tuple[int, int] = (0, 100)
    weight_range: Tuple[int, int] = (1, 99)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert EvolutionRequirement to dictionary."""
        return {
            'min_attributes': {attr.value: val for attr, val in self.min_attributes.items()},
            'max_attributes': {attr.value: val for attr, val in self.max_attributes.items()},
            'max_care_mistakes': self.max_care_mistakes,
            'min_age_hours': self.min_age_hours,
            'max_age_hours': self.max_age_hours,
            'required_actions': self.required_actions,
            'happiness_threshold': self.happiness_threshold,
            'discipline_range': self.discipline_range,
            'weight_range': self.weight_range
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvolutionRequirement':
        """Create EvolutionRequirement from dictionary."""
        data['min_attributes'] = {AttributeType(attr): val for attr, val in data.get('min_attributes', {}).items()}
        data['max_attributes'] = {AttributeType(attr): val for attr, val in data.get('max_attributes', {}).items()}
        return cls(**data)


@dataclass
class EvolutionResult:
    """Result of an evolution attempt."""
    success: bool
    old_stage: LifeStage
    new_stage: LifeStage
    attribute_changes: Dict[str, int] = field(default_factory=dict)
    message: str = ""
    requirements_met: Dict[str, bool] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert EvolutionResult to dictionary."""
        return {
            'success': self.success,
            'old_stage': self.old_stage.value,
            'new_stage': self.new_stage.value,
            'attribute_changes': self.attribute_changes,
            'message': self.message,
            'requirements_met': self.requirements_met
        }


@dataclass
class DNAInheritance:
    """Represents DNA inheritance data for generational passing."""
    parent_attributes: Dict[AttributeType, int] = field(default_factory=dict)
    parent_care_quality: str = "fair"
    parent_final_stage: LifeStage = LifeStage.ADULT
    parent_egg_type: EggType = EggType.RED
    generation: int = 1
    inheritance_bonuses: Dict[AttributeType, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert DNAInheritance to dictionary."""
        return {
            'parent_attributes': {attr.value: val for attr, val in self.parent_attributes.items()},
            'parent_care_quality': self.parent_care_quality,
            'parent_final_stage': self.parent_final_stage.value,
            'parent_egg_type': self.parent_egg_type.value,
            'generation': self.generation,
            'inheritance_bonuses': {attr.value: val for attr, val in self.inheritance_bonuses.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DNAInheritance':
        """Create DNAInheritance from dictionary."""
        data['parent_attributes'] = {AttributeType(attr): val for attr, val in data.get('parent_attributes', {}).items()}
        data['parent_final_stage'] = LifeStage(data['parent_final_stage'])
        data['parent_egg_type'] = EggType(data['parent_egg_type'])
        data['inheritance_bonuses'] = {AttributeType(attr): val for attr, val in data.get('inheritance_bonuses', {}).items()}
        return cls(**data)


class EvolutionController:
    """
    Manages DigiPal evolution through life stages and generational inheritance.
    Implements time-based evolution triggers and DNA-based attribute passing.
    """
    
    # Evolution timing configuration (hours)
    EVOLUTION_TIMINGS = {
        LifeStage.EGG: 0.5,        # 30 minutes to hatch
        LifeStage.BABY: 24.0,      # 1 day as baby
        LifeStage.CHILD: 72.0,     # 3 days as child
        LifeStage.TEEN: 120.0,     # 5 days as teen
        LifeStage.YOUNG_ADULT: 168.0,  # 7 days as young adult
        LifeStage.ADULT: 240.0,    # 10 days as adult
        LifeStage.ELDERLY: 72.0    # 3 days as elderly before death
    }
    
    def __init__(self):
        """Initialize the EvolutionController."""
        self.evolution_requirements = self._initialize_evolution_requirements()
    
    def _initialize_evolution_requirements(self) -> Dict[LifeStage, EvolutionRequirement]:
        """Initialize evolution requirements for each life stage transition."""
        requirements = {}
        
        # EGG -> BABY: Triggered by first speech interaction
        requirements[LifeStage.BABY] = EvolutionRequirement(
            min_age_hours=0.0,
            max_care_mistakes=0,
            happiness_threshold=0
        )
        
        # BABY -> CHILD: Basic care requirements
        requirements[LifeStage.CHILD] = EvolutionRequirement(
            min_age_hours=20.0,
            max_care_mistakes=5,
            happiness_threshold=30,
            min_attributes={
                AttributeType.HP: 80,
                AttributeType.ENERGY: 20
            }
        )
        
        # CHILD -> TEEN: Balanced development
        requirements[LifeStage.TEEN] = EvolutionRequirement(
            min_age_hours=60.0,
            max_care_mistakes=10,
            happiness_threshold=40,
            discipline_range=(20, 80),
            weight_range=(15, 50),
            min_attributes={
                AttributeType.HP: 120,
                AttributeType.OFFENSE: 15,
                AttributeType.DEFENSE: 15
            }
        )
        
        # TEEN -> YOUNG_ADULT: Specialized development paths
        requirements[LifeStage.YOUNG_ADULT] = EvolutionRequirement(
            min_age_hours=100.0,
            max_care_mistakes=15,
            happiness_threshold=50,
            discipline_range=(30, 70),
            weight_range=(15, 40),
            min_attributes={
                AttributeType.HP: 150,
                AttributeType.OFFENSE: 25,
                AttributeType.DEFENSE: 25,
                AttributeType.SPEED: 20,
                AttributeType.BRAINS: 20
            }
        )
        
        # YOUNG_ADULT -> ADULT: Peak performance requirements
        requirements[LifeStage.ADULT] = EvolutionRequirement(
            min_age_hours=150.0,
            max_care_mistakes=20,
            happiness_threshold=60,
            discipline_range=(40, 60),
            weight_range=(20, 35),
            min_attributes={
                AttributeType.HP: 200,
                AttributeType.OFFENSE: 40,
                AttributeType.DEFENSE: 40,
                AttributeType.SPEED: 35,
                AttributeType.BRAINS: 35
            }
        )
        
        # ADULT -> ELDERLY: Automatic after time limit
        requirements[LifeStage.ELDERLY] = EvolutionRequirement(
            min_age_hours=200.0,
            max_care_mistakes=999,  # No care mistake limit for elderly
            happiness_threshold=0
        )
        
        return requirements
    
    def check_evolution_eligibility(self, pet: DigiPal) -> Tuple[bool, LifeStage, Dict[str, bool]]:
        """
        Check if a DigiPal is eligible for evolution to the next stage.
        
        Args:
            pet: The DigiPal to check
            
        Returns:
            Tuple of (eligible, next_stage, requirements_status)
        """
        current_stage = pet.life_stage
        next_stage = self._get_next_life_stage(current_stage)
        
        if next_stage is None:
            return False, current_stage, {}
        
        requirements = self.evolution_requirements.get(next_stage)
        if not requirements:
            return False, current_stage, {}
        
        requirements_status = self._evaluate_evolution_requirements(pet, requirements)
        eligible = all(requirements_status.values())
        
        return eligible, next_stage, requirements_status
    
    def _get_next_life_stage(self, current_stage: LifeStage) -> Optional[LifeStage]:
        """Get the next life stage in the progression."""
        stage_progression = [
            LifeStage.EGG,
            LifeStage.BABY,
            LifeStage.CHILD,
            LifeStage.TEEN,
            LifeStage.YOUNG_ADULT,
            LifeStage.ADULT,
            LifeStage.ELDERLY
        ]
        
        try:
            current_index = stage_progression.index(current_stage)
            if current_index < len(stage_progression) - 1:
                return stage_progression[current_index + 1]
        except ValueError:
            pass
        
        return None
    
    def _evaluate_evolution_requirements(self, pet: DigiPal, requirements: EvolutionRequirement) -> Dict[str, bool]:
        """Evaluate all evolution requirements for a pet."""
        status = {}
        
        # Check age requirements
        age_hours = pet.get_age_hours()
        status['min_age'] = age_hours >= requirements.min_age_hours
        status['max_age'] = age_hours <= requirements.max_age_hours
        
        # Check care mistakes
        status['care_mistakes'] = pet.care_mistakes <= requirements.max_care_mistakes
        
        # Check happiness threshold
        status['happiness'] = pet.happiness >= requirements.happiness_threshold
        
        # Check discipline range
        discipline_min, discipline_max = requirements.discipline_range
        status['discipline'] = discipline_min <= pet.discipline <= discipline_max
        
        # Check weight range
        weight_min, weight_max = requirements.weight_range
        status['weight'] = weight_min <= pet.weight <= weight_max
        
        # Check minimum attributes
        for attr, min_val in requirements.min_attributes.items():
            current_val = pet.get_attribute(attr)
            status[f'min_{attr.value}'] = current_val >= min_val
        
        # Check maximum attributes
        for attr, max_val in requirements.max_attributes.items():
            current_val = pet.get_attribute(attr)
            status[f'max_{attr.value}'] = current_val <= max_val
        
        # Check required actions (placeholder for future implementation)
        if requirements.required_actions:
            status['required_actions'] = True  # Simplified for now
        
        return status
    
    def trigger_evolution(self, pet: DigiPal, force: bool = False) -> EvolutionResult:
        """
        Trigger evolution for a DigiPal if eligible.
        
        Args:
            pet: The DigiPal to evolve
            force: Force evolution regardless of requirements (for testing)
            
        Returns:
            EvolutionResult with evolution outcome
        """
        old_stage = pet.life_stage
        
        if not force:
            eligible, next_stage, requirements_status = self.check_evolution_eligibility(pet)
            
            if not eligible:
                return EvolutionResult(
                    success=False,
                    old_stage=old_stage,
                    new_stage=old_stage,
                    message="Evolution requirements not met",
                    requirements_met=requirements_status
                )
        else:
            next_stage = self._get_next_life_stage(old_stage)
            if next_stage is None:
                return EvolutionResult(
                    success=False,
                    old_stage=old_stage,
                    new_stage=old_stage,
                    message="No next evolution stage available"
                )
            requirements_status = {}
        
        # Perform evolution
        attribute_changes = self._apply_evolution_changes(pet, old_stage, next_stage)
        pet.life_stage = next_stage
        pet.evolution_timer = 0.0  # Reset evolution timer
        
        # Update learned commands based on new stage
        self._update_learned_commands(pet, next_stage)
        
        # Generate evolution message
        message = f"Evolution successful! {old_stage.value.title()} -> {next_stage.value.title()}"
        
        return EvolutionResult(
            success=True,
            old_stage=old_stage,
            new_stage=next_stage,
            attribute_changes=attribute_changes,
            message=message,
            requirements_met=requirements_status
        )
    
    def _apply_evolution_changes(self, pet: DigiPal, old_stage: LifeStage, new_stage: LifeStage) -> Dict[str, int]:
        """Apply attribute changes during evolution."""
        changes = {}
        
        # Base evolution bonuses by stage
        evolution_bonuses = {
            LifeStage.BABY: {
                AttributeType.HP: 20,
                AttributeType.MP: 10,
                AttributeType.HAPPINESS: 10
            },
            LifeStage.CHILD: {
                AttributeType.HP: 30,
                AttributeType.MP: 15,
                AttributeType.OFFENSE: 5,
                AttributeType.DEFENSE: 5,
                AttributeType.SPEED: 5,
                AttributeType.BRAINS: 5
            },
            LifeStage.TEEN: {
                AttributeType.HP: 40,
                AttributeType.MP: 20,
                AttributeType.OFFENSE: 10,
                AttributeType.DEFENSE: 10,
                AttributeType.SPEED: 10,
                AttributeType.BRAINS: 10
            },
            LifeStage.YOUNG_ADULT: {
                AttributeType.HP: 50,
                AttributeType.MP: 25,
                AttributeType.OFFENSE: 15,
                AttributeType.DEFENSE: 15,
                AttributeType.SPEED: 15,
                AttributeType.BRAINS: 15
            },
            LifeStage.ADULT: {
                AttributeType.HP: 60,
                AttributeType.MP: 30,
                AttributeType.OFFENSE: 20,
                AttributeType.DEFENSE: 20,
                AttributeType.SPEED: 20,
                AttributeType.BRAINS: 20
            },
            LifeStage.ELDERLY: {
                AttributeType.HP: -20,  # Elderly lose some physical attributes
                AttributeType.OFFENSE: -10,
                AttributeType.DEFENSE: -5,
                AttributeType.SPEED: -15,
                AttributeType.BRAINS: 10,  # But gain wisdom
                AttributeType.MP: 20
            }
        }
        
        bonuses = evolution_bonuses.get(new_stage, {})
        
        for attr, bonus in bonuses.items():
            old_value = pet.get_attribute(attr)
            pet.modify_attribute(attr, bonus)
            new_value = pet.get_attribute(attr)
            changes[attr.value] = new_value - old_value
        
        # Egg type specific bonuses
        if new_stage in [LifeStage.CHILD, LifeStage.TEEN, LifeStage.YOUNG_ADULT]:
            egg_bonuses = self._get_egg_type_evolution_bonus(pet.egg_type, new_stage)
            for attr, bonus in egg_bonuses.items():
                old_value = pet.get_attribute(attr)
                pet.modify_attribute(attr, bonus)
                new_value = pet.get_attribute(attr)
                if attr.value in changes:
                    changes[attr.value] += new_value - old_value
                else:
                    changes[attr.value] = new_value - old_value
        
        return changes
    
    def _get_egg_type_evolution_bonus(self, egg_type: EggType, stage: LifeStage) -> Dict[AttributeType, int]:
        """Get egg type specific evolution bonuses."""
        bonuses = {}
        
        if egg_type == EggType.RED:  # Fire-oriented
            bonuses = {
                AttributeType.OFFENSE: 5,
                AttributeType.SPEED: 3,
                AttributeType.HP: 2
            }
        elif egg_type == EggType.BLUE:  # Water-oriented
            bonuses = {
                AttributeType.DEFENSE: 5,
                AttributeType.MP: 5,
                AttributeType.BRAINS: 3
            }
        elif egg_type == EggType.GREEN:  # Earth-oriented
            bonuses = {
                AttributeType.HP: 8,
                AttributeType.DEFENSE: 3,
                AttributeType.BRAINS: 2
            }
        
        # Scale bonuses by stage
        stage_multipliers = {
            LifeStage.CHILD: 1.0,
            LifeStage.TEEN: 1.5,
            LifeStage.YOUNG_ADULT: 2.0
        }
        
        multiplier = stage_multipliers.get(stage, 1.0)
        return {attr: int(bonus * multiplier) for attr, bonus in bonuses.items()}
    
    def _update_learned_commands(self, pet: DigiPal, new_stage: LifeStage):
        """Update learned commands based on new life stage."""
        stage_commands = {
            LifeStage.EGG: set(),
            LifeStage.BABY: {"eat", "sleep", "good", "bad"},
            LifeStage.CHILD: {"eat", "sleep", "good", "bad", "play", "train"},
            LifeStage.TEEN: {"eat", "sleep", "good", "bad", "play", "train", "status", "talk"},
            LifeStage.YOUNG_ADULT: {"eat", "sleep", "good", "bad", "play", "train", "status", "talk", "battle"},
            LifeStage.ADULT: {"eat", "sleep", "good", "bad", "play", "train", "status", "talk", "battle", "teach"},
            LifeStage.ELDERLY: {"eat", "sleep", "good", "bad", "play", "train", "status", "talk", "battle", "teach", "wisdom"}
        }
        
        new_commands = stage_commands.get(new_stage, set())
        pet.learned_commands.update(new_commands)
    
    def check_time_based_evolution(self, pet: DigiPal) -> bool:
        """
        Check if enough time has passed for automatic evolution.
        
        Args:
            pet: The DigiPal to check
            
        Returns:
            True if time-based evolution should be triggered
        """
        current_stage = pet.life_stage
        required_time = self.EVOLUTION_TIMINGS.get(current_stage, float('inf'))
        age_hours = pet.get_age_hours()
        
        return age_hours >= required_time
    
    def update_evolution_timer(self, pet: DigiPal, hours_passed: float):
        """
        Update the evolution timer for a DigiPal.
        
        Args:
            pet: The DigiPal to update
            hours_passed: Number of hours that have passed
        """
        pet.evolution_timer += hours_passed
    
    def create_inheritance_dna(self, parent: DigiPal, care_quality: str) -> DNAInheritance:
        """
        Create DNA inheritance data from a parent DigiPal.
        
        Args:
            parent: The parent DigiPal
            care_quality: Quality of care the parent received
            
        Returns:
            DNAInheritance object with inheritance data
        """
        # Capture parent's final attributes
        parent_attributes = {
            AttributeType.HP: parent.hp,
            AttributeType.MP: parent.mp,
            AttributeType.OFFENSE: parent.offense,
            AttributeType.DEFENSE: parent.defense,
            AttributeType.SPEED: parent.speed,
            AttributeType.BRAINS: parent.brains
        }
        
        # Calculate inheritance bonuses based on parent's attributes and care quality
        inheritance_bonuses = self._calculate_inheritance_bonuses(parent, care_quality)
        
        return DNAInheritance(
            parent_attributes=parent_attributes,
            parent_care_quality=care_quality,
            parent_final_stage=parent.life_stage,
            parent_egg_type=parent.egg_type,
            generation=parent.generation + 1,
            inheritance_bonuses=inheritance_bonuses
        )
    
    def _calculate_inheritance_bonuses(self, parent: DigiPal, care_quality: str) -> Dict[AttributeType, int]:
        """Calculate attribute bonuses for inheritance based on parent stats and care quality."""
        bonuses = {}
        
        # Base inheritance percentages by care quality
        inheritance_rates = {
            "perfect": 0.25,
            "excellent": 0.20,
            "good": 0.15,
            "fair": 0.10,
            "poor": 0.05
        }
        
        rate = inheritance_rates.get(care_quality, 0.10)
        
        # Calculate bonuses based on parent's final attributes
        primary_attributes = [
            AttributeType.HP, AttributeType.MP, AttributeType.OFFENSE,
            AttributeType.DEFENSE, AttributeType.SPEED, AttributeType.BRAINS
        ]
        
        for attr in primary_attributes:
            parent_value = parent.get_attribute(attr)
            # Higher parent attributes provide better inheritance bonuses
            if parent_value > 100:  # Above average
                bonus = int((parent_value - 100) * rate)
                bonuses[attr] = max(1, bonus)  # Minimum bonus of 1
        
        # Special bonuses for exceptional care
        if care_quality == "perfect":
            # Perfect care provides additional random bonuses
            bonus_attr = random.choice(primary_attributes)
            bonuses[bonus_attr] = bonuses.get(bonus_attr, 0) + random.randint(5, 15)
        
        return bonuses
    
    def apply_inheritance(self, offspring: DigiPal, dna: DNAInheritance):
        """
        Apply DNA inheritance to a new DigiPal.
        
        Args:
            offspring: The new DigiPal to apply inheritance to
            dna: The DNA inheritance data
        """
        offspring.generation = dna.generation
        
        # Apply inheritance bonuses
        for attr, bonus in dna.inheritance_bonuses.items():
            offspring.modify_attribute(attr, bonus)
        
        # Add some randomization to prevent identical offspring
        primary_attributes = [
            AttributeType.HP, AttributeType.MP, AttributeType.OFFENSE,
            AttributeType.DEFENSE, AttributeType.SPEED, AttributeType.BRAINS
        ]
        
        for attr in primary_attributes:
            # Small random variation (-2 to +2)
            variation = random.randint(-2, 2)
            offspring.modify_attribute(attr, variation)
        
        # Inherit some personality traits (placeholder for future implementation)
        offspring.personality_traits["inherited"] = True
        offspring.personality_traits["parent_care_quality"] = dna.parent_care_quality
    
    def is_death_time(self, pet: DigiPal) -> bool:
        """
        Check if a DigiPal should die (elderly stage time limit reached).
        
        Args:
            pet: The DigiPal to check
            
        Returns:
            True if the DigiPal should die
        """
        if pet.life_stage != LifeStage.ELDERLY:
            return False
        
        # Calculate total age and time limits for all previous stages
        age_hours = pet.get_age_hours()
        elderly_time_limit = self.EVOLUTION_TIMINGS.get(LifeStage.ELDERLY, 72.0)
        
        # Calculate minimum time to reach elderly stage
        min_time_to_elderly = sum(
            self.EVOLUTION_TIMINGS.get(stage, 0) 
            for stage in [LifeStage.EGG, LifeStage.BABY, LifeStage.CHILD, 
                         LifeStage.TEEN, LifeStage.YOUNG_ADULT, LifeStage.ADULT]
        )
        
        # If pet is old enough to have been elderly for the full elderly duration
        total_max_lifetime = min_time_to_elderly + elderly_time_limit
        return age_hours >= total_max_lifetime
    
    def get_evolution_requirements(self, stage: LifeStage) -> Optional[EvolutionRequirement]:
        """Get evolution requirements for a specific stage."""
        return self.evolution_requirements.get(stage)
    
    def get_all_evolution_requirements(self) -> Dict[LifeStage, EvolutionRequirement]:
        """Get all evolution requirements."""
        return self.evolution_requirements.copy()
    
    def get_evolution_timing(self, stage: LifeStage) -> float:
        """Get the evolution timing for a specific stage."""
        return self.EVOLUTION_TIMINGS.get(stage, float('inf'))
    
    def get_all_evolution_timings(self) -> Dict[LifeStage, float]:
        """Get all evolution timings."""
        return self.EVOLUTION_TIMINGS.copy()