"""
Core data models for DigiPal application.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Any
import json
import uuid

from .enums import EggType, LifeStage, CareActionType, AttributeType, CommandType, InteractionResult


@dataclass
class DigiPal:
    """
    Core DigiPal model representing a digital pet with all attributes and lifecycle properties.
    """
    # Identity and Basic Info
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    name: str = "DigiPal"
    egg_type: EggType = EggType.RED
    life_stage: LifeStage = LifeStage.EGG
    generation: int = 1
    
    # Primary Attributes (Digimon World 1 inspired)
    hp: int = 100
    mp: int = 50
    offense: int = 10
    defense: int = 10
    speed: int = 10
    brains: int = 10
    
    # Secondary Attributes
    discipline: int = 0
    happiness: int = 50
    weight: int = 20
    care_mistakes: int = 0
    energy: int = 100
    
    # Lifecycle Management
    birth_time: datetime = field(default_factory=datetime.now)
    last_interaction: datetime = field(default_factory=datetime.now)
    evolution_timer: float = 0.0  # Hours until next evolution check
    
    # Memory and Context
    conversation_history: List['Interaction'] = field(default_factory=list)
    learned_commands: Set[str] = field(default_factory=set)
    personality_traits: Dict[str, float] = field(default_factory=dict)
    
    # Visual Representation
    current_image_path: str = ""
    image_generation_prompt: str = ""
    
    def __post_init__(self):
        """Initialize DigiPal with egg-type specific attributes."""
        if self.life_stage == LifeStage.EGG:
            self._initialize_egg_attributes()
        
        # Initialize basic learned commands for baby stage
        if self.life_stage == LifeStage.BABY:
            self.learned_commands = {"eat", "sleep", "good", "bad"}
    
    def _initialize_egg_attributes(self):
        """Set initial attributes based on egg type."""
        base_attributes = {
            EggType.RED: {
                'offense': 15,
                'defense': 8,
                'speed': 12,
                'brains': 8,
                'hp': 90,
                'mp': 40
            },
            EggType.BLUE: {
                'offense': 8,
                'defense': 15,
                'speed': 8,
                'brains': 12,
                'hp': 110,
                'mp': 60
            },
            EggType.GREEN: {
                'offense': 10,
                'defense': 12,
                'speed': 10,
                'brains': 10,
                'hp': 120,
                'mp': 50
            }
        }
        
        if self.egg_type in base_attributes:
            attrs = base_attributes[self.egg_type]
            for attr, value in attrs.items():
                setattr(self, attr, value)
    
    def get_age_hours(self) -> float:
        """Calculate age in hours since birth."""
        return (datetime.now() - self.birth_time).total_seconds() / 3600
    
    def get_attribute(self, attribute: AttributeType) -> int:
        """Get attribute value by type."""
        return getattr(self, attribute.value, 0)
    
    def set_attribute(self, attribute: AttributeType, value: int):
        """Set attribute value with bounds checking."""
        # Define attribute bounds
        bounds = {
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
        
        min_val, max_val = bounds.get(attribute, (0, 999))
        clamped_value = max(min_val, min(max_val, value))
        setattr(self, attribute.value, clamped_value)
    
    def modify_attribute(self, attribute: AttributeType, change: int):
        """Modify attribute by a delta amount."""
        current_value = self.get_attribute(attribute)
        self.set_attribute(attribute, current_value + change)
    
    def can_understand_command(self, command: str) -> bool:
        """Check if DigiPal can understand a command based on life stage."""
        stage_commands = {
            LifeStage.EGG: set(),
            LifeStage.BABY: {"eat", "sleep", "good", "bad"},
            LifeStage.CHILD: {"eat", "sleep", "good", "bad", "play", "train"},
            LifeStage.TEEN: {"eat", "sleep", "good", "bad", "play", "train", "status", "talk"},
            LifeStage.YOUNG_ADULT: {"eat", "sleep", "good", "bad", "play", "train", "status", "talk", "battle"},
            LifeStage.ADULT: {"eat", "sleep", "good", "bad", "play", "train", "status", "talk", "battle", "teach"},
            LifeStage.ELDERLY: {"eat", "sleep", "good", "bad", "play", "train", "status", "talk", "battle", "teach", "wisdom"}
        }
        
        available_commands = stage_commands.get(self.life_stage, set())
        return command.lower() in available_commands or command.lower() in self.learned_commands
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert DigiPal to dictionary for serialization."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'name': self.name,
            'egg_type': self.egg_type.value,
            'life_stage': self.life_stage.value,
            'generation': self.generation,
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
            'energy': self.energy,
            'birth_time': self.birth_time.isoformat(),
            'last_interaction': self.last_interaction.isoformat(),
            'evolution_timer': self.evolution_timer,
            'conversation_history': [interaction.to_dict() for interaction in self.conversation_history],
            'learned_commands': list(self.learned_commands),
            'personality_traits': self.personality_traits,
            'current_image_path': self.current_image_path,
            'image_generation_prompt': self.image_generation_prompt
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DigiPal':
        """Create DigiPal from dictionary."""
        # Convert string enums back to enum objects
        data['egg_type'] = EggType(data['egg_type'])
        data['life_stage'] = LifeStage(data['life_stage'])
        
        # Convert ISO strings back to datetime objects
        data['birth_time'] = datetime.fromisoformat(data['birth_time'])
        data['last_interaction'] = datetime.fromisoformat(data['last_interaction'])
        
        # Convert conversation history
        data['conversation_history'] = [
            Interaction.from_dict(interaction_data) 
            for interaction_data in data.get('conversation_history', [])
        ]
        
        # Convert learned commands to set
        data['learned_commands'] = set(data.get('learned_commands', []))
        
        return cls(**data)


@dataclass
class Interaction:
    """Represents a single interaction between user and DigiPal."""
    timestamp: datetime = field(default_factory=datetime.now)
    user_input: str = ""
    interpreted_command: str = ""
    pet_response: str = ""
    attribute_changes: Dict[str, int] = field(default_factory=dict)
    success: bool = True
    result: InteractionResult = InteractionResult.SUCCESS
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Interaction to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'user_input': self.user_input,
            'interpreted_command': self.interpreted_command,
            'pet_response': self.pet_response,
            'attribute_changes': self.attribute_changes,
            'success': self.success,
            'result': self.result.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Interaction':
        """Create Interaction from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['result'] = InteractionResult(data['result'])
        return cls(**data)


@dataclass
class Command:
    """Represents a parsed command from user input."""
    action: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    stage_appropriate: bool = True
    energy_required: int = 0
    command_type: CommandType = CommandType.UNKNOWN
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Command to dictionary."""
        return {
            'action': self.action,
            'parameters': self.parameters,
            'stage_appropriate': self.stage_appropriate,
            'energy_required': self.energy_required,
            'command_type': self.command_type.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Command':
        """Create Command from dictionary."""
        data['command_type'] = CommandType(data['command_type'])
        return cls(**data)


@dataclass
class AttributeModifier:
    """Represents a modification to DigiPal attributes."""
    attribute: AttributeType
    change: int
    conditions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert AttributeModifier to dictionary."""
        return {
            'attribute': self.attribute.value,
            'change': self.change,
            'conditions': self.conditions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AttributeModifier':
        """Create AttributeModifier from dictionary."""
        data['attribute'] = AttributeType(data['attribute'])
        return cls(**data)


@dataclass
class CareAction:
    """Represents a care action that can be performed on DigiPal."""
    name: str
    action_type: CareActionType
    energy_cost: int
    happiness_change: int
    attribute_modifiers: List[AttributeModifier] = field(default_factory=list)
    success_conditions: List[str] = field(default_factory=list)
    failure_effects: List[AttributeModifier] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert CareAction to dictionary."""
        return {
            'name': self.name,
            'action_type': self.action_type.value,
            'energy_cost': self.energy_cost,
            'happiness_change': self.happiness_change,
            'attribute_modifiers': [mod.to_dict() for mod in self.attribute_modifiers],
            'success_conditions': self.success_conditions,
            'failure_effects': [mod.to_dict() for mod in self.failure_effects]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CareAction':
        """Create CareAction from dictionary."""
        data['action_type'] = CareActionType(data['action_type'])
        data['attribute_modifiers'] = [
            AttributeModifier.from_dict(mod_data) 
            for mod_data in data.get('attribute_modifiers', [])
        ]
        data['failure_effects'] = [
            AttributeModifier.from_dict(mod_data) 
            for mod_data in data.get('failure_effects', [])
        ]
        return cls(**data)