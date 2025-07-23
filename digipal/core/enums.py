"""
Enum classes for DigiPal constants and types.
"""

from enum import Enum, auto


class EggType(Enum):
    """Types of eggs that determine initial DigiPal attributes."""
    RED = "red"      # Fire-oriented, higher attack
    BLUE = "blue"    # Water-oriented, higher defense  
    GREEN = "green"  # Earth-oriented, higher health and symbiosis


class LifeStage(Enum):
    """Life stages that DigiPal progresses through."""
    EGG = "egg"
    BABY = "baby"
    CHILD = "child"
    TEEN = "teen"
    YOUNG_ADULT = "young_adult"
    ADULT = "adult"
    ELDERLY = "elderly"


class CareActionType(Enum):
    """Types of care actions that can be performed on DigiPal."""
    TRAIN = "train"
    FEED = "feed"
    PRAISE = "praise"
    SCOLD = "scold"
    REST = "rest"
    PLAY = "play"


class AttributeType(Enum):
    """Primary and secondary attribute types."""
    # Primary Attributes (Digimon World 1 inspired)
    HP = "hp"
    MP = "mp"
    OFFENSE = "offense"
    DEFENSE = "defense"
    SPEED = "speed"
    BRAINS = "brains"
    
    # Secondary Attributes
    DISCIPLINE = "discipline"
    HAPPINESS = "happiness"
    WEIGHT = "weight"
    CARE_MISTAKES = "care_mistakes"
    ENERGY = "energy"


class CommandType(Enum):
    """Types of commands DigiPal can understand."""
    EAT = "eat"
    SLEEP = "sleep"
    GOOD = "good"
    BAD = "bad"
    TRAIN = "train"
    PLAY = "play"
    STATUS = "status"
    UNKNOWN = "unknown"


class InteractionResult(Enum):
    """Results of user interactions with DigiPal."""
    SUCCESS = "success"
    FAILURE = "failure"
    INVALID_COMMAND = "invalid_command"
    INSUFFICIENT_ENERGY = "insufficient_energy"
    STAGE_INAPPROPRIATE = "stage_inappropriate"