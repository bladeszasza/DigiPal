"""
Unit tests for AttributeEngine and care mechanics.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from digipal.core import (
    DigiPal, AttributeEngine, AttributeType, CareActionType, 
    LifeStage, EggType, InteractionResult
)


class TestAttributeEngine:
    """Test cases for AttributeEngine functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = AttributeEngine()
        self.pet = DigiPal(
            user_id="test_user",
            name="TestPal",
            egg_type=EggType.RED,
            life_stage=LifeStage.CHILD,
            hp=100,
            mp=50,
            offense=15,
            defense=10,
            speed=12,
            brains=8,
            discipline=50,
            happiness=60,
            weight=25,
            energy=80,
            care_mistakes=0
        )
    
    def test_initialization(self):
        """Test AttributeEngine initialization."""
        assert self.engine is not None
        assert len(self.engine.care_actions) > 0
        assert "strength_training" in self.engine.care_actions
        assert "praise" in self.engine.care_actions
        assert "rest" in self.engine.care_actions
    
    def test_attribute_bounds(self):
        """Test attribute bounds validation."""
        # Test HP bounds
        hp_min, hp_max = self.engine.get_attribute_bounds(AttributeType.HP)
        assert hp_min == 1
        assert hp_max == 999
        
        # Test discipline bounds
        disc_min, disc_max = self.engine.get_attribute_bounds(AttributeType.DISCIPLINE)
        assert disc_min == 0
        assert disc_max == 100
        
        # Test weight bounds
        weight_min, weight_max = self.engine.get_attribute_bounds(AttributeType.WEIGHT)
        assert weight_min == 1
        assert weight_max == 99
    
    def test_validate_attribute_value(self):
        """Test attribute value validation and clamping."""
        # Test normal value
        assert self.engine.validate_attribute_value(AttributeType.HP, 50) == 50
        
        # Test value below minimum
        assert self.engine.validate_attribute_value(AttributeType.HP, -10) == 1
        
        # Test value above maximum
        assert self.engine.validate_attribute_value(AttributeType.HP, 1500) == 999
        
        # Test discipline bounds
        assert self.engine.validate_attribute_value(AttributeType.DISCIPLINE, -5) == 0
        assert self.engine.validate_attribute_value(AttributeType.DISCIPLINE, 150) == 100


class TestCareActions:
    """Test cases for care action functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = AttributeEngine()
        self.pet = DigiPal(
            user_id="test_user",
            name="TestPal",
            egg_type=EggType.RED,
            life_stage=LifeStage.CHILD,
            hp=100,
            mp=50,
            offense=15,
            defense=10,
            speed=12,
            brains=8,
            discipline=50,
            happiness=60,
            weight=25,
            energy=80,
            care_mistakes=0
        )
    
    def test_strength_training_success(self):
        """Test successful strength training."""
        initial_offense = self.pet.offense
        initial_hp = self.pet.hp
        initial_energy = self.pet.energy
        initial_happiness = self.pet.happiness
        initial_weight = self.pet.weight
        
        success, interaction = self.engine.apply_care_action(self.pet, "strength_training")
        
        assert success is True
        assert interaction.success is True
        assert interaction.result == InteractionResult.SUCCESS
        assert self.pet.offense == initial_offense + 3
        assert self.pet.hp == initial_hp + 2
        assert self.pet.energy == initial_energy - 15
        assert self.pet.happiness == initial_happiness - 5
        assert self.pet.weight == initial_weight - 1
        assert len(self.pet.conversation_history) == 1
    
    def test_strength_training_insufficient_energy(self):
        """Test strength training with insufficient energy."""
        self.pet.energy = 10  # Below required 15
        initial_care_mistakes = self.pet.care_mistakes
        
        success, interaction = self.engine.apply_care_action(self.pet, "strength_training")
        
        assert success is False
        assert interaction.success is False
        assert interaction.result == InteractionResult.INSUFFICIENT_ENERGY
        assert self.pet.care_mistakes == initial_care_mistakes + 1
    
    def test_defense_training(self):
        """Test defense training action."""
        initial_defense = self.pet.defense
        initial_hp = self.pet.hp
        
        success, interaction = self.engine.apply_care_action(self.pet, "defense_training")
        
        assert success is True
        assert self.pet.defense == initial_defense + 3
        assert self.pet.hp == initial_hp + 2
        assert self.pet.energy == 80 - 15  # Energy cost
        assert self.pet.happiness == 60 - 5  # Happiness cost
    
    def test_speed_training(self):
        """Test speed training action."""
        initial_speed = self.pet.speed
        initial_weight = self.pet.weight
        
        success, interaction = self.engine.apply_care_action(self.pet, "speed_training")
        
        assert success is True
        assert self.pet.speed == initial_speed + 3
        assert self.pet.weight == initial_weight - 2
        assert self.pet.energy == 80 - 12  # Energy cost
    
    def test_brain_training(self):
        """Test brain training action."""
        initial_brains = self.pet.brains
        initial_mp = self.pet.mp
        
        success, interaction = self.engine.apply_care_action(self.pet, "brain_training")
        
        assert success is True
        assert self.pet.brains == initial_brains + 3
        assert self.pet.mp == initial_mp + 2
        assert self.pet.energy == 80 - 10  # Energy cost
    
    def test_feed_meat(self):
        """Test feeding meat action."""
        initial_weight = self.pet.weight
        initial_hp = self.pet.hp
        initial_offense = self.pet.offense
        initial_happiness = self.pet.happiness
        
        success, interaction = self.engine.apply_care_action(self.pet, "meat")
        
        assert success is True
        assert self.pet.weight == initial_weight + 2
        assert self.pet.hp == initial_hp + 1
        assert self.pet.offense == initial_offense + 1
        assert self.pet.happiness == initial_happiness + 5
        assert self.pet.energy == 80  # No energy cost for feeding
    
    def test_feed_fish(self):
        """Test feeding fish action."""
        initial_weight = self.pet.weight
        initial_brains = self.pet.brains
        initial_mp = self.pet.mp
        
        success, interaction = self.engine.apply_care_action(self.pet, "fish")
        
        assert success is True
        assert self.pet.weight == initial_weight + 1
        assert self.pet.brains == initial_brains + 1
        assert self.pet.mp == initial_mp + 1
        assert self.pet.happiness == 60 + 3
    
    def test_feed_vegetables(self):
        """Test feeding vegetables action."""
        initial_weight = self.pet.weight
        initial_defense = self.pet.defense
        
        success, interaction = self.engine.apply_care_action(self.pet, "vegetables")
        
        assert success is True
        assert self.pet.weight == initial_weight + 1
        assert self.pet.defense == initial_defense + 1
        assert self.pet.happiness == 60 + 2
    
    def test_praise_action(self):
        """Test praise care action."""
        initial_happiness = self.pet.happiness
        initial_discipline = self.pet.discipline
        
        success, interaction = self.engine.apply_care_action(self.pet, "praise")
        
        assert success is True
        assert self.pet.happiness == initial_happiness + 10
        assert self.pet.discipline == initial_discipline - 2
        assert self.pet.energy == 80  # No energy cost
    
    def test_scold_action(self):
        """Test scold care action."""
        initial_happiness = self.pet.happiness
        initial_discipline = self.pet.discipline
        
        success, interaction = self.engine.apply_care_action(self.pet, "scold")
        
        assert success is True
        assert self.pet.happiness == initial_happiness - 8
        assert self.pet.discipline == initial_discipline + 5
    
    def test_rest_action(self):
        """Test rest action restores energy."""
        self.pet.energy = 30
        initial_happiness = self.pet.happiness
        
        success, interaction = self.engine.apply_care_action(self.pet, "rest")
        
        assert success is True
        assert self.pet.energy == 30 + 30  # Rest restores 30 energy
        assert self.pet.happiness == initial_happiness + 3
    
    def test_play_action(self):
        """Test play action."""
        initial_happiness = self.pet.happiness
        initial_weight = self.pet.weight
        
        success, interaction = self.engine.apply_care_action(self.pet, "play")
        
        assert success is True
        assert self.pet.happiness == initial_happiness + 8
        assert self.pet.weight == initial_weight - 1
        assert self.pet.energy == 80 - 8  # Energy cost
    
    def test_play_insufficient_energy(self):
        """Test play action with insufficient energy."""
        self.pet.energy = 5  # Below required 8
        initial_care_mistakes = self.pet.care_mistakes
        
        success, interaction = self.engine.apply_care_action(self.pet, "play")
        
        assert success is False
        assert self.pet.care_mistakes == initial_care_mistakes + 1
    
    def test_invalid_action(self):
        """Test applying invalid care action."""
        success, interaction = self.engine.apply_care_action(self.pet, "invalid_action")
        
        assert success is False
        assert interaction.result == InteractionResult.INVALID_COMMAND
        assert "Unknown care action" in interaction.pet_response


class TestTimeDecay:
    """Test cases for time-based attribute decay."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = AttributeEngine()
        self.pet = DigiPal(
            user_id="test_user",
            name="TestPal",
            egg_type=EggType.RED,
            life_stage=LifeStage.CHILD,
            energy=80,
            happiness=60,
            weight=25
        )
    
    def test_energy_decay(self):
        """Test energy decay over time."""
        initial_energy = self.pet.energy
        hours_passed = 5.0
        
        changes = self.engine.apply_time_decay(self.pet, hours_passed)
        
        expected_decay = int(hours_passed * self.engine.ENERGY_DECAY_RATE)
        assert changes["energy"] == -expected_decay
        assert self.pet.energy == initial_energy - expected_decay
    
    def test_happiness_decay(self):
        """Test happiness decay over time."""
        initial_happiness = self.pet.happiness
        hours_passed = 3.0
        
        changes = self.engine.apply_time_decay(self.pet, hours_passed)
        
        expected_decay = int(hours_passed * self.engine.HAPPINESS_DECAY_RATE)
        assert changes["happiness"] == -expected_decay
        assert self.pet.happiness == initial_happiness - expected_decay
    
    def test_weight_loss_low_energy(self):
        """Test weight loss when energy is very low."""
        self.pet.energy = 15  # Very low energy
        initial_weight = self.pet.weight
        hours_passed = 4.0
        
        changes = self.engine.apply_time_decay(self.pet, hours_passed)
        
        assert "weight" in changes
        assert changes["weight"] < 0  # Weight should decrease
        assert self.pet.weight < initial_weight
    
    def test_no_weight_change_normal_energy(self):
        """Test no weight change when energy is normal."""
        self.pet.energy = 50  # Normal energy
        hours_passed = 2.0
        
        changes = self.engine.apply_time_decay(self.pet, hours_passed)
        
        assert "weight" not in changes or changes.get("weight", 0) == 0
    
    def test_attribute_bounds_respected_in_decay(self):
        """Test that attribute bounds are respected during decay."""
        self.pet.energy = 5  # Very low
        self.pet.happiness = 2  # Very low
        hours_passed = 10.0  # Long time
        
        changes = self.engine.apply_time_decay(self.pet, hours_passed)
        
        # Energy should not go below 0
        assert self.pet.energy >= 0
        # Happiness should not go below 0
        assert self.pet.happiness >= 0


class TestCareMistakes:
    """Test cases for care mistake tracking."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = AttributeEngine()
        self.pet = DigiPal(
            user_id="test_user",
            name="TestPal",
            egg_type=EggType.RED,
            life_stage=LifeStage.CHILD,
            energy=80,
            happiness=60,
            weight=25,
            care_mistakes=0
        )
    
    @patch('random.random')
    def test_care_mistake_calculation_low_energy(self, mock_random):
        """Test care mistake probability with low energy."""
        self.pet.energy = 15  # Low energy
        mock_random.return_value = 0.2  # Below threshold
        
        mistake = self.engine.calculate_care_mistake(self.pet, CareActionType.TRAIN)
        assert mistake is True
    
    @patch('random.random')
    def test_care_mistake_calculation_low_happiness(self, mock_random):
        """Test care mistake probability with low happiness."""
        self.pet.happiness = 15  # Low happiness
        mock_random.return_value = 0.15  # Below threshold
        
        mistake = self.engine.calculate_care_mistake(self.pet, CareActionType.TRAIN)
        assert mistake is True
    
    @patch('random.random')
    def test_care_mistake_calculation_extreme_weight(self, mock_random):
        """Test care mistake probability with extreme weight."""
        self.pet.weight = 5  # Very low weight
        mock_random.return_value = 0.15  # Below threshold
        
        mistake = self.engine.calculate_care_mistake(self.pet, CareActionType.FEED)
        assert mistake is True
    
    @patch('random.random')
    def test_care_mistake_training_when_tired(self, mock_random):
        """Test higher mistake probability when training while tired."""
        self.pet.energy = 25  # Low energy for training
        mock_random.return_value = 0.3  # Below combined threshold
        
        mistake = self.engine.calculate_care_mistake(self.pet, CareActionType.TRAIN)
        assert mistake is True
    
    @patch('random.random')
    def test_no_care_mistake_good_condition(self, mock_random):
        """Test no care mistake when pet is in good condition."""
        # Pet is already in good condition from setup
        mock_random.return_value = 0.8  # Above threshold
        
        mistake = self.engine.calculate_care_mistake(self.pet, CareActionType.TRAIN)
        assert mistake is False


class TestAvailableActions:
    """Test cases for available actions based on pet state."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = AttributeEngine()
        self.pet = DigiPal(
            user_id="test_user",
            name="TestPal",
            egg_type=EggType.RED,
            life_stage=LifeStage.CHILD,
            energy=80
        )
    
    def test_available_actions_normal_energy(self):
        """Test available actions with normal energy."""
        actions = self.engine.get_available_actions(self.pet)
        
        assert "strength_training" in actions
        assert "defense_training" in actions
        assert "speed_training" in actions
        assert "brain_training" in actions
        assert "play" in actions
        assert "meat" in actions
        assert "praise" in actions
        assert "rest" in actions
    
    def test_available_actions_low_energy(self):
        """Test available actions with low energy."""
        self.pet.energy = 5  # Very low energy
        actions = self.engine.get_available_actions(self.pet)
        
        # Training actions should not be available
        assert "strength_training" not in actions
        assert "defense_training" not in actions
        assert "speed_training" not in actions
        assert "play" not in actions
        
        # Basic care should still be available
        assert "meat" in actions
        assert "praise" in actions
        assert "rest" in actions
    
    def test_available_actions_baby_stage(self):
        """Test available actions for baby stage."""
        self.pet.life_stage = LifeStage.BABY
        actions = self.engine.get_available_actions(self.pet)
        
        # Training and play should not be available for babies
        assert "strength_training" not in actions
        assert "play" not in actions
        
        # Basic care should be available
        assert "meat" in actions
        assert "praise" in actions
        assert "rest" in actions
    
    def test_get_care_action(self):
        """Test getting specific care action."""
        action = self.engine.get_care_action("strength_training")
        assert action is not None
        assert action.name == "Strength Training"
        assert action.action_type == CareActionType.TRAIN
        
        # Test invalid action
        invalid_action = self.engine.get_care_action("invalid")
        assert invalid_action is None
    
    def test_get_all_care_actions(self):
        """Test getting all care actions."""
        all_actions = self.engine.get_all_care_actions()
        
        assert len(all_actions) > 0
        assert "strength_training" in all_actions
        assert "praise" in all_actions
        assert "rest" in all_actions
        
        # Ensure it's a copy, not the original
        all_actions["test"] = None
        assert "test" not in self.engine.care_actions


class TestAttributeBounds:
    """Test cases for attribute bounds enforcement."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = AttributeEngine()
        self.pet = DigiPal(
            user_id="test_user",
            name="TestPal",
            egg_type=EggType.RED,
            life_stage=LifeStage.CHILD
        )
    
    def test_hp_minimum_bound(self):
        """Test HP cannot go below 1."""
        self.pet.hp = 2
        # Try to reduce HP by more than current value
        self.pet.modify_attribute(AttributeType.HP, -10)
        assert self.pet.hp == 1  # Should be clamped to minimum
    
    def test_discipline_bounds(self):
        """Test discipline bounds (0-100)."""
        # Test upper bound
        self.pet.discipline = 95
        self.pet.modify_attribute(AttributeType.DISCIPLINE, 10)
        assert self.pet.discipline == 100
        
        # Test lower bound
        self.pet.discipline = 5
        self.pet.modify_attribute(AttributeType.DISCIPLINE, -10)
        assert self.pet.discipline == 0
    
    def test_weight_bounds(self):
        """Test weight bounds (1-99)."""
        # Test upper bound
        self.pet.weight = 95
        self.pet.modify_attribute(AttributeType.WEIGHT, 10)
        assert self.pet.weight == 99
        
        # Test lower bound
        self.pet.weight = 3
        self.pet.modify_attribute(AttributeType.WEIGHT, -5)
        assert self.pet.weight == 1
    
    def test_energy_bounds(self):
        """Test energy bounds (0-100)."""
        # Test upper bound
        self.pet.energy = 95
        self.pet.modify_attribute(AttributeType.ENERGY, 10)
        assert self.pet.energy == 100
        
        # Test lower bound
        self.pet.energy = 5
        self.pet.modify_attribute(AttributeType.ENERGY, -10)
        assert self.pet.energy == 0


if __name__ == "__main__":
    pytest.main([__file__])


class TestNewCareActions:
    """Test cases for new care actions added to the engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = AttributeEngine()
        self.pet = DigiPal(
            user_id="test_user",
            name="TestPal",
            egg_type=EggType.RED,
            life_stage=LifeStage.CHILD,
            hp=100,
            mp=50,
            offense=15,
            defense=10,
            speed=12,
            brains=8,
            discipline=50,
            happiness=60,
            weight=25,
            energy=80,
            care_mistakes=0
        )
    
    def test_protein_shake_feeding(self):
        """Test feeding protein shake action."""
        initial_weight = self.pet.weight
        initial_offense = self.pet.offense
        initial_hp = self.pet.hp
        initial_happiness = self.pet.happiness
        
        success, interaction = self.engine.apply_care_action(self.pet, "protein_shake")
        
        assert success is True
        assert self.pet.weight == initial_weight + 1
        assert self.pet.offense == initial_offense + 2
        assert self.pet.hp == initial_hp + 1
        assert self.pet.happiness == initial_happiness + 1
    
    def test_energy_drink_feeding(self):
        """Test feeding energy drink action."""
        initial_energy = self.pet.energy
        initial_speed = self.pet.speed
        initial_mp = self.pet.mp
        initial_happiness = self.pet.happiness
        
        success, interaction = self.engine.apply_care_action(self.pet, "energy_drink")
        
        assert success is True
        assert self.pet.energy == initial_energy + 5  # Energy drink restores energy
        assert self.pet.speed == initial_speed + 1
        assert self.pet.mp == initial_mp + 1
        assert self.pet.happiness == initial_happiness + 3
    
    def test_endurance_training(self):
        """Test endurance training action."""
        initial_hp = self.pet.hp
        initial_defense = self.pet.defense
        initial_weight = self.pet.weight
        initial_energy = self.pet.energy
        initial_happiness = self.pet.happiness
        
        success, interaction = self.engine.apply_care_action(self.pet, "endurance_training")
        
        assert success is True
        assert self.pet.hp == initial_hp + 4
        assert self.pet.defense == initial_defense + 2
        assert self.pet.weight == initial_weight - 2
        assert self.pet.energy == initial_energy - 20
        assert self.pet.happiness == initial_happiness - 8
    
    def test_endurance_training_insufficient_energy(self):
        """Test endurance training with insufficient energy."""
        self.pet.energy = 15  # Below required 20
        initial_care_mistakes = self.pet.care_mistakes
        
        success, interaction = self.engine.apply_care_action(self.pet, "endurance_training")
        
        assert success is False
        assert self.pet.care_mistakes == initial_care_mistakes + 1
    
    def test_agility_training(self):
        """Test agility training action."""
        initial_speed = self.pet.speed
        initial_offense = self.pet.offense
        initial_weight = self.pet.weight
        initial_energy = self.pet.energy
        initial_happiness = self.pet.happiness
        
        success, interaction = self.engine.apply_care_action(self.pet, "agility_training")
        
        assert success is True
        assert self.pet.speed == initial_speed + 4
        assert self.pet.offense == initial_offense + 1
        assert self.pet.weight == initial_weight - 3
        assert self.pet.energy == initial_energy - 18
        assert self.pet.happiness == initial_happiness - 6
    
    def test_agility_training_insufficient_energy(self):
        """Test agility training with insufficient energy."""
        self.pet.energy = 10  # Below required 18
        initial_care_mistakes = self.pet.care_mistakes
        
        success, interaction = self.engine.apply_care_action(self.pet, "agility_training")
        
        assert success is False
        assert self.pet.care_mistakes == initial_care_mistakes + 1


class TestEnhancedCareMistakes:
    """Test cases for enhanced care mistake system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = AttributeEngine()
        self.pet = DigiPal(
            user_id="test_user",
            name="TestPal",
            egg_type=EggType.RED,
            life_stage=LifeStage.CHILD,
            energy=80,
            happiness=60,
            weight=25,
            discipline=50,
            care_mistakes=0
        )
    
    @patch('random.random')
    def test_care_mistake_overfeeding(self, mock_random):
        """Test care mistake probability when overfeeding."""
        self.pet.weight = 75  # Overweight
        mock_random.return_value = 0.2  # Below threshold
        
        mistake = self.engine.calculate_care_mistake(self.pet, CareActionType.FEED)
        assert mistake is True
    
    @patch('random.random')
    def test_care_mistake_excessive_scolding(self, mock_random):
        """Test care mistake probability with excessive scolding."""
        self.pet.discipline = 85  # Over-disciplined
        mock_random.return_value = 0.2  # Below threshold
        
        mistake = self.engine.calculate_care_mistake(self.pet, CareActionType.SCOLD)
        assert mistake is True
    
    @patch('random.random')
    def test_no_care_mistake_normal_feeding(self, mock_random):
        """Test no care mistake when feeding normally."""
        self.pet.weight = 30  # Normal weight
        mock_random.return_value = 0.8  # Above threshold
        
        mistake = self.engine.calculate_care_mistake(self.pet, CareActionType.FEED)
        assert mistake is False


class TestCareQualityAssessment:
    """Test cases for care quality assessment system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = AttributeEngine()
        self.pet = DigiPal(
            user_id="test_user",
            name="TestPal",
            egg_type=EggType.RED,
            life_stage=LifeStage.CHILD,
            energy=80,
            happiness=60,
            weight=25,
            discipline=50,
            care_mistakes=0
        )
    
    def test_excellent_care_assessment(self):
        """Test assessment with excellent care."""
        self.pet.energy = 90
        self.pet.happiness = 85
        self.pet.weight = 25
        self.pet.discipline = 60
        self.pet.care_mistakes = 0
        
        assessment = self.engine.get_care_quality_assessment(self.pet)
        
        assert assessment["energy"] == "excellent"
        assert assessment["happiness"] == "very_happy"
        assert assessment["weight"] == "healthy"
        assert assessment["discipline"] == "balanced"
        assert assessment["care_quality"] == "perfect"
    
    def test_poor_care_assessment(self):
        """Test assessment with poor care."""
        self.pet.energy = 10
        self.pet.happiness = 15
        self.pet.weight = 5
        self.pet.discipline = 20
        self.pet.care_mistakes = 20
        
        assessment = self.engine.get_care_quality_assessment(self.pet)
        
        assert assessment["energy"] == "critical"
        assert assessment["happiness"] == "very_sad"
        assert assessment["weight"] == "concerning"
        assert assessment["discipline"] == "undisciplined"
        assert assessment["care_quality"] == "poor"
    
    def test_mixed_care_assessment(self):
        """Test assessment with mixed care quality."""
        self.pet.energy = 65
        self.pet.happiness = 45
        self.pet.weight = 40
        self.pet.discipline = 85
        self.pet.care_mistakes = 5
        
        assessment = self.engine.get_care_quality_assessment(self.pet)
        
        assert assessment["energy"] == "good"
        assert assessment["happiness"] == "neutral"
        assert assessment["weight"] == "slightly_off"
        assert assessment["discipline"] == "over_disciplined"
        assert assessment["care_quality"] == "good"
    
    def test_weight_assessment_extremes(self):
        """Test weight assessment at extreme values."""
        # Test very low weight (unhealthy range)
        self.pet.weight = 3
        assessment = self.engine.get_care_quality_assessment(self.pet)
        assert assessment["weight"] == "unhealthy"
        
        # Test concerning weight range
        self.pet.weight = 7
        assessment = self.engine.get_care_quality_assessment(self.pet)
        assert assessment["weight"] == "concerning"
        
        # Test very high weight
        self.pet.weight = 80
        assessment = self.engine.get_care_quality_assessment(self.pet)
        assert assessment["weight"] == "unhealthy"
        
        # Test healthy weight range
        self.pet.weight = 20
        assessment = self.engine.get_care_quality_assessment(self.pet)
        assert assessment["weight"] == "healthy"


class TestNewActionsAvailability:
    """Test cases for availability of new care actions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = AttributeEngine()
        self.pet = DigiPal(
            user_id="test_user",
            name="TestPal",
            egg_type=EggType.RED,
            life_stage=LifeStage.CHILD,
            energy=80
        )
    
    def test_new_actions_available_normal_energy(self):
        """Test new actions are available with normal energy."""
        actions = self.engine.get_available_actions(self.pet)
        
        assert "protein_shake" in actions
        assert "energy_drink" in actions
        assert "endurance_training" in actions
        assert "agility_training" in actions
    
    def test_new_training_actions_unavailable_low_energy(self):
        """Test new training actions unavailable with low energy."""
        self.pet.energy = 15  # Below requirements for advanced training
        actions = self.engine.get_available_actions(self.pet)
        
        # Advanced training should not be available
        assert "endurance_training" not in actions
        assert "agility_training" not in actions
        
        # Feeding should still be available
        assert "protein_shake" in actions
        assert "energy_drink" in actions
    
    def test_all_new_actions_exist(self):
        """Test all new actions are properly registered."""
        all_actions = self.engine.get_all_care_actions()
        
        assert "protein_shake" in all_actions
        assert "energy_drink" in all_actions
        assert "endurance_training" in all_actions
        assert "agility_training" in all_actions
        
        # Verify they have correct properties
        protein_shake = all_actions["protein_shake"]
        assert protein_shake.action_type == CareActionType.FEED
        
        endurance_training = all_actions["endurance_training"]
        assert endurance_training.action_type == CareActionType.TRAIN
        assert endurance_training.energy_cost == 20