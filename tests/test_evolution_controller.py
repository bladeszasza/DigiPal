"""
Tests for EvolutionController - Evolution and lifecycle management system.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from digipal.core.evolution_controller import (
    EvolutionController, EvolutionRequirement, EvolutionResult, DNAInheritance
)
from digipal.core.models import DigiPal
from digipal.core.enums import LifeStage, EggType, AttributeType


class TestEvolutionController:
    """Test cases for EvolutionController."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.controller = EvolutionController()
        self.pet = DigiPal(
            user_id="test_user",
            name="TestPal",
            egg_type=EggType.RED,
            life_stage=LifeStage.BABY
        )
    
    def test_initialization(self):
        """Test EvolutionController initialization."""
        assert self.controller is not None
        assert len(self.controller.evolution_requirements) > 0
        assert LifeStage.BABY in self.controller.evolution_requirements
        assert LifeStage.CHILD in self.controller.evolution_requirements
    
    def test_get_next_life_stage(self):
        """Test getting the next life stage in progression."""
        # Test normal progression
        assert self.controller._get_next_life_stage(LifeStage.EGG) == LifeStage.BABY
        assert self.controller._get_next_life_stage(LifeStage.BABY) == LifeStage.CHILD
        assert self.controller._get_next_life_stage(LifeStage.CHILD) == LifeStage.TEEN
        assert self.controller._get_next_life_stage(LifeStage.TEEN) == LifeStage.YOUNG_ADULT
        assert self.controller._get_next_life_stage(LifeStage.YOUNG_ADULT) == LifeStage.ADULT
        assert self.controller._get_next_life_stage(LifeStage.ADULT) == LifeStage.ELDERLY
        
        # Test end of progression
        assert self.controller._get_next_life_stage(LifeStage.ELDERLY) is None
    
    def test_evolution_requirements_structure(self):
        """Test that evolution requirements are properly structured."""
        for stage, requirement in self.controller.evolution_requirements.items():
            assert isinstance(requirement, EvolutionRequirement)
            assert requirement.min_age_hours >= 0
            assert requirement.max_age_hours > requirement.min_age_hours
            assert 0 <= requirement.happiness_threshold <= 100
            assert requirement.max_care_mistakes >= 0
    
    def test_check_evolution_eligibility_baby_to_child(self):
        """Test evolution eligibility from baby to child."""
        # Set up pet that meets child requirements
        self.pet.life_stage = LifeStage.BABY
        self.pet.birth_time = datetime.now() - timedelta(hours=25)  # Old enough
        self.pet.care_mistakes = 3  # Within limit
        self.pet.happiness = 40  # Above threshold
        self.pet.hp = 90  # Above minimum
        self.pet.energy = 30  # Above minimum
        
        eligible, next_stage, status = self.controller.check_evolution_eligibility(self.pet)
        
        assert eligible is True
        assert next_stage == LifeStage.CHILD
        assert status['min_age'] is True
        assert status['care_mistakes'] is True
        assert status['happiness'] is True
    
    def test_check_evolution_eligibility_insufficient_age(self):
        """Test evolution eligibility with insufficient age."""
        self.pet.life_stage = LifeStage.BABY
        self.pet.birth_time = datetime.now() - timedelta(hours=10)  # Too young
        self.pet.care_mistakes = 0
        self.pet.happiness = 50
        self.pet.hp = 100
        
        eligible, next_stage, status = self.controller.check_evolution_eligibility(self.pet)
        
        assert eligible is False
        assert next_stage == LifeStage.CHILD
        assert status['min_age'] is False
    
    def test_check_evolution_eligibility_too_many_care_mistakes(self):
        """Test evolution eligibility with too many care mistakes."""
        self.pet.life_stage = LifeStage.BABY
        self.pet.birth_time = datetime.now() - timedelta(hours=25)
        self.pet.care_mistakes = 10  # Too many
        self.pet.happiness = 50
        self.pet.hp = 100
        
        eligible, next_stage, status = self.controller.check_evolution_eligibility(self.pet)
        
        assert eligible is False
        assert status['care_mistakes'] is False
    
    def test_trigger_evolution_success(self):
        """Test successful evolution trigger."""
        # Set up pet for successful evolution
        self.pet.life_stage = LifeStage.BABY
        self.pet.birth_time = datetime.now() - timedelta(hours=25)
        self.pet.care_mistakes = 2
        self.pet.happiness = 40
        self.pet.hp = 90
        self.pet.energy = 30
        
        old_hp = self.pet.hp
        result = self.controller.trigger_evolution(self.pet)
        
        assert result.success is True
        assert result.old_stage == LifeStage.BABY
        assert result.new_stage == LifeStage.CHILD
        assert self.pet.life_stage == LifeStage.CHILD
        assert self.pet.hp > old_hp  # Should have gained HP
        assert self.pet.evolution_timer == 0.0  # Should reset timer
        assert len(result.attribute_changes) > 0
    
    def test_trigger_evolution_failure(self):
        """Test failed evolution trigger."""
        # Set up pet that doesn't meet requirements
        self.pet.life_stage = LifeStage.BABY
        self.pet.birth_time = datetime.now() - timedelta(hours=10)  # Too young
        self.pet.care_mistakes = 0
        self.pet.happiness = 50
        
        result = self.controller.trigger_evolution(self.pet)
        
        assert result.success is False
        assert result.old_stage == LifeStage.BABY
        assert result.new_stage == LifeStage.BABY
        assert self.pet.life_stage == LifeStage.BABY  # Should not change
    
    def test_trigger_evolution_force(self):
        """Test forced evolution regardless of requirements."""
        # Set up pet that doesn't meet requirements
        self.pet.life_stage = LifeStage.BABY
        self.pet.birth_time = datetime.now() - timedelta(hours=1)  # Too young
        self.pet.care_mistakes = 20  # Too many
        self.pet.happiness = 10  # Too low
        
        result = self.controller.trigger_evolution(self.pet, force=True)
        
        assert result.success is True
        assert result.old_stage == LifeStage.BABY
        assert result.new_stage == LifeStage.CHILD
        assert self.pet.life_stage == LifeStage.CHILD
    
    def test_apply_evolution_changes_child(self):
        """Test attribute changes during evolution to child."""
        old_attributes = {
            'hp': self.pet.hp,
            'mp': self.pet.mp,
            'offense': self.pet.offense,
            'defense': self.pet.defense,
            'speed': self.pet.speed,
            'brains': self.pet.brains
        }
        
        changes = self.controller._apply_evolution_changes(
            self.pet, LifeStage.BABY, LifeStage.CHILD
        )
        
        # Check that attributes increased
        assert self.pet.hp > old_attributes['hp']
        assert self.pet.mp > old_attributes['mp']
        assert self.pet.offense > old_attributes['offense']
        assert self.pet.defense > old_attributes['defense']
        assert self.pet.speed > old_attributes['speed']
        assert self.pet.brains > old_attributes['brains']
        
        # Check that changes are recorded
        assert 'hp' in changes
        assert changes['hp'] > 0
    
    def test_apply_evolution_changes_elderly(self):
        """Test attribute changes during evolution to elderly (some decrease)."""
        self.pet.life_stage = LifeStage.ADULT
        self.pet.hp = 200
        self.pet.offense = 50
        self.pet.speed = 40
        
        old_hp = self.pet.hp
        old_offense = self.pet.offense
        old_speed = self.pet.speed
        old_brains = self.pet.brains
        
        changes = self.controller._apply_evolution_changes(
            self.pet, LifeStage.ADULT, LifeStage.ELDERLY
        )
        
        # Elderly should lose some physical attributes but gain wisdom
        assert self.pet.hp < old_hp
        assert self.pet.offense < old_offense
        assert self.pet.speed < old_speed
        assert self.pet.brains > old_brains
    
    def test_egg_type_evolution_bonuses(self):
        """Test egg type specific evolution bonuses."""
        # Test RED egg bonuses (fire-oriented)
        red_bonuses = self.controller._get_egg_type_evolution_bonus(
            EggType.RED, LifeStage.CHILD
        )
        assert AttributeType.OFFENSE in red_bonuses
        assert red_bonuses[AttributeType.OFFENSE] > 0
        
        # Test BLUE egg bonuses (water-oriented)
        blue_bonuses = self.controller._get_egg_type_evolution_bonus(
            EggType.BLUE, LifeStage.CHILD
        )
        assert AttributeType.DEFENSE in blue_bonuses
        assert blue_bonuses[AttributeType.DEFENSE] > 0
        
        # Test GREEN egg bonuses (earth-oriented)
        green_bonuses = self.controller._get_egg_type_evolution_bonus(
            EggType.GREEN, LifeStage.CHILD
        )
        assert AttributeType.HP in green_bonuses
        assert green_bonuses[AttributeType.HP] > 0
        
        # Test scaling by stage
        teen_bonuses = self.controller._get_egg_type_evolution_bonus(
            EggType.RED, LifeStage.TEEN
        )
        assert teen_bonuses[AttributeType.OFFENSE] > red_bonuses[AttributeType.OFFENSE]
    
    def test_update_learned_commands(self):
        """Test updating learned commands during evolution."""
        self.pet.learned_commands = {"eat", "sleep"}
        
        self.controller._update_learned_commands(self.pet, LifeStage.CHILD)
        
        expected_commands = {"eat", "sleep", "good", "bad", "play", "train"}
        assert expected_commands.issubset(self.pet.learned_commands)
        
        self.controller._update_learned_commands(self.pet, LifeStage.TEEN)
        
        expected_teen_commands = expected_commands | {"status", "talk"}
        assert expected_teen_commands.issubset(self.pet.learned_commands)
    
    def test_check_time_based_evolution(self):
        """Test time-based evolution checking."""
        # Test baby that's old enough for time-based evolution
        self.pet.life_stage = LifeStage.BABY
        self.pet.birth_time = datetime.now() - timedelta(hours=25)
        
        assert self.controller.check_time_based_evolution(self.pet) is True
        
        # Test baby that's too young
        self.pet.birth_time = datetime.now() - timedelta(hours=10)
        
        assert self.controller.check_time_based_evolution(self.pet) is False
    
    def test_update_evolution_timer(self):
        """Test evolution timer updates."""
        initial_timer = self.pet.evolution_timer
        
        self.controller.update_evolution_timer(self.pet, 2.5)
        
        assert self.pet.evolution_timer == initial_timer + 2.5
    
    def test_create_inheritance_dna(self):
        """Test DNA inheritance creation."""
        # Set up parent with high attributes
        parent = DigiPal(
            user_id="parent_user",
            name="ParentPal",
            egg_type=EggType.BLUE,
            life_stage=LifeStage.ADULT,
            generation=1,
            hp=250,
            mp=150,
            offense=80,
            defense=90,
            speed=70,
            brains=85
        )
        
        dna = self.controller.create_inheritance_dna(parent, "excellent")
        
        assert isinstance(dna, DNAInheritance)
        assert dna.generation == 2
        assert dna.parent_egg_type == EggType.BLUE
        assert dna.parent_final_stage == LifeStage.ADULT
        assert dna.parent_care_quality == "excellent"
        assert AttributeType.HP in dna.parent_attributes
        assert dna.parent_attributes[AttributeType.HP] == 250
        assert len(dna.inheritance_bonuses) > 0
    
    def test_calculate_inheritance_bonuses(self):
        """Test inheritance bonus calculations."""
        # Create parent with high attributes
        parent = DigiPal(
            hp=200, mp=120, offense=150, defense=130, speed=110, brains=140
        )
        
        # Test different care qualities
        perfect_bonuses = self.controller._calculate_inheritance_bonuses(parent, "perfect")
        excellent_bonuses = self.controller._calculate_inheritance_bonuses(parent, "excellent")
        poor_bonuses = self.controller._calculate_inheritance_bonuses(parent, "poor")
        
        # Perfect care should give higher bonuses than excellent
        assert sum(perfect_bonuses.values()) >= sum(excellent_bonuses.values())
        
        # Excellent care should give higher bonuses than poor
        assert sum(excellent_bonuses.values()) >= sum(poor_bonuses.values())
        
        # All bonuses should be positive for high-attribute parent
        for bonus in perfect_bonuses.values():
            assert bonus > 0
    
    def test_apply_inheritance(self):
        """Test applying inheritance to offspring."""
        # Create DNA with bonuses
        dna = DNAInheritance(
            generation=2,
            parent_care_quality="excellent",
            inheritance_bonuses={
                AttributeType.HP: 10,
                AttributeType.OFFENSE: 8,
                AttributeType.DEFENSE: 6
            }
        )
        
        offspring = DigiPal(
            user_id="offspring_user",
            name="OffspringPal",
            egg_type=EggType.RED
        )
        
        old_hp = offspring.hp
        old_offense = offspring.offense
        old_defense = offspring.defense
        
        self.controller.apply_inheritance(offspring, dna)
        
        assert offspring.generation == 2
        assert offspring.hp >= old_hp + 8  # Should get bonus minus potential random variation
        assert offspring.offense >= old_offense + 6
        assert offspring.defense >= old_defense + 4
        assert offspring.personality_traits["inherited"] is True
        assert offspring.personality_traits["parent_care_quality"] == "excellent"
    
    def test_is_death_time(self):
        """Test death time checking for elderly pets."""
        # Test non-elderly pet
        self.pet.life_stage = LifeStage.ADULT
        assert self.controller.is_death_time(self.pet) is False
        
        # Test elderly pet that's been elderly for too long
        # Calculate total max lifetime to ensure test pet is old enough
        min_time_to_elderly = sum(
            self.controller.EVOLUTION_TIMINGS.get(stage, 0) 
            for stage in [LifeStage.EGG, LifeStage.BABY, LifeStage.CHILD, 
                         LifeStage.TEEN, LifeStage.YOUNG_ADULT, LifeStage.ADULT]
        )
        elderly_time_limit = self.controller.EVOLUTION_TIMINGS.get(LifeStage.ELDERLY, 72.0)
        total_max_lifetime = min_time_to_elderly + elderly_time_limit
        
        elderly_pet = DigiPal(
            life_stage=LifeStage.ELDERLY,
            birth_time=datetime.now() - timedelta(hours=total_max_lifetime + 10)  # Past death time
        )
        assert self.controller.is_death_time(elderly_pet) is True
        
        # Test newly elderly pet
        new_elderly = DigiPal(
            life_stage=LifeStage.ELDERLY,
            birth_time=datetime.now() - timedelta(hours=min_time_to_elderly + 10)  # Just became elderly
        )
        assert self.controller.is_death_time(new_elderly) is False
    
    def test_get_evolution_requirements(self):
        """Test getting evolution requirements."""
        child_req = self.controller.get_evolution_requirements(LifeStage.CHILD)
        assert isinstance(child_req, EvolutionRequirement)
        assert child_req.min_age_hours > 0
        
        # Test non-existent stage
        none_req = self.controller.get_evolution_requirements(LifeStage.EGG)
        assert none_req is None or isinstance(none_req, EvolutionRequirement)
    
    def test_get_evolution_timing(self):
        """Test getting evolution timing."""
        baby_timing = self.controller.get_evolution_timing(LifeStage.BABY)
        assert baby_timing > 0
        assert baby_timing == self.controller.EVOLUTION_TIMINGS[LifeStage.BABY]
    
    def test_full_evolution_path(self):
        """Test complete evolution path from baby to elderly."""
        # Start with a baby
        pet = DigiPal(
            user_id="test_user",
            name="TestPal",
            egg_type=EggType.GREEN,
            life_stage=LifeStage.BABY,
            birth_time=datetime.now() - timedelta(hours=1)
        )
        
        # Set up pet with good attributes for evolution
        pet.hp = 100
        pet.mp = 60
        pet.offense = 20
        pet.defense = 20
        pet.speed = 15
        pet.brains = 15
        pet.happiness = 80
        pet.discipline = 50
        pet.weight = 25
        pet.care_mistakes = 0
        
        stages_evolved = []
        
        # Force evolution through all stages
        for target_stage in [LifeStage.CHILD, LifeStage.TEEN, LifeStage.YOUNG_ADULT, 
                           LifeStage.ADULT, LifeStage.ELDERLY]:
            result = self.controller.trigger_evolution(pet, force=True)
            if result.success:
                stages_evolved.append(result.new_stage)
        
        expected_stages = [LifeStage.CHILD, LifeStage.TEEN, LifeStage.YOUNG_ADULT, 
                          LifeStage.ADULT, LifeStage.ELDERLY]
        assert stages_evolved == expected_stages
        assert pet.life_stage == LifeStage.ELDERLY
        
        # Check that attributes increased overall (except for elderly decline)
        assert pet.hp > 100  # Should have grown despite elderly decline
        assert len(pet.learned_commands) > 4  # Should have learned more commands


class TestEvolutionRequirement:
    """Test cases for EvolutionRequirement data class."""
    
    def test_evolution_requirement_creation(self):
        """Test creating EvolutionRequirement."""
        req = EvolutionRequirement(
            min_attributes={AttributeType.HP: 100, AttributeType.OFFENSE: 50},
            max_care_mistakes=5,
            min_age_hours=24.0,
            happiness_threshold=60
        )
        
        assert req.min_attributes[AttributeType.HP] == 100
        assert req.max_care_mistakes == 5
        assert req.min_age_hours == 24.0
        assert req.happiness_threshold == 60
    
    def test_evolution_requirement_serialization(self):
        """Test EvolutionRequirement to_dict and from_dict."""
        req = EvolutionRequirement(
            min_attributes={AttributeType.HP: 100},
            max_care_mistakes=5,
            min_age_hours=24.0
        )
        
        req_dict = req.to_dict()
        assert isinstance(req_dict, dict)
        assert req_dict['min_attributes']['hp'] == 100
        assert req_dict['max_care_mistakes'] == 5
        
        restored_req = EvolutionRequirement.from_dict(req_dict)
        assert restored_req.min_attributes[AttributeType.HP] == 100
        assert restored_req.max_care_mistakes == 5


class TestEvolutionResult:
    """Test cases for EvolutionResult data class."""
    
    def test_evolution_result_creation(self):
        """Test creating EvolutionResult."""
        result = EvolutionResult(
            success=True,
            old_stage=LifeStage.BABY,
            new_stage=LifeStage.CHILD,
            attribute_changes={'hp': 30, 'offense': 5},
            message="Evolution successful!"
        )
        
        assert result.success is True
        assert result.old_stage == LifeStage.BABY
        assert result.new_stage == LifeStage.CHILD
        assert result.attribute_changes['hp'] == 30
        assert result.message == "Evolution successful!"
    
    def test_evolution_result_serialization(self):
        """Test EvolutionResult to_dict."""
        result = EvolutionResult(
            success=True,
            old_stage=LifeStage.BABY,
            new_stage=LifeStage.CHILD,
            attribute_changes={'hp': 30}
        )
        
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict['success'] is True
        assert result_dict['old_stage'] == 'baby'
        assert result_dict['new_stage'] == 'child'


class TestDNAInheritance:
    """Test cases for DNAInheritance data class."""
    
    def test_dna_inheritance_creation(self):
        """Test creating DNAInheritance."""
        dna = DNAInheritance(
            parent_attributes={AttributeType.HP: 200, AttributeType.OFFENSE: 80},
            parent_care_quality="excellent",
            parent_final_stage=LifeStage.ADULT,
            parent_egg_type=EggType.BLUE,
            generation=2,
            inheritance_bonuses={AttributeType.HP: 10, AttributeType.OFFENSE: 5}
        )
        
        assert dna.parent_attributes[AttributeType.HP] == 200
        assert dna.parent_care_quality == "excellent"
        assert dna.parent_final_stage == LifeStage.ADULT
        assert dna.parent_egg_type == EggType.BLUE
        assert dna.generation == 2
        assert dna.inheritance_bonuses[AttributeType.HP] == 10
    
    def test_dna_inheritance_serialization(self):
        """Test DNAInheritance to_dict and from_dict."""
        dna = DNAInheritance(
            parent_attributes={AttributeType.HP: 200},
            parent_care_quality="excellent",
            parent_final_stage=LifeStage.ADULT,
            parent_egg_type=EggType.BLUE,
            generation=2,
            inheritance_bonuses={AttributeType.HP: 10}
        )
        
        dna_dict = dna.to_dict()
        assert isinstance(dna_dict, dict)
        assert dna_dict['parent_attributes']['hp'] == 200
        assert dna_dict['parent_care_quality'] == "excellent"
        assert dna_dict['parent_final_stage'] == 'adult'
        assert dna_dict['parent_egg_type'] == 'blue'
        
        restored_dna = DNAInheritance.from_dict(dna_dict)
        assert restored_dna.parent_attributes[AttributeType.HP] == 200
        assert restored_dna.parent_care_quality == "excellent"
        assert restored_dna.parent_final_stage == LifeStage.ADULT
        assert restored_dna.parent_egg_type == EggType.BLUE


class TestEvolutionIntegration:
    """Integration tests for evolution system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.controller = EvolutionController()
    
    def test_complete_lifecycle_with_inheritance(self):
        """Test complete lifecycle from egg to death with inheritance."""
        # Create initial pet
        parent = DigiPal(
            user_id="test_user",
            name="ParentPal",
            egg_type=EggType.RED,
            life_stage=LifeStage.BABY,
            generation=1
        )
        
        # Simulate good care
        parent.hp = 150
        parent.offense = 60
        parent.defense = 50
        parent.happiness = 80
        parent.care_mistakes = 2
        
        # Force evolution to adult
        for _ in range(4):  # Baby -> Child -> Teen -> Young Adult -> Adult
            self.controller.trigger_evolution(parent, force=True)
        
        assert parent.life_stage == LifeStage.ADULT
        
        # Create inheritance DNA
        dna = self.controller.create_inheritance_dna(parent, "excellent")
        
        # Create offspring
        offspring = DigiPal(
            user_id="test_user",
            name="OffspringPal",
            egg_type=EggType.RED,
            life_stage=LifeStage.EGG
        )
        
        # Apply inheritance
        self.controller.apply_inheritance(offspring, dna)
        
        assert offspring.generation == 2
        assert offspring.hp >= parent.hp * 0.1  # Should have some inheritance bonus
        assert offspring.personality_traits["inherited"] is True
    
    @patch('random.randint')
    @patch('random.choice')
    def test_inheritance_randomization(self, mock_choice, mock_randint):
        """Test that inheritance includes proper randomization."""
        # Mock random functions for predictable testing
        mock_randint.return_value = 1  # Small variation
        mock_choice.return_value = AttributeType.HP
        
        parent = DigiPal(hp=200, offense=150, defense=120)
        dna = self.controller.create_inheritance_dna(parent, "perfect")
        
        offspring = DigiPal()
        old_hp = offspring.hp
        
        self.controller.apply_inheritance(offspring, dna)
        
        # Should have inheritance bonus plus/minus random variation
        assert offspring.hp != old_hp
        
        # Verify random functions were called for variation
        assert mock_randint.called
    
    def test_evolution_timing_consistency(self):
        """Test that evolution timings are consistent and logical."""
        timings = self.controller.get_all_evolution_timings()
        
        # Check that each stage has a reasonable timing
        assert timings[LifeStage.EGG] < timings[LifeStage.BABY]
        assert timings[LifeStage.BABY] < timings[LifeStage.CHILD]
        assert timings[LifeStage.CHILD] < timings[LifeStage.TEEN]
        
        # Check that all timings are positive
        for timing in timings.values():
            assert timing > 0
    
    def test_evolution_requirements_progression(self):
        """Test that evolution requirements become progressively more demanding."""
        child_req = self.controller.get_evolution_requirements(LifeStage.CHILD)
        teen_req = self.controller.get_evolution_requirements(LifeStage.TEEN)
        adult_req = self.controller.get_evolution_requirements(LifeStage.ADULT)
        
        # Age requirements should increase
        assert child_req.min_age_hours < teen_req.min_age_hours
        assert teen_req.min_age_hours < adult_req.min_age_hours
        
        # Happiness thresholds should generally increase
        assert child_req.happiness_threshold <= teen_req.happiness_threshold
        assert teen_req.happiness_threshold <= adult_req.happiness_threshold
        
        # Attribute requirements should become more demanding
        if AttributeType.HP in child_req.min_attributes and AttributeType.HP in adult_req.min_attributes:
            assert child_req.min_attributes[AttributeType.HP] < adult_req.min_attributes[AttributeType.HP]