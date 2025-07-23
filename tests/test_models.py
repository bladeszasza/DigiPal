"""
Unit tests for DigiPal core data models.
"""

import unittest
from datetime import datetime, timedelta
from digipal.core.models import DigiPal, Interaction, Command, AttributeModifier, CareAction
from digipal.core.enums import (
    EggType, LifeStage, AttributeType, CommandType, 
    InteractionResult, CareActionType
)


class TestDigiPal(unittest.TestCase):
    """Test cases for DigiPal model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.digipal = DigiPal(
            user_id="test_user",
            name="TestPal",
            egg_type=EggType.RED
        )
    
    def test_digipal_initialization(self):
        """Test DigiPal initialization with default values."""
        self.assertIsNotNone(self.digipal.id)
        self.assertEqual(self.digipal.user_id, "test_user")
        self.assertEqual(self.digipal.name, "TestPal")
        self.assertEqual(self.digipal.egg_type, EggType.RED)
        self.assertEqual(self.digipal.life_stage, LifeStage.EGG)
        self.assertEqual(self.digipal.generation, 1)
    
    def test_egg_type_attribute_initialization(self):
        """Test that egg types initialize with correct attributes."""
        # Test RED egg (fire-oriented, higher attack)
        red_pal = DigiPal(egg_type=EggType.RED)
        self.assertEqual(red_pal.offense, 15)
        self.assertEqual(red_pal.defense, 8)
        
        # Test BLUE egg (water-oriented, higher defense)
        blue_pal = DigiPal(egg_type=EggType.BLUE)
        self.assertEqual(blue_pal.defense, 15)
        self.assertEqual(blue_pal.offense, 8)
        
        # Test GREEN egg (earth-oriented, higher health)
        green_pal = DigiPal(egg_type=EggType.GREEN)
        self.assertEqual(green_pal.hp, 120)
        self.assertEqual(green_pal.offense, 10)
        self.assertEqual(green_pal.defense, 12)
    
    def test_attribute_bounds_checking(self):
        """Test attribute bounds checking."""
        # Test setting within bounds
        self.digipal.set_attribute(AttributeType.HP, 150)
        self.assertEqual(self.digipal.hp, 150)
        
        # Test upper bound clamping
        self.digipal.set_attribute(AttributeType.HP, 1500)
        self.assertEqual(self.digipal.hp, 999)
        
        # Test lower bound clamping
        self.digipal.set_attribute(AttributeType.HP, -50)
        self.assertEqual(self.digipal.hp, 0)
        
        # Test happiness bounds (0-100)
        self.digipal.set_attribute(AttributeType.HAPPINESS, 150)
        self.assertEqual(self.digipal.happiness, 100)
        
        self.digipal.set_attribute(AttributeType.HAPPINESS, -10)
        self.assertEqual(self.digipal.happiness, 0)
    
    def test_attribute_modification(self):
        """Test attribute modification with deltas."""
        initial_hp = self.digipal.hp
        self.digipal.modify_attribute(AttributeType.HP, 20)
        self.assertEqual(self.digipal.hp, initial_hp + 20)
        
        self.digipal.modify_attribute(AttributeType.HP, -30)
        self.assertEqual(self.digipal.hp, initial_hp - 10)
    
    def test_command_understanding_by_stage(self):
        """Test command understanding based on life stage."""
        # EGG stage - no commands
        self.digipal.life_stage = LifeStage.EGG
        self.assertFalse(self.digipal.can_understand_command("eat"))
        
        # BABY stage - basic commands only
        self.digipal.life_stage = LifeStage.BABY
        self.assertTrue(self.digipal.can_understand_command("eat"))
        self.assertTrue(self.digipal.can_understand_command("sleep"))
        self.assertTrue(self.digipal.can_understand_command("good"))
        self.assertTrue(self.digipal.can_understand_command("bad"))
        self.assertFalse(self.digipal.can_understand_command("train"))
        
        # CHILD stage - more commands
        self.digipal.life_stage = LifeStage.CHILD
        self.assertTrue(self.digipal.can_understand_command("train"))
        self.assertTrue(self.digipal.can_understand_command("play"))
        
        # ADULT stage - advanced commands
        self.digipal.life_stage = LifeStage.ADULT
        self.assertTrue(self.digipal.can_understand_command("teach"))
    
    def test_age_calculation(self):
        """Test age calculation in hours."""
        # Set birth time to 2 hours ago
        self.digipal.birth_time = datetime.now() - timedelta(hours=2)
        age = self.digipal.get_age_hours()
        self.assertAlmostEqual(age, 2.0, delta=0.1)
    
    def test_serialization_to_dict(self):
        """Test DigiPal serialization to dictionary."""
        data = self.digipal.to_dict()
        
        # Check required fields
        self.assertIn('id', data)
        self.assertIn('user_id', data)
        self.assertIn('egg_type', data)
        self.assertIn('life_stage', data)
        self.assertEqual(data['egg_type'], 'red')
        self.assertEqual(data['life_stage'], 'egg')
        
        # Check datetime serialization
        self.assertIsInstance(data['birth_time'], str)
        self.assertIsInstance(data['last_interaction'], str)
    
    def test_deserialization_from_dict(self):
        """Test DigiPal deserialization from dictionary."""
        # First serialize
        original_data = self.digipal.to_dict()
        
        # Then deserialize
        restored_pal = DigiPal.from_dict(original_data)
        
        # Check key attributes
        self.assertEqual(restored_pal.id, self.digipal.id)
        self.assertEqual(restored_pal.user_id, self.digipal.user_id)
        self.assertEqual(restored_pal.egg_type, self.digipal.egg_type)
        self.assertEqual(restored_pal.life_stage, self.digipal.life_stage)
        self.assertEqual(restored_pal.hp, self.digipal.hp)


class TestInteraction(unittest.TestCase):
    """Test cases for Interaction model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.interaction = Interaction(
            user_input="eat",
            interpreted_command="feed",
            pet_response="*munch munch*",
            attribute_changes={"happiness": 5, "energy": -10},
            success=True,
            result=InteractionResult.SUCCESS
        )
    
    def test_interaction_initialization(self):
        """Test Interaction initialization."""
        self.assertEqual(self.interaction.user_input, "eat")
        self.assertEqual(self.interaction.interpreted_command, "feed")
        self.assertEqual(self.interaction.pet_response, "*munch munch*")
        self.assertTrue(self.interaction.success)
        self.assertEqual(self.interaction.result, InteractionResult.SUCCESS)
    
    def test_interaction_serialization(self):
        """Test Interaction serialization and deserialization."""
        data = self.interaction.to_dict()
        
        # Check serialization
        self.assertIn('timestamp', data)
        self.assertIn('user_input', data)
        self.assertEqual(data['result'], 'success')
        
        # Test deserialization
        restored_interaction = Interaction.from_dict(data)
        self.assertEqual(restored_interaction.user_input, self.interaction.user_input)
        self.assertEqual(restored_interaction.result, self.interaction.result)


class TestCommand(unittest.TestCase):
    """Test cases for Command model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.command = Command(
            action="feed",
            parameters={"food_type": "meat"},
            stage_appropriate=True,
            energy_required=5,
            command_type=CommandType.EAT
        )
    
    def test_command_initialization(self):
        """Test Command initialization."""
        self.assertEqual(self.command.action, "feed")
        self.assertEqual(self.command.parameters["food_type"], "meat")
        self.assertTrue(self.command.stage_appropriate)
        self.assertEqual(self.command.energy_required, 5)
        self.assertEqual(self.command.command_type, CommandType.EAT)
    
    def test_command_serialization(self):
        """Test Command serialization and deserialization."""
        data = self.command.to_dict()
        
        # Check serialization
        self.assertEqual(data['command_type'], 'eat')
        self.assertEqual(data['action'], 'feed')
        
        # Test deserialization
        restored_command = Command.from_dict(data)
        self.assertEqual(restored_command.command_type, self.command.command_type)
        self.assertEqual(restored_command.action, self.command.action)


class TestAttributeModifier(unittest.TestCase):
    """Test cases for AttributeModifier model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.modifier = AttributeModifier(
            attribute=AttributeType.HP,
            change=10,
            conditions=["well_fed", "happy"]
        )
    
    def test_attribute_modifier_initialization(self):
        """Test AttributeModifier initialization."""
        self.assertEqual(self.modifier.attribute, AttributeType.HP)
        self.assertEqual(self.modifier.change, 10)
        self.assertIn("well_fed", self.modifier.conditions)
    
    def test_attribute_modifier_serialization(self):
        """Test AttributeModifier serialization and deserialization."""
        data = self.modifier.to_dict()
        
        # Check serialization
        self.assertEqual(data['attribute'], 'hp')
        self.assertEqual(data['change'], 10)
        
        # Test deserialization
        restored_modifier = AttributeModifier.from_dict(data)
        self.assertEqual(restored_modifier.attribute, self.modifier.attribute)
        self.assertEqual(restored_modifier.change, self.modifier.change)


class TestCareAction(unittest.TestCase):
    """Test cases for CareAction model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.care_action = CareAction(
            name="Basic Training",
            action_type=CareActionType.TRAIN,
            energy_cost=15,
            happiness_change=-5,
            attribute_modifiers=[
                AttributeModifier(AttributeType.OFFENSE, 2),
                AttributeModifier(AttributeType.SPEED, 1)
            ],
            success_conditions=["energy > 15"],
            failure_effects=[
                AttributeModifier(AttributeType.HAPPINESS, -10)
            ]
        )
    
    def test_care_action_initialization(self):
        """Test CareAction initialization."""
        self.assertEqual(self.care_action.name, "Basic Training")
        self.assertEqual(self.care_action.action_type, CareActionType.TRAIN)
        self.assertEqual(self.care_action.energy_cost, 15)
        self.assertEqual(len(self.care_action.attribute_modifiers), 2)
    
    def test_care_action_serialization(self):
        """Test CareAction serialization and deserialization."""
        data = self.care_action.to_dict()
        
        # Check serialization
        self.assertEqual(data['action_type'], 'train')
        self.assertEqual(len(data['attribute_modifiers']), 2)
        
        # Test deserialization
        restored_action = CareAction.from_dict(data)
        self.assertEqual(restored_action.action_type, self.care_action.action_type)
        self.assertEqual(len(restored_action.attribute_modifiers), 2)


if __name__ == '__main__':
    unittest.main()