"""
Automated test data generation for various pet states and scenarios.

This module provides utilities to generate test data for different
DigiPal states, life stages, and interaction scenarios.
"""

import pytest
import random
import tempfile
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import Mock

from digipal.core.digipal_core import DigiPalCore
from digipal.core.models import DigiPal, Interaction
from digipal.core.enums import EggType, LifeStage, InteractionResult
from digipal.storage.storage_manager import StorageManager
from digipal.ai.communication import AICommunication


class TestDataGenerator:
    """Generates test data for various DigiPal scenarios."""
    
    def __init__(self, storage_manager: StorageManager, digipal_core: DigiPalCore):
        self.storage = storage_manager
        self.core = digipal_core
        self.random = random.Random(42)  # Fixed seed for reproducible tests
    
    def generate_random_pet(self, user_id: str, name: str = None, egg_type: EggType = None, 
                           life_stage: LifeStage = None) -> DigiPal:
        """Generate a DigiPal with random attributes."""
        if name is None:
            name = f"TestPal_{self.random.randint(1000, 9999)}"
        
        if egg_type is None:
            egg_type = self.random.choice(list(EggType))
        
        if life_stage is None:
            life_stage = self.random.choice(list(LifeStage))
        
        pet = self.core.create_new_pet(egg_type, user_id, name)
        
        # Randomize attributes
        pet.life_stage = life_stage
        pet.hp = self.random.randint(50, 300)
        pet.mp = self.random.randint(20, 150)
        pet.offense = self.random.randint(5, 100)
        pet.defense = self.random.randint(5, 100)
        pet.speed = self.random.randint(5, 100)
        pet.brains = self.random.randint(5, 100)
        pet.discipline = self.random.randint(0, 100)
        pet.happiness = self.random.randint(0, 100)
        pet.weight = self.random.randint(5, 50)
        pet.care_mistakes = self.random.randint(0, 20)
        pet.energy = self.random.randint(0, 100)
        
        # Randomize age
        hours_old = self.random.randint(1, 200)
        pet.birth_time = datetime.now() - timedelta(hours=hours_old)
        
        return pet
    
    def generate_pet_at_life_stage(self, user_id: str, life_stage: LifeStage, 
                                  name: str = None) -> DigiPal:
        """Generate a DigiPal at a specific life stage with appropriate attributes."""
        if name is None:
            name = f"{life_stage.value.title()}Pal_{self.random.randint(100, 999)}"
        
        egg_type = self.random.choice(list(EggType))
        pet = self.core.create_new_pet(egg_type, user_id, name)
        pet.life_stage = life_stage
        
        # Set age-appropriate attributes
        if life_stage == LifeStage.EGG:
            pet.birth_time = datetime.now()
            # Eggs have minimal attributes
        elif life_stage == LifeStage.BABY:
            pet.birth_time = datetime.now() - timedelta(hours=self.random.randint(1, 23))
            pet.hp = self.random.randint(80, 120)
            pet.energy = self.random.randint(40, 80)
            pet.happiness = self.random.randint(60, 90)
        elif life_stage == LifeStage.CHILD:
            pet.birth_time = datetime.now() - timedelta(hours=self.random.randint(24, 71))
            pet.hp = self.random.randint(120, 180)
            pet.offense = self.random.randint(15, 40)
            pet.defense = self.random.randint(15, 40)
            pet.energy = self.random.randint(50, 90)
        elif life_stage == LifeStage.TEEN:
            pet.birth_time = datetime.now() - timedelta(hours=self.random.randint(72, 119))
            pet.hp = self.random.randint(150, 220)
            pet.offense = self.random.randint(25, 60)
            pet.defense = self.random.randint(25, 60)
            pet.speed = self.random.randint(20, 50)
            pet.brains = self.random.randint(20, 50)
        elif life_stage == LifeStage.YOUNG_ADULT:
            pet.birth_time = datetime.now() - timedelta(hours=self.random.randint(120, 167))
            pet.hp = self.random.randint(180, 250)
            pet.offense = self.random.randint(40, 80)
            pet.defense = self.random.randint(40, 80)
            pet.speed = self.random.randint(30, 70)
            pet.brains = self.random.randint(30, 70)
        elif life_stage == LifeStage.ADULT:
            pet.birth_time = datetime.now() - timedelta(hours=self.random.randint(168, 239))
            pet.hp = self.random.randint(200, 300)
            pet.offense = self.random.randint(50, 100)
            pet.defense = self.random.randint(50, 100)
            pet.speed = self.random.randint(40, 80)
            pet.brains = self.random.randint(40, 80)
        elif life_stage == LifeStage.ELDERLY:
            pet.birth_time = datetime.now() - timedelta(hours=self.random.randint(240, 400))
            pet.hp = self.random.randint(150, 250)  # Slightly reduced from adult
            pet.offense = self.random.randint(60, 120)  # Can be higher due to experience
            pet.defense = self.random.randint(60, 120)
            pet.speed = self.random.randint(30, 60)  # Reduced speed
            pet.brains = self.random.randint(60, 100)  # High wisdom
        
        return pet
    
    def generate_interaction_history(self, pet: DigiPal, num_interactions: int = 10) -> List[Interaction]:
        """Generate random interaction history for a pet."""
        interactions = []
        
        commands = ["eat", "sleep", "play", "train", "praise", "scold", "status", "hello"]
        responses = [
            "I'm happy!", "Yummy!", "That was fun!", "I'm getting stronger!",
            "Thank you!", "I understand.", "I'm doing well!", "Hello there!"
        ]
        
        for i in range(num_interactions):
            command = self.random.choice(commands)
            response = self.random.choice(responses)
            success = self.random.choice([True, True, True, False])  # 75% success rate
            
            # Generate attribute changes
            attribute_changes = {}
            if success:
                if command == "eat":
                    attribute_changes = {"energy": self.random.randint(5, 15), "happiness": self.random.randint(1, 5)}
                elif command == "train":
                    attribute_changes = {"offense": self.random.randint(1, 3), "energy": -self.random.randint(5, 10)}
                elif command == "praise":
                    attribute_changes = {"happiness": self.random.randint(3, 8), "discipline": self.random.randint(1, 3)}
                elif command == "play":
                    attribute_changes = {"happiness": self.random.randint(5, 10), "energy": -self.random.randint(3, 8)}
            
            interaction = Interaction(
                timestamp=datetime.now() - timedelta(hours=self.random.randint(1, 48)),
                user_input=command,
                interpreted_command=command,
                pet_response=response,
                attribute_changes=attribute_changes,
                success=success,
                result=InteractionResult.SUCCESS if success else InteractionResult.FAILURE
            )
            
            interactions.append(interaction)
        
        # Sort by timestamp
        interactions.sort(key=lambda x: x.timestamp)
        return interactions
    
    def generate_care_scenario(self, scenario_type: str) -> Dict[str, Any]:
        """Generate specific care scenarios for testing."""
        scenarios = {
            "neglected": {
                "happiness": self.random.randint(0, 20),
                "energy": self.random.randint(0, 15),
                "care_mistakes": self.random.randint(10, 25),
                "weight": self.random.randint(5, 12),  # Underweight
                "description": "Neglected pet with low happiness and energy"
            },
            "well_cared": {
                "happiness": self.random.randint(80, 100),
                "energy": self.random.randint(70, 100),
                "care_mistakes": self.random.randint(0, 3),
                "weight": self.random.randint(18, 25),  # Healthy weight
                "description": "Well-cared pet with high happiness and energy"
            },
            "overfed": {
                "happiness": self.random.randint(60, 80),
                "energy": self.random.randint(50, 70),
                "care_mistakes": self.random.randint(5, 15),
                "weight": self.random.randint(35, 50),  # Overweight
                "description": "Overfed pet with weight issues"
            },
            "overtrained": {
                "happiness": self.random.randint(30, 50),
                "energy": self.random.randint(10, 30),
                "care_mistakes": self.random.randint(8, 20),
                "offense": self.random.randint(80, 120),
                "defense": self.random.randint(80, 120),
                "description": "Overtrained pet with high stats but low happiness"
            },
            "balanced": {
                "happiness": self.random.randint(60, 80),
                "energy": self.random.randint(60, 80),
                "care_mistakes": self.random.randint(2, 8),
                "weight": self.random.randint(18, 28),
                "description": "Balanced pet with moderate stats"
            }
        }
        
        return scenarios.get(scenario_type, scenarios["balanced"])
    
    def apply_care_scenario(self, pet: DigiPal, scenario_type: str) -> DigiPal:
        """Apply a care scenario to a pet."""
        scenario = self.generate_care_scenario(scenario_type)
        
        for attribute, value in scenario.items():
            if attribute != "description" and hasattr(pet, attribute):
                setattr(pet, attribute, value)
        
        return pet
    
    def generate_evolution_ready_pet(self, user_id: str, current_stage: LifeStage) -> DigiPal:
        """Generate a pet that's ready for evolution."""
        pet = self.generate_pet_at_life_stage(user_id, current_stage)
        
        # Set attributes that make evolution likely
        pet.care_mistakes = self.random.randint(0, 2)  # Very few mistakes
        pet.happiness = self.random.randint(80, 100)  # High happiness
        pet.discipline = self.random.randint(70, 100)  # Good discipline
        
        # Set age to be ready for evolution
        if current_stage == LifeStage.BABY:
            pet.birth_time = datetime.now() - timedelta(hours=25)  # Past baby stage
        elif current_stage == LifeStage.CHILD:
            pet.birth_time = datetime.now() - timedelta(hours=73)  # Past child stage
        elif current_stage == LifeStage.TEEN:
            pet.birth_time = datetime.now() - timedelta(hours=121)  # Past teen stage
        elif current_stage == LifeStage.YOUNG_ADULT:
            pet.birth_time = datetime.now() - timedelta(hours=169)  # Past young adult stage
        elif current_stage == LifeStage.ADULT:
            pet.birth_time = datetime.now() - timedelta(hours=241)  # Past adult stage
        
        return pet
    
    def generate_test_dataset(self, num_pets: int = 50) -> List[DigiPal]:
        """Generate a comprehensive test dataset with various pet states."""
        pets = []
        
        # Generate pets at each life stage
        for life_stage in LifeStage:
            for i in range(num_pets // len(LifeStage)):
                user_id = f"dataset_user_{len(pets)}"
                self.storage.create_user(user_id, user_id)
                
                pet = self.generate_pet_at_life_stage(user_id, life_stage)
                
                # Apply random care scenario
                scenario = self.random.choice(["neglected", "well_cared", "overfed", "overtrained", "balanced"])
                pet = self.apply_care_scenario(pet, scenario)
                
                # Add interaction history
                interactions = self.generate_interaction_history(pet, self.random.randint(5, 20))
                pet.conversation_history = interactions
                
                # Save pet
                self.storage.save_pet(pet)
                pets.append(pet)
        
        return pets


class TestDataGenerationValidation:
    """Tests to validate the test data generation functionality."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def test_system(self, temp_db_path):
        """Create system for test data generation."""
        storage_manager = StorageManager(temp_db_path)
        
        mock_ai = Mock(spec=AICommunication)
        mock_ai.memory_manager = Mock()
        mock_ai.memory_manager.get_interaction_summary.return_value = {}
        mock_ai.unload_all_models.return_value = None
        
        digipal_core = DigiPalCore(storage_manager, mock_ai)
        generator = TestDataGenerator(storage_manager, digipal_core)
        
        return {
            'core': digipal_core,
            'storage': storage_manager,
            'generator': generator
        }
    
    def test_random_pet_generation(self, test_system):
        """Test random pet generation."""
        generator = test_system['generator']
        storage = test_system['storage']
        
        # Create user
        storage.create_user("random_user", "random_user")
        
        # Generate random pets
        pets = []
        for i in range(10):
            pet = generator.generate_random_pet("random_user", f"RandomPet_{i}")
            pets.append(pet)
        
        # Validate pets are different
        assert len(pets) == 10
        assert len(set(pet.name for pet in pets)) == 10  # All names should be unique
        
        # Validate attribute ranges
        for pet in pets:
            assert 50 <= pet.hp <= 300
            assert 20 <= pet.mp <= 150
            assert 5 <= pet.offense <= 100
            assert 5 <= pet.defense <= 100
            assert 0 <= pet.happiness <= 100
            assert 0 <= pet.energy <= 100
    
    def test_life_stage_specific_generation(self, test_system):
        """Test generation of pets at specific life stages."""
        generator = test_system['generator']
        storage = test_system['storage']
        
        for life_stage in LifeStage:
            user_id = f"stage_user_{life_stage.value}"
            storage.create_user(user_id, user_id)
            
            pet = generator.generate_pet_at_life_stage(user_id, life_stage)
            
            assert pet.life_stage == life_stage
            assert pet.user_id == user_id
            
            # Validate age is appropriate for life stage
            age_hours = (datetime.now() - pet.birth_time).total_seconds() / 3600
            
            if life_stage == LifeStage.EGG:
                assert age_hours < 1
            elif life_stage == LifeStage.BABY:
                assert 1 <= age_hours <= 23
            elif life_stage == LifeStage.CHILD:
                assert 24 <= age_hours <= 71
            elif life_stage == LifeStage.TEEN:
                assert 72 <= age_hours <= 119
            elif life_stage == LifeStage.YOUNG_ADULT:
                assert 120 <= age_hours <= 167
            elif life_stage == LifeStage.ADULT:
                assert 168 <= age_hours <= 239
            elif life_stage == LifeStage.ELDERLY:
                assert age_hours >= 240
    
    def test_interaction_history_generation(self, test_system):
        """Test interaction history generation."""
        generator = test_system['generator']
        storage = test_system['storage']
        
        # Create user and pet
        storage.create_user("history_user", "history_user")
        pet = generator.generate_random_pet("history_user", "HistoryPal")
        
        # Generate interaction history
        interactions = generator.generate_interaction_history(pet, 15)
        
        assert len(interactions) == 15
        
        # Validate interactions are sorted by timestamp
        timestamps = [interaction.timestamp for interaction in interactions]
        assert timestamps == sorted(timestamps)
        
        # Validate interaction structure
        for interaction in interactions:
            assert interaction.user_input is not None
            assert interaction.pet_response is not None
            assert isinstance(interaction.success, bool)
            assert interaction.result in [InteractionResult.SUCCESS, InteractionResult.FAILURE]
    
    def test_care_scenario_generation(self, test_system):
        """Test care scenario generation and application."""
        generator = test_system['generator']
        storage = test_system['storage']
        
        # Create user and pet
        storage.create_user("scenario_user", "scenario_user")
        pet = generator.generate_random_pet("scenario_user", "ScenarioPal")
        
        # Test different scenarios
        scenarios = ["neglected", "well_cared", "overfed", "overtrained", "balanced"]
        
        for scenario_type in scenarios:
            # Apply scenario
            test_pet = generator.apply_care_scenario(pet, scenario_type)
            scenario = generator.generate_care_scenario(scenario_type)
            
            # Validate scenario was applied
            if scenario_type == "neglected":
                assert test_pet.happiness <= 20
                assert test_pet.energy <= 15
                assert test_pet.care_mistakes >= 10
            elif scenario_type == "well_cared":
                assert test_pet.happiness >= 80
                assert test_pet.energy >= 70
                assert test_pet.care_mistakes <= 3
            elif scenario_type == "overfed":
                assert test_pet.weight >= 35
            elif scenario_type == "overtrained":
                assert test_pet.happiness <= 50
                assert test_pet.energy <= 30
    
    def test_evolution_ready_pet_generation(self, test_system):
        """Test generation of evolution-ready pets."""
        generator = test_system['generator']
        storage = test_system['storage']
        
        # Test evolution readiness for different stages
        stages_to_test = [LifeStage.BABY, LifeStage.CHILD, LifeStage.TEEN]
        
        for stage in stages_to_test:
            user_id = f"evolution_user_{stage.value}"
            storage.create_user(user_id, user_id)
            
            pet = generator.generate_evolution_ready_pet(user_id, stage)
            
            assert pet.life_stage == stage
            assert pet.care_mistakes <= 2
            assert pet.happiness >= 80
            assert pet.discipline >= 70
            
            # Check age is appropriate for evolution
            age_hours = (datetime.now() - pet.birth_time).total_seconds() / 3600
            
            if stage == LifeStage.BABY:
                assert age_hours >= 24  # Ready to evolve to child
            elif stage == LifeStage.CHILD:
                assert age_hours >= 72  # Ready to evolve to teen
            elif stage == LifeStage.TEEN:
                assert age_hours >= 120  # Ready to evolve to young adult
    
    def test_comprehensive_dataset_generation(self, test_system):
        """Test generation of comprehensive test dataset."""
        generator = test_system['generator']
        
        # Generate dataset
        pets = generator.generate_test_dataset(num_pets=21)  # 3 pets per life stage
        
        assert len(pets) == 21
        
        # Validate distribution across life stages
        stage_counts = {}
        for pet in pets:
            stage = pet.life_stage
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        
        # Should have pets in each life stage
        assert len(stage_counts) == len(LifeStage)
        
        # Validate pets have interaction history
        pets_with_history = [pet for pet in pets if len(pet.conversation_history) > 0]
        assert len(pets_with_history) > 0
        
        # Validate pets are saved to storage
        for pet in pets[:5]:  # Check first 5 pets
            loaded_pet = test_system['storage'].load_pet(pet.user_id)
            assert loaded_pet is not None
            assert loaded_pet.name == pet.name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])