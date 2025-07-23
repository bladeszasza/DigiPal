"""
Integration tests for DigiPal Core Engine.

Tests the complete pet lifecycle management, interaction processing,
and orchestration of all components.
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from digipal.core.digipal_core import DigiPalCore, PetState, InteractionProcessor
from digipal.core.models import DigiPal, Interaction
from digipal.core.enums import EggType, LifeStage, InteractionResult, AttributeType
from digipal.storage.storage_manager import StorageManager
from digipal.ai.communication import AICommunication


class TestPetState:
    """Test PetState functionality."""
    
    def test_pet_state_creation(self):
        """Test creating PetState from DigiPal."""
        pet = DigiPal(
            user_id="test_user",
            name="TestPal",
            egg_type=EggType.RED,
            life_stage=LifeStage.CHILD,
            hp=150,
            energy=30,
            happiness=80
        )
        
        state = PetState(pet)
        
        assert state.id == pet.id
        assert state.user_id == "test_user"
        assert state.name == "TestPal"
        assert state.life_stage == LifeStage.CHILD
        assert state.hp == 150
        assert state.energy == 30
        assert state.happiness == 80
        assert state.needs_attention == True  # Energy < 40
        assert "tired" in state.status_summary.lower()  # Energy=30 should show "Getting tired"
    
    def test_pet_state_needs_attention(self):
        """Test needs attention calculation."""
        # Low energy pet
        pet_low_energy = DigiPal(energy=15)
        state = PetState(pet_low_energy)
        assert state.needs_attention == True
        
        # Unhappy pet
        pet_unhappy = DigiPal(happiness=20, energy=80)
        state = PetState(pet_unhappy)
        assert state.needs_attention == True
        
        # Underweight pet
        pet_underweight = DigiPal(weight=5, energy=80, happiness=80)
        state = PetState(pet_underweight)
        assert state.needs_attention == True
        
        # Healthy pet
        pet_healthy = DigiPal(energy=80, happiness=80, weight=25)
        state = PetState(pet_healthy)
        assert state.needs_attention == False
    
    def test_pet_state_to_dict(self):
        """Test converting PetState to dictionary."""
        pet = DigiPal(
            user_id="test_user",
            name="TestPal",
            egg_type=EggType.BLUE,
            life_stage=LifeStage.TEEN
        )
        
        state = PetState(pet)
        state_dict = state.to_dict()
        
        assert state_dict['user_id'] == "test_user"
        assert state_dict['name'] == "TestPal"
        assert state_dict['life_stage'] == "teen"
        assert 'attributes' in state_dict
        assert 'status' in state_dict
        assert state_dict['attributes']['hp'] == pet.hp
        assert state_dict['status']['needs_attention'] == state.needs_attention


class TestInteractionProcessor:
    """Test InteractionProcessor functionality."""
    
    @pytest.fixture
    def mock_ai_communication(self):
        """Create mock AI communication."""
        ai_comm = Mock(spec=AICommunication)
        ai_comm.process_interaction.return_value = Interaction(
            user_input="test input",
            interpreted_command="eat",
            pet_response="Test response",
            success=True,
            result=InteractionResult.SUCCESS
        )
        ai_comm.process_speech.return_value = "hello there"
        return ai_comm
    
    @pytest.fixture
    def mock_attribute_engine(self):
        """Create mock attribute engine."""
        from digipal.core.attribute_engine import AttributeEngine
        engine = Mock(spec=AttributeEngine)
        engine.apply_care_action.return_value = (True, Interaction(
            user_input="eat",
            interpreted_command="eat",
            pet_response="Yummy!",
            attribute_changes={"energy": 5, "happiness": 3},
            success=True,
            result=InteractionResult.SUCCESS
        ))
        return engine
    
    @pytest.fixture
    def interaction_processor(self, mock_ai_communication, mock_attribute_engine):
        """Create InteractionProcessor with mocked dependencies."""
        return InteractionProcessor(mock_ai_communication, mock_attribute_engine)
    
    def test_process_text_interaction(self, interaction_processor, mock_ai_communication):
        """Test processing text interaction."""
        pet = DigiPal(user_id="test_user", life_stage=LifeStage.CHILD)
        
        result = interaction_processor.process_text_interaction("feed me", pet)
        
        assert result.user_input == "test input"
        assert result.interpreted_command == "eat"
        assert result.success == True
        mock_ai_communication.process_interaction.assert_called_once_with("feed me", pet)
    
    def test_process_speech_interaction_success(self, interaction_processor, mock_ai_communication):
        """Test processing successful speech interaction."""
        pet = DigiPal(user_id="test_user", life_stage=LifeStage.CHILD)
        audio_data = b"fake_audio_data"
        
        result = interaction_processor.process_speech_interaction(audio_data, pet)
        
        assert result.success == True
        mock_ai_communication.process_speech.assert_called_once_with(audio_data, None)
        mock_ai_communication.process_interaction.assert_called_once_with("hello there", pet)
    
    def test_process_speech_interaction_failure(self, interaction_processor, mock_ai_communication):
        """Test processing failed speech interaction."""
        mock_ai_communication.process_speech.return_value = ""  # Speech recognition failed
        
        pet = DigiPal(user_id="test_user", life_stage=LifeStage.CHILD)
        audio_data = b"fake_audio_data"
        
        result = interaction_processor.process_speech_interaction(audio_data, pet)
        
        assert result.success == False
        assert result.result == InteractionResult.FAILURE
        assert "couldn't understand" in result.pet_response
    
    def test_process_speech_interaction_egg_hatching(self, interaction_processor, mock_ai_communication):
        """Test speech interaction with egg triggers special response."""
        pet = DigiPal(user_id="test_user", life_stage=LifeStage.EGG)
        audio_data = b"fake_audio_data"
        
        result = interaction_processor.process_speech_interaction(audio_data, pet)
        
        assert "hatching" in result.pet_response.lower()
        assert "hatching" in result.attribute_changes


class TestDigiPalCore:
    """Test DigiPalCore functionality."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()
        yield temp_file.name
        os.unlink(temp_file.name)
    
    @pytest.fixture
    def mock_storage_manager(self):
        """Create mock storage manager."""
        storage = Mock(spec=StorageManager)
        storage.save_pet.return_value = True
        storage.load_pet.return_value = None
        return storage
    
    @pytest.fixture
    def mock_ai_communication(self):
        """Create mock AI communication."""
        ai_comm = Mock(spec=AICommunication)
        ai_comm.process_interaction.return_value = Interaction(
            user_input="test",
            interpreted_command="eat",
            pet_response="Test response",
            success=True
        )
        ai_comm.process_speech.return_value = "hello"
        ai_comm.memory_manager = Mock()
        ai_comm.memory_manager.get_interaction_summary.return_value = {
            'total_interactions': 5,
            'successful_interactions': 4,
            'success_rate': 0.8
        }
        ai_comm.unload_all_models.return_value = None
        return ai_comm
    
    @pytest.fixture
    def digipal_core(self, mock_storage_manager, mock_ai_communication):
        """Create DigiPalCore with mocked dependencies."""
        return DigiPalCore(mock_storage_manager, mock_ai_communication)
    
    def test_create_new_pet(self, digipal_core, mock_storage_manager):
        """Test creating a new DigiPal."""
        pet = digipal_core.create_new_pet(EggType.RED, "test_user", "TestPal")
        
        assert pet.user_id == "test_user"
        assert pet.name == "TestPal"
        assert pet.egg_type == EggType.RED
        assert pet.life_stage == LifeStage.EGG
        assert pet.offense == 15  # Red egg bonus
        mock_storage_manager.save_pet.assert_called_once_with(pet)
        assert "test_user" in digipal_core.active_pets
    
    def test_create_new_pet_storage_failure(self, digipal_core, mock_storage_manager):
        """Test creating new pet when storage fails."""
        mock_storage_manager.save_pet.return_value = False
        
        with pytest.raises(RuntimeError, match="Failed to save new DigiPal"):
            digipal_core.create_new_pet(EggType.BLUE, "test_user")
    
    def test_load_existing_pet_from_cache(self, digipal_core):
        """Test loading pet from cache."""
        # Add pet to cache
        pet = DigiPal(user_id="test_user", name="CachedPal")
        digipal_core.active_pets["test_user"] = pet
        
        loaded_pet = digipal_core.load_existing_pet("test_user")
        
        assert loaded_pet == pet
        assert loaded_pet.name == "CachedPal"
    
    def test_load_existing_pet_from_storage(self, digipal_core, mock_storage_manager):
        """Test loading pet from storage."""
        pet = DigiPal(user_id="test_user", name="StoredPal")
        mock_storage_manager.load_pet.return_value = pet
        
        loaded_pet = digipal_core.load_existing_pet("test_user")
        
        assert loaded_pet == pet
        assert loaded_pet.name == "StoredPal"
        mock_storage_manager.load_pet.assert_called_once_with("test_user")
        assert "test_user" in digipal_core.active_pets
    
    def test_load_existing_pet_not_found(self, digipal_core, mock_storage_manager):
        """Test loading pet when none exists."""
        mock_storage_manager.load_pet.return_value = None
        
        loaded_pet = digipal_core.load_existing_pet("test_user")
        
        assert loaded_pet is None
    
    def test_get_pet_state(self, digipal_core):
        """Test getting pet state."""
        pet = DigiPal(user_id="test_user", name="TestPal", energy=50)
        digipal_core.active_pets["test_user"] = pet
        
        state = digipal_core.get_pet_state("test_user")
        
        assert state is not None
        assert state.name == "TestPal"
        assert state.energy == 50
    
    def test_get_pet_state_not_found(self, digipal_core):
        """Test getting pet state when no pet exists."""
        state = digipal_core.get_pet_state("nonexistent_user")
        
        assert state is None
    
    def test_process_interaction_success(self, digipal_core, mock_storage_manager):
        """Test processing successful interaction."""
        pet = DigiPal(user_id="test_user", life_stage=LifeStage.CHILD)
        digipal_core.active_pets["test_user"] = pet
        
        success, interaction = digipal_core.process_interaction("test_user", "feed me")
        
        assert success == True
        assert interaction.success == True
        mock_storage_manager.save_pet.assert_called_once_with(pet)
    
    def test_process_interaction_no_pet(self, digipal_core):
        """Test processing interaction when no pet exists."""
        success, interaction = digipal_core.process_interaction("nonexistent_user", "hello")
        
        assert success == False
        assert interaction.success == False
        assert "No DigiPal found" in interaction.pet_response
    
    def test_process_speech_interaction_success(self, digipal_core, mock_storage_manager):
        """Test processing successful speech interaction."""
        pet = DigiPal(user_id="test_user", life_stage=LifeStage.CHILD)
        digipal_core.active_pets["test_user"] = pet
        
        success, interaction = digipal_core.process_speech_interaction("test_user", b"audio_data")
        
        assert success == True
        mock_storage_manager.save_pet.assert_called_once_with(pet)
    
    def test_process_speech_interaction_egg_hatching(self, digipal_core, mock_storage_manager):
        """Test speech interaction with egg triggers hatching."""
        pet = DigiPal(user_id="test_user", life_stage=LifeStage.EGG)
        digipal_core.active_pets["test_user"] = pet
        
        with patch.object(digipal_core.evolution_controller, 'trigger_evolution') as mock_evolution:
            from digipal.core.evolution_controller import EvolutionResult
            mock_evolution.return_value = EvolutionResult(
                success=True,
                old_stage=LifeStage.EGG,
                new_stage=LifeStage.BABY,
                attribute_changes={"hp": 20}
            )
            
            success, interaction = digipal_core.process_speech_interaction("test_user", b"audio_data")
            
            assert success == True
            assert "hatched" in interaction.pet_response.lower()
            mock_evolution.assert_called_once_with(pet, force=True)
    
    def test_update_pet_state_with_changes(self, digipal_core, mock_storage_manager):
        """Test updating pet state when changes occur."""
        pet = DigiPal(user_id="test_user")
        # Set last interaction to 2 hours ago to trigger time decay
        pet.last_interaction = datetime.now() - timedelta(hours=2)
        digipal_core.active_pets["test_user"] = pet
        
        result = digipal_core.update_pet_state("test_user")
        
        assert result == True
        mock_storage_manager.save_pet.assert_called_once_with(pet)
    
    def test_update_pet_state_no_changes(self, digipal_core, mock_storage_manager):
        """Test updating pet state when no changes needed."""
        pet = DigiPal(user_id="test_user")
        # Set last interaction to very recent
        pet.last_interaction = datetime.now()
        digipal_core.active_pets["test_user"] = pet
        
        result = digipal_core.update_pet_state("test_user")
        
        assert result == True
        mock_storage_manager.save_pet.assert_not_called()
    
    def test_update_pet_state_force_save(self, digipal_core, mock_storage_manager):
        """Test updating pet state with forced save."""
        pet = DigiPal(user_id="test_user")
        pet.last_interaction = datetime.now()
        digipal_core.active_pets["test_user"] = pet
        
        result = digipal_core.update_pet_state("test_user", force_save=True)
        
        assert result == True
        mock_storage_manager.save_pet.assert_called_once_with(pet)
    
    def test_trigger_evolution_success(self, digipal_core, mock_storage_manager):
        """Test manually triggering evolution."""
        pet = DigiPal(user_id="test_user", life_stage=LifeStage.BABY)
        digipal_core.active_pets["test_user"] = pet
        
        with patch.object(digipal_core.evolution_controller, 'trigger_evolution') as mock_evolution:
            from digipal.core.evolution_controller import EvolutionResult
            mock_evolution.return_value = EvolutionResult(
                success=True,
                old_stage=LifeStage.BABY,
                new_stage=LifeStage.CHILD
            )
            
            success, result = digipal_core.trigger_evolution("test_user")
            
            assert success == True
            assert result.success == True
            mock_storage_manager.save_pet.assert_called_once_with(pet)
    
    def test_trigger_evolution_no_pet(self, digipal_core):
        """Test triggering evolution when no pet exists."""
        success, result = digipal_core.trigger_evolution("nonexistent_user")
        
        assert success == False
        assert result.success == False
        assert "No DigiPal found" in result.message
    
    def test_get_care_actions(self, digipal_core):
        """Test getting available care actions."""
        pet = DigiPal(user_id="test_user", life_stage=LifeStage.CHILD, energy=50)
        digipal_core.active_pets["test_user"] = pet
        
        with patch.object(digipal_core.attribute_engine, 'get_available_actions') as mock_actions:
            mock_actions.return_value = ["meat", "play", "rest"]
            
            actions = digipal_core.get_care_actions("test_user")
            
            assert actions == ["meat", "play", "rest"]
            mock_actions.assert_called_once_with(pet)
    
    def test_apply_care_action_success(self, digipal_core, mock_storage_manager):
        """Test applying care action successfully."""
        pet = DigiPal(user_id="test_user", energy=50)
        digipal_core.active_pets["test_user"] = pet
        
        with patch.object(digipal_core.attribute_engine, 'apply_care_action') as mock_action:
            mock_action.return_value = (True, Interaction(
                user_input="meat",
                pet_response="Yummy!",
                success=True
            ))
            
            success, interaction = digipal_core.apply_care_action("test_user", "meat")
            
            assert success == True
            assert interaction.success == True
            mock_storage_manager.save_pet.assert_called_once_with(pet)
    
    def test_apply_care_action_no_pet(self, digipal_core):
        """Test applying care action when no pet exists."""
        success, interaction = digipal_core.apply_care_action("nonexistent_user", "meat")
        
        assert success == False
        assert interaction.success == False
        assert "No DigiPal found" in interaction.pet_response
    
    def test_get_pet_statistics(self, digipal_core, mock_ai_communication):
        """Test getting comprehensive pet statistics."""
        pet = DigiPal(
            user_id="test_user",
            name="TestPal",
            life_stage=LifeStage.TEEN,
            generation=2,
            hp=150,
            happiness=75
        )
        digipal_core.active_pets["test_user"] = pet
        
        with patch.object(digipal_core.attribute_engine, 'get_care_quality_assessment') as mock_care:
            mock_care.return_value = {"care_quality": "good", "happiness": "happy"}
            
            with patch.object(digipal_core.evolution_controller, 'check_evolution_eligibility') as mock_evolution:
                mock_evolution.return_value = (True, LifeStage.YOUNG_ADULT, {"min_age": True})
                
                stats = digipal_core.get_pet_statistics("test_user")
                
                assert stats['basic_info']['name'] == "TestPal"
                assert stats['basic_info']['life_stage'] == "teen"
                assert stats['basic_info']['generation'] == 2
                assert stats['attributes']['hp'] == 150
                assert stats['attributes']['happiness'] == 75
                assert stats['care_assessment']['care_quality'] == "good"
                assert stats['interaction_summary']['total_interactions'] == 5
                assert stats['evolution_status']['eligible'] == True
                assert stats['evolution_status']['next_stage'] == "young_adult"
    
    def test_background_updates_start_stop(self, digipal_core):
        """Test starting and stopping background updates."""
        # Start background updates
        digipal_core.start_background_updates()
        assert digipal_core._update_thread is not None
        assert digipal_core._update_thread.is_alive()
        assert digipal_core._stop_updates == False
        
        # Stop background updates
        digipal_core.stop_background_updates()
        assert digipal_core._stop_updates == True
    
    def test_shutdown(self, digipal_core, mock_storage_manager, mock_ai_communication):
        """Test shutting down DigiPalCore."""
        # Add some pets to active cache
        pet1 = DigiPal(user_id="user1")
        pet2 = DigiPal(user_id="user2")
        digipal_core.active_pets["user1"] = pet1
        digipal_core.active_pets["user2"] = pet2
        
        digipal_core.shutdown()
        
        # Check that all pets were saved
        assert mock_storage_manager.save_pet.call_count == 2
        
        # Check that cache was cleared
        assert len(digipal_core.active_pets) == 0
        
        # Check that AI models were unloaded
        mock_ai_communication.unload_all_models.assert_called_once()


class TestDigiPalCoreIntegration:
    """Integration tests with real components."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()
        yield temp_file.name
        os.unlink(temp_file.name)
    
    @pytest.fixture
    def real_storage_manager(self, temp_db_path):
        """Create real storage manager with temporary database."""
        # Ensure database schema is created
        from digipal.storage.database import DatabaseSchema
        DatabaseSchema.create_database(temp_db_path)
        return StorageManager(temp_db_path, "test_assets")
    
    @pytest.fixture
    def mock_ai_communication(self):
        """Create mock AI communication for integration tests."""
        ai_comm = Mock(spec=AICommunication)
        ai_comm.process_interaction.return_value = Interaction(
            user_input="test",
            interpreted_command="eat",
            pet_response="Nom nom!",
            success=True,
            result=InteractionResult.SUCCESS
        )
        ai_comm.process_speech.return_value = "hello"
        ai_comm.memory_manager = Mock()
        ai_comm.memory_manager.get_interaction_summary.return_value = {
            'total_interactions': 0,
            'successful_interactions': 0,
            'success_rate': 0.0
        }
        ai_comm.unload_all_models.return_value = None
        return ai_comm
    
    def test_complete_pet_lifecycle(self, real_storage_manager, mock_ai_communication):
        """Test complete pet lifecycle from creation to interaction."""
        core = DigiPalCore(real_storage_manager, mock_ai_communication)
        
        # Create user first (required for foreign key constraint)
        real_storage_manager.create_user("integration_user", "integration_user", "fake_token")
        
        # Create new pet
        pet = core.create_new_pet(EggType.RED, "integration_user", "IntegrationPal")
        assert pet.life_stage == LifeStage.EGG
        assert pet.name == "IntegrationPal"
        
        # Process speech interaction to trigger hatching
        success, interaction = core.process_speech_interaction("integration_user", b"hello")
        assert success == True
        
        # Reload pet from storage
        core.active_pets.clear()  # Clear cache
        loaded_pet = core.load_existing_pet("integration_user")
        assert loaded_pet is not None
        assert loaded_pet.name == "IntegrationPal"
        
        # Get pet state
        state = core.get_pet_state("integration_user")
        assert state is not None
        assert state.name == "IntegrationPal"
        
        # Apply care action
        success, care_interaction = core.apply_care_action("integration_user", "meat")
        assert success == True
        
        # Get statistics
        stats = core.get_pet_statistics("integration_user")
        assert stats['basic_info']['name'] == "IntegrationPal"
        assert 'attributes' in stats
        assert 'care_assessment' in stats
        
        # Cleanup
        core.shutdown()
    
    def test_pet_persistence_across_sessions(self, real_storage_manager, mock_ai_communication):
        """Test that pet data persists across different core instances."""
        # Create first core instance and pet
        core1 = DigiPalCore(real_storage_manager, mock_ai_communication)
        
        # Create user first (required for foreign key constraint)
        real_storage_manager.create_user("persistence_user", "persistence_user", "fake_token")
        
        pet = core1.create_new_pet(EggType.BLUE, "persistence_user", "PersistentPal")
        pet.happiness = 90  # Modify attribute
        core1.update_pet_state("persistence_user", force_save=True)
        core1.shutdown()
        
        # Create second core instance and load pet
        core2 = DigiPalCore(real_storage_manager, mock_ai_communication)
        loaded_pet = core2.load_existing_pet("persistence_user")
        
        assert loaded_pet is not None
        assert loaded_pet.name == "PersistentPal"
        assert loaded_pet.happiness == 90
        assert loaded_pet.egg_type == EggType.BLUE
        
        core2.shutdown()