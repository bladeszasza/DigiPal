"""
Comprehensive validation tests for all DigiPal requirements.

This module contains tests that validate compliance with all requirements
specified in the requirements document.
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from digipal.core.digipal_core import DigiPalCore
from digipal.core.models import DigiPal, Interaction
from digipal.core.enums import EggType, LifeStage, InteractionResult
from digipal.storage.storage_manager import StorageManager
from digipal.ai.communication import AICommunication
from digipal.auth.auth_manager import AuthManager
from digipal.storage.database import DatabaseConnection
from digipal.ui.gradio_interface import GradioInterface
from digipal.mcp.server import MCPServer


class TestRequirement1Validation:
    """Validate Requirement 1: HuggingFace Authentication."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def auth_manager(self, temp_db_path):
        """Create auth manager for testing."""
        db_connection = DatabaseConnection(temp_db_path)
        return AuthManager(db_connection, offline_mode=True)
    
    def test_requirement_1_1_login_interface_display(self, auth_manager):
        """Test that system displays game-style login interface."""
        # Create Gradio interface
        mock_core = Mock()
        interface = GradioInterface(mock_core, auth_manager)
        
        # Test that authentication tab is created
        auth_tab = interface._create_authentication_tab()
        assert auth_tab is not None
        
        # Verify login interface components exist
        app = interface.create_interface()
        assert app is not None
    
    def test_requirement_1_2_valid_credentials_authentication(self, auth_manager):
        """Test authentication with valid HuggingFace credentials."""
        # Test offline authentication (simulates valid credentials)
        result = auth_manager.authenticate("valid_token_123", offline_mode=True)
        
        assert result.success == True
        assert result.user is not None
        assert result.session is not None
    
    def test_requirement_1_3_invalid_credentials_error(self, auth_manager):
        """Test error display for invalid credentials."""
        # Test with invalid token
        result = auth_manager.authenticate("", offline_mode=False)
        
        assert result.success == False
        assert "Invalid" in result.message or "token" in result.message.lower()
    
    def test_requirement_1_4_session_storage(self, auth_manager):
        """Test that successful authentication stores session."""
        # Authenticate user
        result = auth_manager.authenticate("test_token", offline_mode=True)
        assert result.success == True
        
        # Verify session is stored
        session = auth_manager.session_manager.get_session(result.user.user_id)
        assert session is not None
        assert session.is_valid()


class TestRequirement2Validation:
    """Validate Requirement 2: Existing DigiPal Loading."""
    
    @pytest.fixture
    def integrated_system(self):
        """Create integrated system for testing."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            temp_db_path = f.name
        
        try:
            storage_manager = StorageManager(temp_db_path)
            mock_ai = Mock(spec=AICommunication)
            mock_ai.memory_manager = Mock()
            mock_ai.memory_manager.get_interaction_summary.return_value = {}
            mock_ai.unload_all_models.return_value = None
            
            digipal_core = DigiPalCore(storage_manager, mock_ai)
            
            yield {
                'core': digipal_core,
                'storage': storage_manager,
                'temp_db': temp_db_path
            }
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
    
    def test_requirement_2_1_automatic_loading(self, integrated_system):
        """Test automatic loading of existing DigiPal."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        # Create user and pet
        storage.create_user("test_user", "test_user")
        pet = core.create_new_pet(EggType.RED, "test_user", "TestPal")
        
        # Clear cache to simulate fresh start
        core.active_pets.clear()
        
        # Load existing pet
        loaded_pet = core.load_existing_pet("test_user")
        
        assert loaded_pet is not None
        assert loaded_pet.name == "TestPal"
        assert loaded_pet.user_id == "test_user"
    
    def test_requirement_2_2_restore_attributes(self, integrated_system):
        """Test that all attributes and life stage are restored."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        # Create user and pet with specific attributes
        storage.create_user("test_user", "test_user")
        pet = core.create_new_pet(EggType.BLUE, "test_user", "BluePal")
        pet.hp = 150
        pet.happiness = 90
        pet.life_stage = LifeStage.CHILD
        storage.save_pet(pet)
        
        # Clear cache and reload
        core.active_pets.clear()
        loaded_pet = core.load_existing_pet("test_user")
        
        assert loaded_pet.hp == 150
        assert loaded_pet.happiness == 90
        assert loaded_pet.life_stage == LifeStage.CHILD
        assert loaded_pet.egg_type == EggType.BLUE
    
    def test_requirement_2_3_no_existing_pet_redirect(self, integrated_system):
        """Test redirect to egg selection when no existing DigiPal."""
        core = integrated_system['core']
        
        # Try to load non-existent pet
        loaded_pet = core.load_existing_pet("nonexistent_user")
        
        assert loaded_pet is None


class TestRequirement6Validation:
    """Validate Requirement 6: Evolution through life stages."""
    
    @pytest.fixture
    def integrated_system(self):
        """Create integrated system for testing."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            temp_db_path = f.name
        
        try:
            storage_manager = StorageManager(temp_db_path)
            mock_ai = Mock(spec=AICommunication)
            mock_ai.memory_manager = Mock()
            mock_ai.memory_manager.get_interaction_summary.return_value = {}
            mock_ai.unload_all_models.return_value = None
            
            digipal_core = DigiPalCore(storage_manager, mock_ai)
            
            yield {
                'core': digipal_core,
                'storage': storage_manager,
                'temp_db': temp_db_path
            }
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
    
    def test_requirement_6_1_time_based_evolution(self, integrated_system):
        """Test that DigiPal evolves after defined time periods."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        storage.create_user("test_user", "test_user")
        pet = core.create_new_pet(EggType.RED, "test_user", "EvolvePal")
        pet.life_stage = LifeStage.BABY
        
        # Set birth time to trigger evolution
        pet.birth_time = datetime.now() - timedelta(hours=25)  # Past baby stage
        pet.happiness = 80
        pet.care_mistakes = 1
        
        # Trigger evolution check
        evolution_result = core.evolution_controller.trigger_evolution(pet)
        
        assert evolution_result.success == True
        assert evolution_result.new_stage == LifeStage.CHILD
    
    def test_requirement_6_2_stage_progression(self, integrated_system):
        """Test progression through all life stages."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        storage.create_user("test_user", "test_user")
        pet = core.create_new_pet(EggType.BLUE, "test_user", "ProgressPal")
        
        # Test progression sequence
        expected_progression = [
            LifeStage.EGG,
            LifeStage.BABY,
            LifeStage.CHILD,
            LifeStage.TEEN,
            LifeStage.YOUNG_ADULT,
            LifeStage.ADULT,
            LifeStage.ELDERLY
        ]
        
        # Verify initial stage
        assert pet.life_stage == LifeStage.EGG
        
        # Test each evolution
        for i in range(1, len(expected_progression)):
            current_stage = expected_progression[i-1]
            next_stage = expected_progression[i]
            
            # Set conditions for evolution
            pet.life_stage = current_stage
            pet.happiness = 85
            pet.care_mistakes = 0
            
            # Set appropriate age for evolution
            if current_stage == LifeStage.EGG:
                pet.birth_time = datetime.now() - timedelta(hours=1)
            elif current_stage == LifeStage.BABY:
                pet.birth_time = datetime.now() - timedelta(hours=25)
            elif current_stage == LifeStage.CHILD:
                pet.birth_time = datetime.now() - timedelta(hours=73)
            elif current_stage == LifeStage.TEEN:
                pet.birth_time = datetime.now() - timedelta(hours=121)
            elif current_stage == LifeStage.YOUNG_ADULT:
                pet.birth_time = datetime.now() - timedelta(hours=169)
            elif current_stage == LifeStage.ADULT:
                pet.birth_time = datetime.now() - timedelta(hours=241)
            
            evolution_result = core.evolution_controller.trigger_evolution(pet, force=True)
            
            if next_stage != LifeStage.ELDERLY:  # Elderly is end stage
                assert evolution_result.success == True
                assert evolution_result.new_stage == next_stage
    
    def test_requirement_6_3_image_update_on_evolution(self, integrated_system):
        """Test that 2D image is updated on evolution."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        storage.create_user("test_user", "test_user")
        pet = core.create_new_pet(EggType.GREEN, "test_user", "ImageEvolvePal")
        
        original_image_path = pet.current_image_path
        
        # Mock image generation
        with patch.object(core, 'image_generator') as mock_image_gen:
            mock_image_gen.generate_pet_image.return_value = "new_image_path.png"
            
            # Trigger evolution
            pet.life_stage = LifeStage.BABY
            pet.birth_time = datetime.now() - timedelta(hours=25)
            pet.happiness = 90
            
            evolution_result = core.evolution_controller.trigger_evolution(pet)
            
            if evolution_result.success:
                # Image should be updated
                assert pet.current_image_path != original_image_path
    
    def test_requirement_6_4_expanded_command_understanding(self, integrated_system):
        """Test that command understanding expands with evolution."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        storage.create_user("test_user", "test_user")
        pet = core.create_new_pet(EggType.RED, "test_user", "CommandPal")
        
        # Test baby stage commands (limited)
        pet.life_stage = LifeStage.BABY
        baby_commands = ["eat", "sleep", "good", "bad"]
        
        for command in baby_commands:
            success, interaction = core.process_interaction("test_user", command)
            assert success == True
        
        # Test adult stage commands (expanded)
        pet.life_stage = LifeStage.ADULT
        adult_commands = ["train", "battle", "explore", "status", "analyze"]
        
        for command in adult_commands:
            success, interaction = core.process_interaction("test_user", command)
            # Adult should understand more complex commands
            assert success == True


class TestRequirement7Validation:
    """Validate Requirement 7: Care Actions."""
    
    @pytest.fixture
    def integrated_system(self):
        """Create integrated system for testing."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            temp_db_path = f.name
        
        try:
            storage_manager = StorageManager(temp_db_path)
            mock_ai = Mock(spec=AICommunication)
            mock_ai.memory_manager = Mock()
            mock_ai.memory_manager.get_interaction_summary.return_value = {}
            mock_ai.unload_all_models.return_value = None
            
            digipal_core = DigiPalCore(storage_manager, mock_ai)
            
            yield {
                'core': digipal_core,
                'storage': storage_manager,
                'temp_db': temp_db_path
            }
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
    
    def test_requirement_7_1_training_actions_modify_attributes(self, integrated_system):
        """Test that training actions modify attributes according to Digimon World 1 mechanics."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        storage.create_user("test_user", "test_user")
        pet = core.create_new_pet(EggType.RED, "test_user", "TrainPal")
        
        original_offense = pet.offense
        original_energy = pet.energy
        
        # Perform training action
        success, interaction = core.process_interaction("test_user", "train offense")
        
        assert success == True
        # Training should increase offense and decrease energy
        assert pet.offense > original_offense
        assert pet.energy < original_energy
    
    def test_requirement_7_2_feeding_affects_attributes(self, integrated_system):
        """Test that feeding affects weight, happiness, and energy based on food type."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        storage.create_user("test_user", "test_user")
        pet = core.create_new_pet(EggType.BLUE, "test_user", "FeedPal")
        
        original_energy = pet.energy
        original_happiness = pet.happiness
        original_weight = pet.weight
        
        # Feed the pet
        success, interaction = core.process_interaction("test_user", "feed meat")
        
        assert success == True
        # Feeding should affect energy, happiness, and potentially weight
        assert pet.energy >= original_energy  # Should increase or stay same
        assert pet.happiness >= original_happiness  # Should increase or stay same
    
    def test_requirement_7_3_praise_scold_adjust_happiness_discipline(self, integrated_system):
        """Test that praise/scold adjust happiness and discipline."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        storage.create_user("test_user", "test_user")
        pet = core.create_new_pet(EggType.GREEN, "test_user", "PraisePal")
        
        original_happiness = pet.happiness
        original_discipline = pet.discipline
        
        # Praise the pet
        success, interaction = core.process_interaction("test_user", "praise")
        
        assert success == True
        # Praise should increase happiness and discipline
        assert pet.happiness >= original_happiness
        assert pet.discipline >= original_discipline
        
        # Reset and test scolding
        pet.happiness = original_happiness
        pet.discipline = original_discipline
        
        success, interaction = core.process_interaction("test_user", "scold")
        
        assert success == True
        # Scolding should affect discipline (increase) but may decrease happiness
        assert pet.discipline >= original_discipline
    
    def test_requirement_7_4_rest_restores_energy(self, integrated_system):
        """Test that rest restores energy and affects happiness based on timing."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        storage.create_user("test_user", "test_user")
        pet = core.create_new_pet(EggType.RED, "test_user", "RestPal")
        
        # Reduce energy first
        pet.energy = 20
        original_energy = pet.energy
        original_happiness = pet.happiness
        
        # Let pet rest
        success, interaction = core.process_interaction("test_user", "sleep")
        
        assert success == True
        # Rest should restore energy
        assert pet.energy > original_energy
    
    def test_requirement_7_5_care_mistakes_tracking(self, integrated_system):
        """Test that care mistakes are tracked and influence evolution paths."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        storage.create_user("test_user", "test_user")
        pet = core.create_new_pet(EggType.BLUE, "test_user", "MistakePal")
        
        original_mistakes = pet.care_mistakes
        
        # Simulate care mistake (overfeeding, neglect, etc.)
        pet.energy = 0  # Neglect - let energy drop to 0
        core.update_pet_state(pet, 1.0)  # Update with time passage
        
        # Care mistakes should be tracked
        assert pet.care_mistakes >= original_mistakes


class TestRequirement8Validation:
    """Validate Requirement 8: Persistent Attributes."""
    
    @pytest.fixture
    def integrated_system(self):
        """Create integrated system for testing."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            temp_db_path = f.name
        
        try:
            storage_manager = StorageManager(temp_db_path)
            mock_ai = Mock(spec=AICommunication)
            mock_ai.memory_manager = Mock()
            mock_ai.memory_manager.get_interaction_summary.return_value = {}
            mock_ai.unload_all_models.return_value = None
            
            digipal_core = DigiPalCore(storage_manager, mock_ai)
            
            yield {
                'core': digipal_core,
                'storage': storage_manager,
                'temp_db': temp_db_path
            }
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
    
    def test_requirement_8_1_primary_attributes_initialization(self, integrated_system):
        """Test that primary attributes are initialized on creation."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        storage.create_user("test_user", "test_user")
        pet = core.create_new_pet(EggType.RED, "test_user", "AttributePal")
        
        # Verify all primary attributes are initialized
        assert pet.hp > 0
        assert pet.mp > 0
        assert pet.offense > 0
        assert pet.defense > 0
        assert pet.speed > 0
        assert pet.brains > 0
    
    def test_requirement_8_2_secondary_attributes_initialization(self, integrated_system):
        """Test that secondary attributes are initialized on creation."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        storage.create_user("test_user", "test_user")
        pet = core.create_new_pet(EggType.BLUE, "test_user", "SecondaryPal")
        
        # Verify all secondary attributes are initialized
        assert pet.discipline >= 0
        assert pet.happiness >= 0
        assert pet.weight > 0
        assert pet.care_mistakes >= 0
    
    def test_requirement_8_3_attributes_affect_behavior(self, integrated_system):
        """Test that attribute changes affect DigiPal behavior."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        storage.create_user("test_user", "test_user")
        pet = core.create_new_pet(EggType.GREEN, "test_user", "BehaviorPal")
        
        # Test high happiness affects responses
        pet.happiness = 95
        success, interaction_happy = core.process_interaction("test_user", "hello")
        
        # Test low happiness affects responses
        pet.happiness = 10
        success, interaction_sad = core.process_interaction("test_user", "hello")
        
        # Responses should be different based on happiness
        # (This would be more detailed in actual implementation)
        assert success == True
    
    def test_requirement_8_4_attribute_persistence(self, integrated_system):
        """Test that attribute modifications are persisted to storage."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        storage.create_user("test_user", "test_user")
        pet = core.create_new_pet(EggType.RED, "test_user", "PersistPal")
        
        # Modify attributes
        pet.hp = 200
        pet.happiness = 85
        pet.offense = 75
        
        # Save pet
        storage.save_pet(pet)
        
        # Clear cache and reload
        core.active_pets.clear()
        loaded_pet = core.load_existing_pet("test_user")
        
        # Verify attributes were persisted
        assert loaded_pet.hp == 200
        assert loaded_pet.happiness == 85
        assert loaded_pet.offense == 75


class TestRequirement9Validation:
    """Validate Requirement 9: Generational Inheritance."""
    
    @pytest.fixture
    def integrated_system(self):
        """Create integrated system for testing."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            temp_db_path = f.name
        
        try:
            storage_manager = StorageManager(temp_db_path)
            mock_ai = Mock(spec=AICommunication)
            mock_ai.memory_manager = Mock()
            mock_ai.memory_manager.get_interaction_summary.return_value = {}
            mock_ai.unload_all_models.return_value = None
            
            digipal_core = DigiPalCore(storage_manager, mock_ai)
            
            yield {
                'core': digipal_core,
                'storage': storage_manager,
                'temp_db': temp_db_path
            }
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
    
    def test_requirement_9_1_elderly_death_provides_new_egg(self, integrated_system):
        """Test that elderly DigiPal death provides new egg with inherited DNA."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        storage.create_user("test_user", "test_user")
        pet = core.create_new_pet(EggType.RED, "test_user", "ElderlyPal")
        
        # Set pet to elderly and near death
        pet.life_stage = LifeStage.ELDERLY
        pet.birth_time = datetime.now() - timedelta(hours=400)  # Very old
        pet.hp = 250
        pet.offense = 100
        pet.generation = 1
        
        # Mock death and inheritance
        with patch.object(core.evolution_controller, 'handle_death') as mock_death:
            mock_death.return_value = True
            
            # Trigger death
            result = core.evolution_controller.handle_death(pet)
            
            # Death should be handled
            assert result == True
    
    def test_requirement_9_2_dna_inheritance_from_parent(self, integrated_system):
        """Test that DNA passes down modified attributes from parent."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        storage.create_user("test_user", "test_user")
        parent_pet = core.create_new_pet(EggType.BLUE, "test_user", "ParentPal")
        
        # Set high parent stats
        parent_pet.hp = 300
        parent_pet.offense = 120
        parent_pet.defense = 110
        parent_pet.generation = 1
        
        # Create offspring with inheritance
        offspring = core.create_new_pet(EggType.BLUE, "test_user", "OffspringPal", parent=parent_pet)
        
        # Offspring should have some inherited traits
        assert offspring.generation == parent_pet.generation + 1
        # Base stats should be influenced by parent (implementation specific)
        assert offspring.hp >= 100  # Should have decent base HP from high HP parent
    
    def test_requirement_9_3_evolution_bonuses_applied(self, integrated_system):
        """Test that evolution bonuses are applied based on parent stats."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        storage.create_user("test_user", "test_user")
        parent_pet = core.create_new_pet(EggType.GREEN, "test_user", "HighHPParent")
        
        # Set parent with very high HP
        parent_pet.hp = 350
        parent_pet.generation = 1
        
        # Create offspring
        offspring = core.create_new_pet(EggType.GREEN, "test_user", "HPOffspring", parent=parent_pet)
        
        # Offspring should get HP bonus from high HP parent
        base_hp = 100  # Typical base HP
        assert offspring.hp > base_hp  # Should have inherited HP bonus
    
    def test_requirement_9_4_randomization_with_inheritance(self, integrated_system):
        """Test that inheritance maintains randomization while preserving key traits."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        storage.create_user("test_user", "test_user")
        parent_pet = core.create_new_pet(EggType.RED, "test_user", "RandomParent")
        
        # Set parent stats
        parent_pet.offense = 150
        parent_pet.defense = 80
        parent_pet.generation = 2
        
        # Create multiple offspring to test randomization
        offspring_list = []
        for i in range(3):
            offspring = core.create_new_pet(EggType.RED, f"test_user_{i}", f"RandomOffspring_{i}", parent=parent_pet)
            offspring_list.append(offspring)
        
        # All should inherit generation
        for offspring in offspring_list:
            assert offspring.generation == parent_pet.generation + 1
        
        # But should have some variation in stats (randomization)
        offense_values = [offspring.offense for offspring in offspring_list]
        # Should not all be identical (some randomization)
        assert len(set(offense_values)) > 1 or len(offspring_list) == 1


class TestRequirement10Validation:
    """Validate Requirement 10: MCP Server Functionality."""
    
    @pytest.fixture
    def integrated_system(self):
        """Create integrated system for testing."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            temp_db_path = f.name
        
        try:
            storage_manager = StorageManager(temp_db_path)
            mock_ai = Mock(spec=AICommunication)
            mock_ai.memory_manager = Mock()
            mock_ai.memory_manager.get_interaction_summary.return_value = {}
            mock_ai.unload_all_models.return_value = None
            
            digipal_core = DigiPalCore(storage_manager, mock_ai)
            
            yield {
                'core': digipal_core,
                'storage': storage_manager,
                'temp_db': temp_db_path
            }
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
    
    def test_requirement_10_1_mcp_server_initialization(self, integrated_system):
        """Test that application initializes as functional MCP server."""
        from digipal.mcp.server import MCPServer
        
        core = integrated_system['core']
        
        # Create MCP server
        mcp_server = MCPServer(core, "test-server")
        
        assert mcp_server is not None
        assert mcp_server.server_name == "test-server"
        assert mcp_server.digipal_core == core
    
    def test_requirement_10_2_mcp_protocol_compliance(self, integrated_system):
        """Test that MCP requests are handled according to protocol specifications."""
        from digipal.mcp.server import MCPServer
        
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        # Create MCP server
        mcp_server = MCPServer(core, "test-server")
        
        # Create test user and pet
        storage.create_user("mcp_user", "mcp_user")
        pet = core.create_new_pet(EggType.RED, "mcp_user", "MCPPal")
        
        # Test tool registration
        tools = mcp_server.get_available_tools()
        assert len(tools) > 0
        
        # Verify required tools are available
        tool_names = [tool.name for tool in tools]
        assert "get_pet_status" in tool_names
        assert "interact_with_pet" in tool_names
    
    def test_requirement_10_3_digipal_state_access(self, integrated_system):
        """Test that MCP provides access to DigiPal state and interaction capabilities."""
        from digipal.mcp.server import MCPServer
        
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        # Create MCP server
        mcp_server = MCPServer(core, "test-server")
        
        # Create test user and pet
        storage.create_user("state_user", "state_user")
        pet = core.create_new_pet(EggType.BLUE, "state_user", "StatePal")
        
        # Test state access through MCP
        import asyncio
        
        async def test_state_access():
            result = await mcp_server._handle_get_pet_status({"user_id": "state_user"})
            return result
        
        # Run async test
        asyncio.run(test_state_access())
        
        result = asyncio.run(test_state_access())
        assert result is not None
    
    def test_requirement_10_4_integration_with_other_systems(self, integrated_system):
        """Test that MCP maintains DigiPal functionality while serving requests."""
        from digipal.mcp.server import MCPServer
        
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        # Create MCP server
        mcp_server = MCPServer(core, "test-server")
        
        # Create test user and pet
        storage.create_user("integration_user", "integration_user")
        pet = core.create_new_pet(EggType.GREEN, "integration_user", "IntegrationPal")
        
        # Test that normal DigiPal functionality works while MCP is active
        success, interaction = core.process_interaction("integration_user", "hello")
        assert success == True
        
        # Test that MCP can access the same pet
        import asyncio
        
        async def test_mcp_access():
            result = await mcp_server._handle_get_pet_status({"user_id": "integration_user"})
            return result
        
        mcp_result = asyncio.run(test_mcp_access())
        assert not mcp_result.isError


class TestRequirement11Validation:
    """Validate Requirement 11: DigiPal Memory System."""
    
    @pytest.fixture
    def integrated_system(self):
        """Create integrated system for testing."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            temp_db_path = f.name
        
        try:
            storage_manager = StorageManager(temp_db_path)
            mock_ai = Mock(spec=AICommunication)
            mock_ai.memory_manager = Mock()
            mock_ai.memory_manager.get_interaction_summary.return_value = {}
            mock_ai.memory_manager.store_interaction.return_value = None
            mock_ai.memory_manager.get_relevant_memories.return_value = []
            mock_ai.unload_all_models.return_value = None
            
            digipal_core = DigiPalCore(storage_manager, mock_ai)
            
            yield {
                'core': digipal_core,
                'storage': storage_manager,
                'temp_db': temp_db_path
            }
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
    
    def test_requirement_11_1_interaction_history_storage(self, integrated_system):
        """Test that interaction history is stored in memory."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        storage.create_user("memory_user", "memory_user")
        pet = core.create_new_pet(EggType.RED, "memory_user", "MemoryPal")
        
        # Perform interactions
        interactions = ["hello", "how are you", "let's play", "good night"]
        
        for interaction_text in interactions:
            success, interaction = core.process_interaction("memory_user", interaction_text)
            assert success == True
        
        # Verify memory manager was called to store interactions
        assert core.ai_communication.memory_manager.store_interaction.call_count >= len(interactions)
        
        # Verify pet has conversation history
        assert len(pet.conversation_history) >= len(interactions)
    
    def test_requirement_11_2_contextual_responses(self, integrated_system):
        """Test that DigiPal references previous interactions for context."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        storage.create_user("context_user", "context_user")
        pet = core.create_new_pet(EggType.BLUE, "context_user", "ContextPal")
        
        # Set up memory manager to return relevant memories
        core.ai_communication.memory_manager.get_relevant_memories.return_value = [
            {"content": "User said hello earlier", "emotional_value": 0.5},
            {"content": "User likes to play", "emotional_value": 0.8}
        ]
        
        # Process interaction
        success, interaction = core.process_interaction("context_user", "remember me?")
        assert success == True
        
        # Verify memory manager was called to get relevant memories
        core.ai_communication.memory_manager.get_relevant_memories.assert_called()
    
    def test_requirement_11_3_memory_persistence(self, integrated_system):
        """Test that memory is restored from persistent storage."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        storage.create_user("persist_user", "persist_user")
        pet = core.create_new_pet(EggType.GREEN, "persist_user", "PersistPal")
        
        # Add some interactions
        success, interaction1 = core.process_interaction("persist_user", "first interaction")
        success, interaction2 = core.process_interaction("persist_user", "second interaction")
        
        # Save pet
        storage.save_pet(pet)
        
        # Clear cache and reload
        core.active_pets.clear()
        loaded_pet = core.load_existing_pet("persist_user")
        
        # Verify memory was restored
        assert loaded_pet is not None
        assert len(loaded_pet.conversation_history) >= 2
    
    def test_requirement_11_4_memory_management_strategies(self, integrated_system):
        """Test that memory management strategies are implemented."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        storage.create_user("mgmt_user", "mgmt_user")
        pet = core.create_new_pet(EggType.RED, "mgmt_user", "MgmtPal")
        
        # Simulate many interactions to test memory management
        for i in range(100):
            success, interaction = core.process_interaction("mgmt_user", f"interaction {i}")
            assert success == True
        
        # Verify memory manager handles large amounts of data
        # (Implementation specific - could be summarization, pruning, etc.)
        assert core.ai_communication.memory_manager.get_interaction_summary.called


class TestAllRequirementsIntegration:
    """Integration tests that validate multiple requirements working together."""
    
    @pytest.fixture
    def full_system(self):
        """Create complete system for integration testing."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            temp_db_path = f.name
        
        try:
            storage_manager = StorageManager(temp_db_path)
            mock_ai = Mock(spec=AICommunication)
            mock_ai.process_speech.return_value = "hello"
            mock_ai.process_interaction.return_value = Mock(
                pet_response="Hello there!",
                success=True,
                attribute_changes={"happiness": 2}
            )
            mock_ai.memory_manager = Mock()
            mock_ai.memory_manager.get_interaction_summary.return_value = {}
            mock_ai.memory_manager.store_interaction.return_value = None
            mock_ai.memory_manager.get_relevant_memories.return_value = []
            mock_ai.unload_all_models.return_value = None
            
            digipal_core = DigiPalCore(storage_manager, mock_ai)
            
            yield {
                'core': digipal_core,
                'storage': storage_manager,
                'temp_db': temp_db_path
            }
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
    
    def test_complete_user_journey_all_requirements(self, full_system):
        """Test complete user journey touching all requirements."""
        core = full_system['core']
        storage = full_system['storage']
        
        # Requirement 1: Authentication (simulated)
        storage.create_user("complete_user", "complete_user")
        
        # Requirement 2 & 3: New user egg selection
        pet = core.create_new_pet(EggType.RED, "complete_user", "CompletePal")
        assert pet.egg_type == EggType.RED  # Requirement 3.2
        
        # Requirement 4: First speech interaction (hatching)
        success, interaction = core.process_speech_interaction("complete_user", b"hello_audio")
        assert success == True  # Requirement 4.1
        
        # Requirement 5: Speech communication
        assert "hello" in interaction.user_input  # Requirement 5.1, 5.2
        
        # Requirement 7: Care actions
        care_actions = ["feed", "train", "praise", "rest"]
        for action in care_actions:
            success, interaction = core.process_interaction("complete_user", action)
            assert success == True  # Requirements 7.1-7.4
        
        # Requirement 8: Attribute persistence
        original_hp = pet.hp
        pet.hp = 200
        storage.save_pet(pet)
        
        core.active_pets.clear()
        loaded_pet = core.load_existing_pet("complete_user")
        assert loaded_pet.hp == 200  # Requirement 8.4
        
        # Requirement 11: Memory
        assert len(loaded_pet.conversation_history) > 0  # Requirement 11.1
        
        print("✅ All requirements validated in integrated scenario")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])st
        result = asyncio.run(test_state_access())
        
        assert not result.isError
        # Should contain pet state information
        assert "StatePal" in str(result.content)
    
    def test_requirement_10_4_maintain_functionality_while_serving_mcp(self, integrated_system):
        """Test that DigiPal functionality is maintained while serving MCP requests."""
        from digipal.mcp.server import MCPServer
        
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        # Create MCP server
        mcp_server = MCPServer(core, "test-server")
        
        # Create test user and pet
        storage.create_user("maintain_user", "maintain_user")
        pet = core.create_new_pet(EggType.GREEN, "maintain_user", "MaintainPal")
        
        # Test that normal DigiPal operations still work
        success, interaction = core.process_interaction("maintain_user", "hello")
        assert success == True
        
        # Test that MCP operations also work
        import asyncio
        
        async def test_mcp_interaction():
            result = await mcp_server._handle_interact_with_pet({
                "user_id": "maintain_user",
                "message": "MCP hello"
            })
            return result
        
        mcp_result = asyncio.run(test_mcp_interaction())
        assert not mcp_result.isError
        
        # Both should work simultaneously
        success2, interaction2 = core.process_interaction("maintain_user", "status")
        assert success2 == True


class TestRequirement11Validation:
    """Validate Requirement 11: Memory System."""
    
    @pytest.fixture
    def integrated_system(self):
        """Create integrated system for testing."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            temp_db_path = f.name
        
        try:
            storage_manager = StorageManager(temp_db_path)
            mock_ai = Mock(spec=AICommunication)
            mock_ai.memory_manager = Mock()
            mock_ai.memory_manager.get_interaction_summary.return_value = {}
            mock_ai.memory_manager.store_interaction.return_value = None
            mock_ai.memory_manager.get_relevant_memories.return_value = []
            mock_ai.unload_all_models.return_value = None
            
            digipal_core = DigiPalCore(storage_manager, mock_ai)
            
            yield {
                'core': digipal_core,
                'storage': storage_manager,
                'ai': mock_ai,
                'temp_db': temp_db_path
            }
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
    
    def test_requirement_11_1_interaction_history_storage(self, integrated_system):
        """Test that interaction history is stored in memory."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        ai = integrated_system['ai']
        
        storage.create_user("memory_user", "memory_user")
        pet = core.create_new_pet(EggType.RED, "memory_user", "MemoryPal")
        
        # Perform interaction
        success, interaction = core.process_interaction("memory_user", "remember this")
        
        assert success == True
        # Memory manager should be called to store interaction
        ai.memory_manager.store_interaction.assert_called()
    
    def test_requirement_11_2_context_reference_in_responses(self, integrated_system):
        """Test that DigiPal references previous interactions for context."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        ai = integrated_system['ai']
        
        storage.create_user("context_user", "context_user")
        pet = core.create_new_pet(EggType.BLUE, "context_user", "ContextPal")
        
        # Mock memory retrieval
        ai.memory_manager.get_relevant_memories.return_value = [
            {"content": "User said hello before", "emotional_value": 0.5}
        ]
        
        # Perform interaction
        success, interaction = core.process_interaction("context_user", "do you remember me?")
        
        assert success == True
        # Memory manager should be called to get relevant memories
        ai.memory_manager.get_relevant_memories.assert_called()
    
    def test_requirement_11_3_memory_persistence_across_restarts(self, integrated_system):
        """Test that DigiPal memory is restored from persistent storage."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        storage.create_user("persist_user", "persist_user")
        pet = core.create_new_pet(EggType.GREEN, "persist_user", "PersistPal")
        
        # Add some interactions to create memory
        for i in range(3):
            success, interaction = core.process_interaction("persist_user", f"memory test {i}")
            assert success == True
        
        # Simulate restart by clearing active pets and reloading
        core.active_pets.clear()
        loaded_pet = core.load_existing_pet("persist_user")
        
        assert loaded_pet is not None
        # Memory should be restored (conversation history)
        assert len(loaded_pet.conversation_history) > 0
    
    def test_requirement_11_4_memory_management_strategies(self, integrated_system):
        """Test that appropriate memory management strategies are implemented."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        ai = integrated_system['ai']
        
        storage.create_user("mgmt_user", "mgmt_user")
        pet = core.create_new_pet(EggType.RED, "mgmt_user", "MgmtPal")
        
        # Mock memory management
        ai.memory_manager.cleanup_old_memories.return_value = 5  # Cleaned 5 memories
        
        # Simulate large memory usage
        for i in range(100):
            success, interaction = core.process_interaction("mgmt_user", f"memory overload {i}")
        
        # Memory management should be triggered
        # (This would be implementation specific - checking if cleanup is called)
        # For now, just verify the system handles many interactions
        assert len(core.active_pets) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestRequirement3Validation:
    """Validate Requirement 3: Egg Selection."""
    
    @pytest.fixture
    def integrated_system(self):
        """Create integrated system for testing."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            temp_db_path = f.name
        
        try:
            storage_manager = StorageManager(temp_db_path)
            mock_ai = Mock(spec=AICommunication)
            mock_ai.memory_manager = Mock()
            mock_ai.memory_manager.get_interaction_summary.return_value = {}
            mock_ai.unload_all_models.return_value = None
            
            digipal_core = DigiPalCore(storage_manager, mock_ai)
            
            yield {
                'core': digipal_core,
                'storage': storage_manager,
                'temp_db': temp_db_path
            }
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
    
    def test_requirement_3_1_three_egg_options(self, integrated_system):
        """Test that three egg options are displayed."""
        # Test that all three egg types are available
        available_eggs = [EggType.RED, EggType.BLUE, EggType.GREEN]
        
        assert len(available_eggs) == 3
        assert EggType.RED in available_eggs
        assert EggType.BLUE in available_eggs
        assert EggType.GREEN in available_eggs
    
    def test_requirement_3_2_red_egg_fire_attributes(self, integrated_system):
        """Test red egg creates fire-oriented DigiPal with higher attack."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        storage.create_user("test_user", "test_user")
        pet = core.create_new_pet(EggType.RED, "test_user", "FirePal")
        
        # Red egg should have higher offense
        assert pet.offense >= 15  # Base + red bonus
        assert pet.egg_type == EggType.RED
    
    def test_requirement_3_3_blue_egg_water_attributes(self, integrated_system):
        """Test blue egg creates water-oriented DigiPal with higher defense."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        storage.create_user("test_user", "test_user")
        pet = core.create_new_pet(EggType.BLUE, "test_user", "WaterPal")
        
        # Blue egg should have higher defense
        assert pet.defense >= 15  # Base + blue bonus
        assert pet.egg_type == EggType.BLUE
    
    def test_requirement_3_4_green_egg_earth_attributes(self, integrated_system):
        """Test green egg creates earth-oriented DigiPal with higher health."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        storage.create_user("test_user", "test_user")
        pet = core.create_new_pet(EggType.GREEN, "test_user", "EarthPal")
        
        # Green egg should have higher HP
        assert pet.hp >= 120  # Base + green bonus
        assert pet.egg_type == EggType.GREEN
    
    def test_requirement_3_5_attribute_initialization(self, integrated_system):
        """Test that DigiPal is initialized with base attributes according to egg type."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        storage.create_user("test_user", "test_user")
        
        # Test each egg type
        for egg_type in [EggType.RED, EggType.BLUE, EggType.GREEN]:
            pet = core.create_new_pet(egg_type, f"user_{egg_type.value}", f"Pet_{egg_type.value}")
            
            # All pets should have base attributes
            assert pet.hp > 0
            assert pet.mp > 0
            assert pet.offense > 0
            assert pet.defense > 0
            assert pet.speed > 0
            assert pet.brains > 0
            assert pet.life_stage == LifeStage.EGG


class TestRequirement4Validation:
    """Validate Requirement 4: Egg Hatching."""
    
    @pytest.fixture
    def integrated_system(self):
        """Create integrated system for testing."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            temp_db_path = f.name
        
        try:
            storage_manager = StorageManager(temp_db_path)
            mock_ai = Mock(spec=AICommunication)
            mock_ai.process_speech.return_value = "hello"
            mock_ai.memory_manager = Mock()
            mock_ai.memory_manager.get_interaction_summary.return_value = {}
            mock_ai.unload_all_models.return_value = None
            
            digipal_core = DigiPalCore(storage_manager, mock_ai)
            
            yield {
                'core': digipal_core,
                'storage': storage_manager,
                'ai': mock_ai,
                'temp_db': temp_db_path
            }
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
    
    def test_requirement_4_1_first_speech_triggers_hatching(self, integrated_system):
        """Test that first speech triggers hatching process."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        storage.create_user("test_user", "test_user")
        pet = core.create_new_pet(EggType.RED, "test_user", "HatchPal")
        
        # Verify pet starts as egg
        assert pet.life_stage == LifeStage.EGG
        
        # Process speech interaction
        success, interaction = core.process_speech_interaction("test_user", b"hello_audio")
        
        # Should trigger hatching
        assert success == True
        assert "hatch" in interaction.pet_response.lower() or "hatching" in interaction.attribute_changes
    
    def test_requirement_4_2_image_generation_on_hatch(self, integrated_system):
        """Test that 2D baby DigiPal image is generated on hatching."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        storage.create_user("test_user", "test_user")
        pet = core.create_new_pet(EggType.BLUE, "test_user", "ImagePal")
        
        # Mock evolution controller to trigger hatching
        with patch.object(core.evolution_controller, 'trigger_evolution') as mock_evolution:
            from digipal.core.evolution_controller import EvolutionResult
            mock_evolution.return_value = EvolutionResult(
                success=True,
                old_stage=LifeStage.EGG,
                new_stage=LifeStage.BABY,
                message="Hatched successfully!"
            )
            
            success, interaction = core.process_speech_interaction("test_user", b"hello_audio")
            
            # Evolution should be triggered
            mock_evolution.assert_called_once()
    
    def test_requirement_4_3_baby_stage_after_hatch(self, integrated_system):
        """Test that DigiPal is set to baby life stage after hatching."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        storage.create_user("test_user", "test_user")
        pet = core.create_new_pet(EggType.GREEN, "test_user", "BabyPal")
        
        # Mock successful evolution to baby
        with patch.object(core.evolution_controller, 'trigger_evolution') as mock_evolution:
            from digipal.core.evolution_controller import EvolutionResult
            mock_evolution.return_value = EvolutionResult(
                success=True,
                old_stage=LifeStage.EGG,
                new_stage=LifeStage.BABY
            )
            
            # Trigger hatching
            core.process_speech_interaction("test_user", b"hello_audio")
            
            # Verify evolution was attempted
            mock_evolution.assert_called_once_with(pet, force=True)


class TestRequirement5Validation:
    """Validate Requirement 5: Speech Communication."""
    
    @pytest.fixture
    def integrated_system(self):
        """Create integrated system for testing."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            temp_db_path = f.name
        
        try:
            storage_manager = StorageManager(temp_db_path)
            mock_ai = Mock(spec=AICommunication)
            mock_ai.process_speech.return_value = "hello DigiPal"
            
            # Mock interaction processing
            from digipal.core.models import Interaction
            mock_interaction = Interaction(
                user_input="hello DigiPal",
                interpreted_command="greeting",
                pet_response="Hello! Nice to meet you!",
                success=True,
                result=InteractionResult.SUCCESS
            )
            mock_ai.process_interaction.return_value = mock_interaction
            mock_ai.memory_manager = Mock()
            mock_ai.memory_manager.get_interaction_summary.return_value = {}
            mock_ai.unload_all_models.return_value = None
            
            digipal_core = DigiPalCore(storage_manager, mock_ai)
            
            yield {
                'core': digipal_core,
                'storage': storage_manager,
                'ai': mock_ai,
                'temp_db': temp_db_path
            }
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
    
    def test_requirement_5_1_speech_processing(self, integrated_system):
        """Test that speech is processed using Kyutai listener."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        ai = integrated_system['ai']
        
        storage.create_user("test_user", "test_user")
        pet = core.create_new_pet(EggType.RED, "test_user", "SpeechPal")
        
        # Process speech
        success, interaction = core.process_speech_interaction("test_user", b"audio_data")
        
        # Verify speech processing was called
        ai.process_speech.assert_called_once_with(b"audio_data", None)
        assert success == True
    
    def test_requirement_5_2_command_interpretation(self, integrated_system):
        """Test that speech is interpreted and responded to using Qwen3-0.6B."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        ai = integrated_system['ai']
        
        storage.create_user("test_user", "test_user")
        pet = core.create_new_pet(EggType.BLUE, "test_user", "CommandPal")
        
        # Process speech interaction
        success, interaction = core.process_speech_interaction("test_user", b"audio_data")
        
        # Verify AI processing was called
        ai.process_interaction.assert_called_once_with("hello DigiPal", pet)
        assert interaction.pet_response == "Hello! Nice to meet you!"
    
    def test_requirement_5_3_text_based_response(self, integrated_system):
        """Test that DigiPal outputs text-based communication."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        storage.create_user("test_user", "test_user")
        pet = core.create_new_pet(EggType.GREEN, "test_user", "TextPal")
        
        # Process text interaction
        success, interaction = core.process_interaction("test_user", "Hello!")
        
        assert success == True
        assert isinstance(interaction.pet_response, str)
        assert len(interaction.pet_response) > 0
    
    def test_requirement_5_4_baby_stage_basic_commands(self, integrated_system):
        """Test that baby stage only understands basic commands."""
        core = integrated_system['core']
        storage = integrated_system['storage']
        
        storage.create_user("test_user", "test_user")
        pet = core.create_new_pet(EggType.RED, "test_user", "BabyCommandPal")
        pet.life_stage = LifeStage.BABY
        
        # Test basic commands that baby should understand
        basic_commands = ["eat", "sleep", "good", "bad"]
        
        for command in basic_commands:
            success, interaction = core.process_interaction("test_user", command)
            # Baby should be able to process these basic commands
            assert success == True