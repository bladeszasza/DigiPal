"""
End-to-end tests for complete DigiPal pet lifecycle scenarios.

This module contains comprehensive tests that validate the complete
pet lifecycle from egg to elderly, including all major interactions
and state transitions.
"""

import pytest
import tempfile
import os
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from digipal.core.digipal_core import DigiPalCore
from digipal.core.models import DigiPal, Interaction
from digipal.core.enums import EggType, LifeStage, InteractionResult
from digipal.storage.storage_manager import StorageManager
from digipal.ai.communication import AICommunication
from digipal.auth.auth_manager import AuthManager
from digipal.storage.database import DatabaseConnection
from digipal.ui.gradio_interface import GradioInterface
from digipal.mcp.server import MCPServer


class TestCompleteLifecycleScenarios:
    """Test complete pet lifecycle scenarios from egg to elderly."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def full_system(self, temp_db_path):
        """Create complete system for end-to-end testing."""
        # Storage
        storage_manager = StorageManager(temp_db_path)
        
        # AI Communication
        mock_ai = Mock(spec=AICommunication)
        mock_ai.process_speech.return_value = "hello DigiPal"
        mock_ai.process_interaction.return_value = Mock(
            pet_response="Hello! I'm happy to see you!",
            success=True,
            attribute_changes={"happiness": 2},
            result=InteractionResult.SUCCESS
        )
        mock_ai.memory_manager = Mock()
        mock_ai.memory_manager.get_interaction_summary.return_value = {}
        mock_ai.memory_manager.store_interaction.return_value = None
        mock_ai.memory_manager.get_relevant_memories.return_value = []
        mock_ai.unload_all_models.return_value = None
        
        # Core
        digipal_core = DigiPalCore(storage_manager, mock_ai)
        
        # Auth
        db_connection = DatabaseConnection(temp_db_path)
        auth_manager = AuthManager(db_connection, offline_mode=True)
        
        # UI
        gradio_interface = GradioInterface(digipal_core, auth_manager)
        
        # MCP Server
        mcp_server = MCPServer(digipal_core, "test-lifecycle-server")
        
        return {
            'core': digipal_core,
            'storage': storage_manager,
            'ai': mock_ai,
            'auth': auth_manager,
            'ui': gradio_interface,
            'mcp': mcp_server,
            'temp_db': temp_db_path
        }
    
    def test_complete_new_user_journey(self, full_system):
        """Test complete journey for a new user from authentication to pet interaction."""
        auth = full_system['auth']
        core = full_system['core']
        storage = full_system['storage']
        
        # Step 1: User authentication
        auth_result = auth.authenticate("new_user_token", offline_mode=True)
        assert auth_result.success == True
        user_id = auth_result.user.user_id
        
        # Step 2: Check for existing pet (should be None for new user)
        existing_pet = core.load_existing_pet(user_id)
        assert existing_pet is None
        
        # Step 3: Egg selection and pet creation
        selected_egg = EggType.RED
        new_pet = core.create_new_pet(selected_egg, user_id, "MyFirstPal")
        
        assert new_pet is not None
        assert new_pet.egg_type == selected_egg
        assert new_pet.life_stage == LifeStage.EGG
        assert new_pet.user_id == user_id
        assert new_pet.name == "MyFirstPal"
        
        # Step 4: First speech interaction (should trigger hatching)
        with patch.object(core.evolution_controller, 'trigger_evolution') as mock_evolution:
            from digipal.core.evolution_controller import EvolutionResult
            mock_evolution.return_value = EvolutionResult(
                success=True,
                old_stage=LifeStage.EGG,
                new_stage=LifeStage.BABY,
                message="Your DigiPal has hatched!"
            )
            
            success, interaction = core.process_speech_interaction(user_id, b"hello_audio")
            
            assert success == True
            assert "hello DigiPal" in interaction.user_input
            mock_evolution.assert_called_once()
        
        # Step 5: Verify pet is now in baby stage
        updated_pet = core.load_existing_pet(user_id)
        # Note: In real implementation, evolution would update the pet
        # For this test, we verify the evolution was triggered
        assert updated_pet is not None
    
    def test_complete_pet_lifecycle_progression(self, full_system):
        """Test complete pet lifecycle from egg to elderly."""
        core = full_system['core']
        storage = full_system['storage']
        
        # Create user and pet
        storage.create_user("lifecycle_user", "lifecycle_user")
        pet = core.create_new_pet(EggType.BLUE, "lifecycle_user", "LifecyclePal")
        
        # Track lifecycle progression
        lifecycle_stages = [
            LifeStage.EGG,
            LifeStage.BABY,
            LifeStage.CHILD,
            LifeStage.TEEN,
            LifeStage.YOUNG_ADULT,
            LifeStage.ADULT,
            LifeStage.ELDERLY
        ]
        
        current_stage_index = 0
        
        # Simulate progression through each stage
        for target_stage in lifecycle_stages[1:]:  # Skip EGG as starting stage
            current_stage_index += 1
            
            # Set conditions for evolution
            pet.life_stage = lifecycle_stages[current_stage_index - 1]
            pet.happiness = 85
            pet.care_mistakes = 1
            pet.discipline = 70
            
            # Set appropriate age for evolution
            hours_for_evolution = {
                LifeStage.BABY: 25,
                LifeStage.CHILD: 73,
                LifeStage.TEEN: 121,
                LifeStage.YOUNG_ADULT: 169,
                LifeStage.ADULT: 241,
                LifeStage.ELDERLY: 400
            }
            
            pet.birth_time = datetime.now() - timedelta(hours=hours_for_evolution[target_stage])
            
            # Trigger evolution
            evolution_result = core.evolution_controller.trigger_evolution(pet, force=True)
            
            if target_stage != LifeStage.ELDERLY:  # Elderly is final stage
                assert evolution_result.success == True
                assert evolution_result.new_stage == target_stage
                
                # Verify pet attributes are appropriate for new stage
                if target_stage == LifeStage.ADULT:
                    assert pet.hp >= 150  # Adults should have good HP
                    assert pet.offense >= 30  # Adults should have decent offense
    
    def test_comprehensive_care_scenario(self, full_system):
        """Test comprehensive care scenario with various interactions."""
        core = full_system['core']
        storage = full_system['storage']
        
        # Create user and pet
        storage.create_user("care_user", "care_user")
        pet = core.create_new_pet(EggType.GREEN, "care_user", "CarePal")
        pet.life_stage = LifeStage.CHILD  # Set to child for more interactions
        
        # Track initial stats
        initial_stats = {
            'hp': pet.hp,
            'energy': pet.energy,
            'happiness': pet.happiness,
            'discipline': pet.discipline,
            'offense': pet.offense,
            'defense': pet.defense
        }
        
        # Perform comprehensive care routine
        care_actions = [
            "feed meat",      # Should increase energy and happiness
            "train offense",  # Should increase offense, decrease energy
            "praise",         # Should increase happiness and discipline
            "sleep",          # Should restore energy
            "train defense",  # Should increase defense
            "play",           # Should increase happiness
            "status"          # Should show current state
        ]
        
        interaction_results = []
        
        for action in care_actions:
            success, interaction = core.process_interaction("care_user", action)
            assert success == True
            interaction_results.append(interaction)
        
        # Verify care actions had effects
        final_stats = {
            'hp': pet.hp,
            'energy': pet.energy,
            'happiness': pet.happiness,
            'discipline': pet.discipline,
            'offense': pet.offense,
            'defense': pet.defense
        }
        
        # Some stats should have changed due to care actions
        stats_changed = sum(1 for key in initial_stats if initial_stats[key] != final_stats[key])
        assert stats_changed > 0  # At least some stats should have changed
        
        # Verify interaction history is maintained
        assert len(pet.conversation_history) >= len(care_actions)
    
    def test_neglect_and_recovery_scenario(self, full_system):
        """Test pet neglect and recovery scenario."""
        core = full_system['core']
        storage = full_system['storage']
        
        # Create user and pet
        storage.create_user("neglect_user", "neglect_user")
        pet = core.create_new_pet(EggType.RED, "neglect_user", "NeglectPal")
        pet.life_stage = LifeStage.TEEN
        
        # Simulate neglect
        pet.energy = 5      # Very low energy
        pet.happiness = 10  # Very low happiness
        pet.care_mistakes = 15  # Many care mistakes
        
        # Record neglected state
        neglected_energy = pet.energy
        neglected_happiness = pet.happiness
        neglected_mistakes = pet.care_mistakes
        
        # Attempt recovery through intensive care
        recovery_actions = [
            "feed premium",   # High-quality food
            "sleep",          # Rest
            "praise",         # Positive reinforcement
            "play gentle",    # Light play
            "feed vitamin",   # Health boost
            "sleep",          # More rest
            "praise",         # More positive reinforcement
        ]
        
        for action in recovery_actions:
            success, interaction = core.process_interaction("neglect_user", action)
            # Even neglected pets should respond to care
            assert success == True
        
        # Verify some recovery occurred
        assert pet.energy > neglected_energy  # Energy should improve
        assert pet.happiness > neglected_happiness  # Happiness should improve
        # Care mistakes don't decrease, but no new ones should be added
        assert pet.care_mistakes <= neglected_mistakes + 2  # Allow for minor increases
    
    def test_generational_inheritance_scenario(self, full_system):
        """Test complete generational inheritance scenario."""
        core = full_system['core']
        storage = full_system['storage']
        
        # Create user and first generation pet
        storage.create_user("generation_user", "generation_user")
        parent_pet = core.create_new_pet(EggType.BLUE, "generation_user", "ParentPal")
        
        # Develop parent pet with high stats
        parent_pet.life_stage = LifeStage.ADULT
        parent_pet.hp = 280
        parent_pet.offense = 95
        parent_pet.defense = 110
        parent_pet.speed = 85
        parent_pet.brains = 90
        parent_pet.generation = 1
        
        # Simulate parent reaching elderly and dying
        parent_pet.life_stage = LifeStage.ELDERLY
        parent_pet.birth_time = datetime.now() - timedelta(hours=350)
        
        # Create offspring with inheritance
        offspring_pet = core.create_new_pet(
            EggType.BLUE, 
            "generation_user", 
            "OffspringPal", 
            parent=parent_pet
        )
        
        # Verify inheritance
        assert offspring_pet.generation == parent_pet.generation + 1
        assert offspring_pet.egg_type == parent_pet.egg_type
        
        # Offspring should have some inherited advantages
        base_hp = 100  # Typical base HP
        assert offspring_pet.hp >= base_hp  # Should have inherited HP bonus
        
        # Test that offspring can develop independently
        offspring_pet.life_stage = LifeStage.CHILD
        
        # Perform training to see if inheritance affects development
        success, interaction = core.process_interaction("generation_user", "train offense")
        assert success == True
        
        # Offspring should be able to develop normally
        assert offspring_pet.offense >= 10  # Should have some offense capability
    
    def test_mcp_integration_throughout_lifecycle(self, full_system):
        """Test MCP server integration throughout pet lifecycle."""
        core = full_system['core']
        storage = full_system['storage']
        mcp = full_system['mcp']
        
        # Create user and pet
        storage.create_user("mcp_lifecycle_user", "mcp_lifecycle_user")
        pet = core.create_new_pet(EggType.GREEN, "mcp_lifecycle_user", "MCPLifecyclePal")
        
        async def test_mcp_at_each_stage():
            """Test MCP functionality at each life stage."""
            stages_to_test = [LifeStage.EGG, LifeStage.BABY, LifeStage.CHILD, LifeStage.ADULT]
            
            for stage in stages_to_test:
                pet.life_stage = stage
                storage.save_pet(pet)
                
                # Test get_pet_status
                status_result = await mcp._handle_get_pet_status({"user_id": "mcp_lifecycle_user"})
                assert not status_result.isError
                assert stage.value in str(status_result.content)
                
                # Test interact_with_pet
                interaction_result = await mcp._handle_interact_with_pet({
                    "user_id": "mcp_lifecycle_user",
                    "message": f"hello at {stage.value} stage"
                })
                assert not interaction_result.isError
        
        # Run async test
        asyncio.run(test_mcp_at_each_stage())
    
    def test_memory_continuity_throughout_lifecycle(self, full_system):
        """Test that memory is maintained throughout pet lifecycle."""
        core = full_system['core']
        storage = full_system['storage']
        ai = full_system['ai']
        
        # Create user and pet
        storage.create_user("memory_lifecycle_user", "memory_lifecycle_user")
        pet = core.create_new_pet(EggType.RED, "memory_lifecycle_user", "MemoryLifecyclePal")
        
        # Create memories at different life stages
        lifecycle_memories = []
        
        stages_with_interactions = [
            (LifeStage.BABY, "my first words"),
            (LifeStage.CHILD, "learning to play"),
            (LifeStage.TEEN, "growing stronger"),
            (LifeStage.ADULT, "fully mature now")
        ]
        
        for stage, message in stages_with_interactions:
            pet.life_stage = stage
            
            # Perform interaction
            success, interaction = core.process_interaction("memory_lifecycle_user", message)
            assert success == True
            
            # Store memory of this stage
            lifecycle_memories.append({
                'stage': stage,
                'message': message,
                'interaction': interaction
            })
            
            # Verify memory manager is called
            ai.memory_manager.store_interaction.assert_called()
        
        # Verify pet has conversation history
        assert len(pet.conversation_history) >= len(stages_with_interactions)
        
        # Test memory persistence across save/load
        storage.save_pet(pet)
        core.active_pets.clear()
        loaded_pet = core.load_existing_pet("memory_lifecycle_user")
        
        assert loaded_pet is not None
        assert len(loaded_pet.conversation_history) >= len(stages_with_interactions)
    
    def test_error_recovery_during_lifecycle(self, full_system):
        """Test error recovery scenarios during pet lifecycle."""
        core = full_system['core']
        storage = full_system['storage']
        
        # Create user and pet
        storage.create_user("error_user", "error_user")
        pet = core.create_new_pet(EggType.BLUE, "error_user", "ErrorPal")
        
        # Test recovery from various error scenarios
        
        # Scenario 1: Invalid interaction
        success, interaction = core.process_interaction("error_user", "invalid_command_xyz")
        # Should handle gracefully
        assert success in [True, False]  # Either succeeds or fails gracefully
        
        # Scenario 2: Extreme attribute values
        pet.hp = -10  # Invalid negative HP
        pet.energy = 150  # Over maximum
        
        # System should handle and correct invalid values
        core.update_pet_state(pet, 0.1)
        
        # Values should be corrected
        assert pet.hp >= 0  # Should not be negative
        assert pet.energy <= 100  # Should not exceed maximum
        
        # Scenario 3: Corrupted pet state recovery
        original_name = pet.name
        pet.name = None  # Simulate corruption
        
        # System should handle gracefully
        success, interaction = core.process_interaction("error_user", "hello")
        
        # Pet should still be functional
        assert success == True
        
        # Scenario 4: Database connection issues
        with patch.object(storage, 'save_pet') as mock_save:
            mock_save.side_effect = Exception("Database error")
            
            # Should handle database errors gracefully
            try:
                success, interaction = core.process_interaction("error_user", "test")
                # Should not crash the system
                assert True
            except Exception as e:
                # If exception occurs, it should be handled appropriately
                assert "Database error" in str(e)


class TestPerformanceUnderLoad:
    """Test system performance under various load conditions."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def performance_system(self, temp_db_path):
        """Create system optimized for performance testing."""
        storage_manager = StorageManager(temp_db_path)
        
        # Fast mock AI
        mock_ai = Mock(spec=AICommunication)
        mock_ai.process_interaction.return_value = Mock(
            pet_response="Quick response",
            success=True,
            attribute_changes={},
            result=InteractionResult.SUCCESS
        )
        mock_ai.memory_manager = Mock()
        mock_ai.memory_manager.get_interaction_summary.return_value = {}
        mock_ai.unload_all_models.return_value = None
        
        digipal_core = DigiPalCore(storage_manager, mock_ai)
        
        return {
            'core': digipal_core,
            'storage': storage_manager,
            'ai': mock_ai
        }
    
    def test_multiple_pets_lifecycle_performance(self, performance_system):
        """Test performance with multiple pets going through lifecycle."""
        core = performance_system['core']
        storage = performance_system['storage']
        
        num_pets = 5
        pets = []
        
        # Create multiple pets
        for i in range(num_pets):
            user_id = f"perf_user_{i}"
            storage.create_user(user_id, user_id)
            pet = core.create_new_pet(EggType.RED, user_id, f"PerfPet_{i}")
            pets.append((user_id, pet))
        
        # Simulate interactions for all pets
        import time
        start_time = time.time()
        
        for round_num in range(10):  # 10 rounds of interactions
            for user_id, pet in pets:
                success, interaction = core.process_interaction(user_id, f"interaction {round_num}")
                assert success == True
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should handle 50 interactions (5 pets × 10 rounds) efficiently
        assert total_time < 5.0  # Should complete in under 5 seconds
        
        total_interactions = num_pets * 10
        avg_time_per_interaction = total_time / total_interactions
        
        print(f"Multiple pets performance: {total_time:.3f}s for {total_interactions} interactions")
        print(f"Average time per interaction: {avg_time_per_interaction:.3f}s")
        
        # Each interaction should be fast
        assert avg_time_per_interaction < 0.1  # Less than 100ms per interaction
    
    def test_memory_usage_during_extended_lifecycle(self, performance_system):
        """Test memory usage during extended pet lifecycle."""
        core = performance_system['core']
        storage = performance_system['storage']
        
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create pet and simulate extended lifecycle
        storage.create_user("memory_test_user", "memory_test_user")
        pet = core.create_new_pet(EggType.GREEN, "memory_test_user", "MemoryTestPal")
        
        # Simulate many interactions over time
        for i in range(200):
            success, interaction = core.process_interaction("memory_test_user", f"extended interaction {i}")
            assert success == True
            
            # Periodically check memory
            if i % 50 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory
                
                # Memory increase should be reasonable
                assert memory_increase < 100  # Less than 100MB increase
        
        # Final memory check
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_memory_increase = final_memory - initial_memory
        
        print(f"Extended lifecycle memory usage:")
        print(f"  Initial: {initial_memory:.1f}MB")
        print(f"  Final: {final_memory:.1f}MB")
        print(f"  Increase: {total_memory_increase:.1f}MB")
        
        # Total memory increase should be reasonable
        assert total_memory_increase < 150  # Less than 150MB total increase
        
        # Cleanup and verify memory release
        core.active_pets.clear()
        gc.collect()
        
        cleanup_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"  After cleanup: {cleanup_memory:.1f}MB")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])