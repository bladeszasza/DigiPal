"""
Performance benchmarks and load testing for DigiPal system.

This module contains performance tests to ensure the system meets
performance requirements and can handle expected load.
"""

import pytest
import time
import tempfile
import os
import threading
import concurrent.futures
from datetime import datetime, timedelta
from unittest.mock import Mock
import psutil
import gc

from digipal.core.digipal_core import DigiPalCore
from digipal.core.models import DigiPal
from digipal.core.enums import EggType, LifeStage
from digipal.storage.storage_manager import StorageManager
from digipal.ai.communication import AICommunication
from digipal.mcp.server import MCPServer


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
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
        """Create system for performance testing."""
        storage_manager = StorageManager(temp_db_path)
        
        # Mock AI with fast responses
        mock_ai = Mock(spec=AICommunication)
        mock_ai.process_interaction.return_value = Mock(
            pet_response="Quick response",
            success=True,
            attribute_changes={}
        )
        mock_ai.process_speech.return_value = "quick speech"
        mock_ai.memory_manager = Mock()
        mock_ai.memory_manager.get_interaction_summary.return_value = {}
        mock_ai.unload_all_models.return_value = None
        
        digipal_core = DigiPalCore(storage_manager, mock_ai)
        
        return {
            'core': digipal_core,
            'storage': storage_manager,
            'ai': mock_ai
        }
    
    def test_pet_creation_performance(self, performance_system):
        """Test pet creation performance."""
        core = performance_system['core']
        storage = performance_system['storage']
        
        # Create users first
        for i in range(10):
            storage.create_user(f"user_{i}", f"user_{i}")
        
        # Benchmark pet creation
        start_time = time.time()
        
        for i in range(10):
            pet = core.create_new_pet(EggType.RED, f"user_{i}", f"Pet_{i}")
            assert pet is not None
        
        end_time = time.time()
        creation_time = end_time - start_time
        
        # Should create 10 pets in under 1 second
        assert creation_time < 1.0
        print(f"Pet creation: {creation_time:.3f}s for 10 pets ({creation_time/10:.3f}s per pet)")
    
    def test_interaction_processing_performance(self, performance_system):
        """Test interaction processing performance."""
        core = performance_system['core']
        storage = performance_system['storage']
        
        # Create user and pet
        storage.create_user("interact_user", "interact_user")
        pet = core.create_new_pet(EggType.BLUE, "interact_user", "InteractPal")
        
        # Benchmark interaction processing
        start_time = time.time()
        
        for i in range(50):
            success, interaction = core.process_interaction("interact_user", f"test message {i}")
            assert success == True
        
        end_time = time.time()
        interaction_time = end_time - start_time
        
        # Should process 50 interactions in under 2 seconds
        assert interaction_time < 2.0
        print(f"Interaction processing: {interaction_time:.3f}s for 50 interactions ({interaction_time/50:.3f}s per interaction)")
    
    def test_database_query_performance(self, performance_system):
        """Test database query performance."""
        storage = performance_system['storage']
        
        # Create multiple users and pets
        for i in range(20):
            storage.create_user(f"db_user_{i}", f"db_user_{i}")
            pet = DigiPal(user_id=f"db_user_{i}", name=f"DBPet_{i}")
            storage.save_pet(pet)
        
        # Benchmark pet loading
        start_time = time.time()
        
        for i in range(20):
            pet = storage.load_pet(f"db_user_{i}")
            assert pet is not None
        
        end_time = time.time()
        query_time = end_time - start_time
        
        # Should load 20 pets in under 0.5 seconds
        assert query_time < 0.5
        print(f"Database queries: {query_time:.3f}s for 20 pet loads ({query_time/20:.3f}s per query)")
    
    def test_memory_usage_performance(self, performance_system):
        """Test memory usage during operations."""
        core = performance_system['core']
        storage = performance_system['storage']
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple pets and perform operations
        storage.create_user("memory_user", "memory_user")
        
        pets = []
        for i in range(10):
            pet = core.create_new_pet(EggType.GREEN, f"mem_user_{i}", f"MemPet_{i}")
            pets.append(pet)
            
            # Perform some interactions
            for j in range(5):
                core.process_interaction(f"mem_user_{i}", f"interaction {j}")
        
        # Get peak memory usage
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100
        print(f"Memory usage: {initial_memory:.1f}MB -> {peak_memory:.1f}MB (increase: {memory_increase:.1f}MB)")
        
        # Cleanup and check memory release
        pets.clear()
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Memory after cleanup: {final_memory:.1f}MB")
    
    def test_concurrent_user_performance(self, performance_system):
        """Test performance with concurrent users."""
        core = performance_system['core']
        storage = performance_system['storage']
        
        # Create users
        num_users = 5
        for i in range(num_users):
            storage.create_user(f"concurrent_user_{i}", f"concurrent_user_{i}")
            core.create_new_pet(EggType.RED, f"concurrent_user_{i}", f"ConcurrentPet_{i}")
        
        def user_interactions(user_id):
            """Simulate user interactions."""
            interactions = []
            for i in range(10):
                start = time.time()
                success, interaction = core.process_interaction(user_id, f"message {i}")
                end = time.time()
                interactions.append(end - start)
                assert success == True
            return interactions
        
        # Run concurrent user interactions
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = []
            for i in range(num_users):
                future = executor.submit(user_interactions, f"concurrent_user_{i}")
                futures.append(future)
            
            # Wait for all to complete
            results = []
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should handle 5 concurrent users with 10 interactions each in under 5 seconds
        assert total_time < 5.0
        print(f"Concurrent users: {total_time:.3f}s for {num_users} users with 10 interactions each")
        
        # Check individual interaction times
        all_interaction_times = [time for user_times in results for time in user_times]
        avg_interaction_time = sum(all_interaction_times) / len(all_interaction_times)
        max_interaction_time = max(all_interaction_times)
        
        print(f"Average interaction time: {avg_interaction_time:.3f}s")
        print(f"Max interaction time: {max_interaction_time:.3f}s")
        
        # No single interaction should take more than 1 second
        assert max_interaction_time < 1.0


class TestLoadTesting:
    """Load testing for system scalability."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def load_test_system(self, temp_db_path):
        """Create system for load testing."""
        storage_manager = StorageManager(temp_db_path)
        
        # Mock AI with realistic response times
        mock_ai = Mock(spec=AICommunication)
        
        def mock_process_interaction(text, pet):
            # Simulate some processing time
            time.sleep(0.01)  # 10ms processing time
            return Mock(
                pet_response=f"Response to: {text}",
                success=True,
                attribute_changes={"happiness": 1}
            )
        
        mock_ai.process_interaction.side_effect = mock_process_interaction
        mock_ai.process_speech.return_value = "speech result"
        mock_ai.memory_manager = Mock()
        mock_ai.memory_manager.get_interaction_summary.return_value = {}
        mock_ai.unload_all_models.return_value = None
        
        digipal_core = DigiPalCore(storage_manager, mock_ai)
        
        return {
            'core': digipal_core,
            'storage': storage_manager,
            'ai': mock_ai
        }
    
    def test_high_volume_interactions(self, load_test_system):
        """Test system under high volume of interactions."""
        core = load_test_system['core']
        storage = load_test_system['storage']
        
        # Create user and pet
        storage.create_user("load_user", "load_user")
        pet = core.create_new_pet(EggType.BLUE, "load_user", "LoadPal")
        
        # High volume interaction test
        num_interactions = 100
        start_time = time.time()
        
        successful_interactions = 0
        failed_interactions = 0
        
        for i in range(num_interactions):
            try:
                success, interaction = core.process_interaction("load_user", f"load test {i}")
                if success:
                    successful_interactions += 1
                else:
                    failed_interactions += 1
            except Exception as e:
                failed_interactions += 1
                print(f"Interaction {i} failed: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate metrics
        success_rate = successful_interactions / num_interactions
        avg_time_per_interaction = total_time / num_interactions
        interactions_per_second = num_interactions / total_time
        
        print(f"High volume test results:")
        print(f"  Total interactions: {num_interactions}")
        print(f"  Successful: {successful_interactions}")
        print(f"  Failed: {failed_interactions}")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Avg time per interaction: {avg_time_per_interaction:.3f}s")
        print(f"  Interactions per second: {interactions_per_second:.1f}")
        
        # Assertions
        assert success_rate >= 0.95  # At least 95% success rate
        assert avg_time_per_interaction < 0.1  # Less than 100ms per interaction
        assert interactions_per_second >= 10  # At least 10 interactions per second
    
    def test_sustained_load(self, load_test_system):
        """Test system under sustained load."""
        core = load_test_system['core']
        storage = load_test_system['storage']
        
        # Create multiple users
        num_users = 3
        for i in range(num_users):
            storage.create_user(f"sustained_user_{i}", f"sustained_user_{i}")
            core.create_new_pet(EggType.GREEN, f"sustained_user_{i}", f"SustainedPet_{i}")
        
        def sustained_user_load(user_id, duration_seconds):
            """Generate sustained load for a user."""
            end_time = time.time() + duration_seconds
            interaction_count = 0
            successful_count = 0
            
            while time.time() < end_time:
                try:
                    success, interaction = core.process_interaction(user_id, f"sustained {interaction_count}")
                    interaction_count += 1
                    if success:
                        successful_count += 1
                    
                    # Small delay to simulate realistic usage
                    time.sleep(0.05)  # 50ms between interactions
                    
                except Exception as e:
                    print(f"Error in sustained load for {user_id}: {e}")
                    interaction_count += 1
            
            return {
                'user_id': user_id,
                'total_interactions': interaction_count,
                'successful_interactions': successful_count,
                'success_rate': successful_count / interaction_count if interaction_count > 0 else 0
            }
        
        # Run sustained load test for 10 seconds
        test_duration = 10
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = []
            for i in range(num_users):
                future = executor.submit(sustained_user_load, f"sustained_user_{i}", test_duration)
                futures.append(future)
            
            # Wait for all to complete
            results = []
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        
        end_time = time.time()
        actual_duration = end_time - start_time
        
        # Analyze results
        total_interactions = sum(r['total_interactions'] for r in results)
        total_successful = sum(r['successful_interactions'] for r in results)
        overall_success_rate = total_successful / total_interactions if total_interactions > 0 else 0
        
        print(f"Sustained load test results:")
        print(f"  Duration: {actual_duration:.1f}s")
        print(f"  Users: {num_users}")
        print(f"  Total interactions: {total_interactions}")
        print(f"  Successful interactions: {total_successful}")
        print(f"  Overall success rate: {overall_success_rate:.2%}")
        print(f"  Interactions per second: {total_interactions / actual_duration:.1f}")
        
        for result in results:
            print(f"  {result['user_id']}: {result['total_interactions']} interactions, {result['success_rate']:.2%} success")
        
        # Assertions
        assert overall_success_rate >= 0.90  # At least 90% success rate under sustained load
        assert total_interactions >= num_users * 100  # Each user should complete at least 100 interactions
    
    def test_memory_stability_under_load(self, load_test_system):
        """Test memory stability under extended load."""
        core = load_test_system['core']
        storage = load_test_system['storage']
        
        # Create user and pet
        storage.create_user("memory_load_user", "memory_load_user")
        pet = core.create_new_pet(EggType.RED, "memory_load_user", "MemoryLoadPal")
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_samples = [initial_memory]
        
        # Run interactions and sample memory usage
        num_interactions = 200
        sample_interval = 20  # Sample every 20 interactions
        
        for i in range(num_interactions):
            success, interaction = core.process_interaction("memory_load_user", f"memory test {i}")
            assert success == True
            
            if i % sample_interval == 0:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_samples.append(current_memory)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_samples.append(final_memory)
        
        # Analyze memory usage
        max_memory = max(memory_samples)
        memory_growth = final_memory - initial_memory
        
        print(f"Memory stability test:")
        print(f"  Initial memory: {initial_memory:.1f}MB")
        print(f"  Final memory: {final_memory:.1f}MB")
        print(f"  Max memory: {max_memory:.1f}MB")
        print(f"  Memory growth: {memory_growth:.1f}MB")
        print(f"  Memory samples: {[f'{m:.1f}' for m in memory_samples]}")
        
        # Assertions
        assert memory_growth < 50  # Memory growth should be less than 50MB
        assert max_memory < initial_memory + 100  # Peak memory should not exceed initial + 100MB


class TestMCPServerPerformance:
    """Performance tests for MCP server functionality."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def mcp_performance_system(self, temp_db_path):
        """Create MCP system for performance testing."""
        storage_manager = StorageManager(temp_db_path)
        
        mock_ai = Mock(spec=AICommunication)
        mock_ai.process_interaction.return_value = Mock(
            pet_response="MCP response",
            success=True,
            attribute_changes={}
        )
        mock_ai.memory_manager = Mock()
        mock_ai.memory_manager.get_interaction_summary.return_value = {}
        mock_ai.unload_all_models.return_value = None
        
        digipal_core = DigiPalCore(storage_manager, mock_ai)
        mcp_server = MCPServer(digipal_core, "perf-test-server")
        
        return {
            'core': digipal_core,
            'storage': storage_manager,
            'mcp': mcp_server
        }
    
    @pytest.mark.asyncio
    async def test_mcp_tool_call_performance(self, mcp_performance_system):
        """Test MCP tool call performance."""
        mcp_server = mcp_performance_system['mcp']
        storage = mcp_performance_system['storage']
        core = mcp_performance_system['core']
        
        # Create test user and pet
        storage.create_user("mcp_perf_user", "mcp_perf_user")
        pet = core.create_new_pet(EggType.BLUE, "mcp_perf_user", "MCPPerfPal")
        
        # Benchmark get_pet_status calls
        start_time = time.time()
        
        for i in range(20):
            result = await mcp_server._handle_get_pet_status({"user_id": "mcp_perf_user"})
            assert not result.isError
        
        end_time = time.time()
        status_time = end_time - start_time
        
        # Benchmark interact_with_pet calls
        start_time = time.time()
        
        for i in range(20):
            result = await mcp_server._handle_interact_with_pet({
                "user_id": "mcp_perf_user",
                "message": f"performance test {i}"
            })
            assert not result.isError
        
        end_time = time.time()
        interaction_time = end_time - start_time
        
        print(f"MCP Performance:")
        print(f"  Status calls: {status_time:.3f}s for 20 calls ({status_time/20:.3f}s per call)")
        print(f"  Interaction calls: {interaction_time:.3f}s for 20 calls ({interaction_time/20:.3f}s per call)")
        
        # Assertions
        assert status_time < 1.0  # 20 status calls in under 1 second
        assert interaction_time < 2.0  # 20 interaction calls in under 2 seconds


class TestScalabilityBenchmarks:
    """Advanced scalability and stress testing."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def scalability_system(self, temp_db_path):
        """Create system for scalability testing."""
        storage_manager = StorageManager(temp_db_path)
        
        mock_ai = Mock(spec=AICommunication)
        mock_ai.process_interaction.return_value = Mock(
            pet_response="Scalable response",
            success=True,
            attribute_changes={}
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
    
    def test_large_scale_pet_creation(self, scalability_system):
        """Test creating large numbers of pets."""
        core = scalability_system['core']
        storage = scalability_system['storage']
        
        num_pets = 100
        start_time = time.time()
        
        created_pets = []
        for i in range(num_pets):
            user_id = f"scale_user_{i}"
            storage.create_user(user_id, user_id)
            pet = core.create_new_pet(EggType.RED, user_id, f"ScalePet_{i}")
            created_pets.append(pet)
        
        end_time = time.time()
        creation_time = end_time - start_time
        
        print(f"Large scale creation: {creation_time:.3f}s for {num_pets} pets")
        print(f"Average creation time: {creation_time/num_pets:.3f}s per pet")
        
        # Should create 100 pets in reasonable time
        assert creation_time < 10.0  # Less than 10 seconds
        assert len(created_pets) == num_pets
    
    def test_database_performance_under_load(self, scalability_system):
        """Test database performance with many concurrent operations."""
        storage = scalability_system['storage']
        
        # Create many pets first
        num_pets = 50
        for i in range(num_pets):
            storage.create_user(f"db_load_user_{i}", f"db_load_user_{i}")
            pet = DigiPal(user_id=f"db_load_user_{i}", name=f"DBLoadPet_{i}")
            storage.save_pet(pet)
        
        # Test concurrent database operations
        def database_operations():
            """Perform database operations."""
            operations = []
            for i in range(10):
                start = time.time()
                pet = storage.load_pet(f"db_load_user_{i % num_pets}")
                end = time.time()
                operations.append(end - start)
                assert pet is not None
            return operations
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(database_operations) for _ in range(5)]
            results = [future.result() for future in futures]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Analyze results
        all_operation_times = [time for result in results for time in result]
        avg_operation_time = sum(all_operation_times) / len(all_operation_times)
        max_operation_time = max(all_operation_times)
        
        print(f"Database load test: {total_time:.3f}s total")
        print(f"Average operation time: {avg_operation_time:.3f}s")
        print(f"Max operation time: {max_operation_time:.3f}s")
        
        # Database operations should be fast even under load
        assert avg_operation_time < 0.1  # Less than 100ms average
        assert max_operation_time < 0.5   # Less than 500ms max
    
    def test_memory_efficiency_with_many_pets(self, scalability_system):
        """Test memory efficiency with many active pets."""
        core = scalability_system['core']
        storage = scalability_system['storage']
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and activate many pets
        num_pets = 20
        for i in range(num_pets):
            user_id = f"memory_eff_user_{i}"
            storage.create_user(user_id, user_id)
            pet = core.create_new_pet(EggType.BLUE, user_id, f"MemEffPet_{i}")
            
            # Perform some interactions to populate memory
            for j in range(5):
                core.process_interaction(user_id, f"interaction {j}")
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_per_pet = (peak_memory - initial_memory) / num_pets
        
        print(f"Memory efficiency test:")
        print(f"  Initial memory: {initial_memory:.1f}MB")
        print(f"  Peak memory: {peak_memory:.1f}MB")
        print(f"  Memory per pet: {memory_per_pet:.1f}MB")
        
        # Memory usage per pet should be reasonable
        assert memory_per_pet < 10  # Less than 10MB per pet
        
        # Test cleanup
        core.active_pets.clear()
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"  After cleanup: {final_memory:.1f}MB")


class TestRealWorldScenarios:
    """Test realistic usage scenarios and edge cases."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def realistic_system(self, temp_db_path):
        """Create system with realistic response times."""
        storage_manager = StorageManager(temp_db_path)
        
        mock_ai = Mock(spec=AICommunication)
        
        def realistic_interaction(text, pet):
            # Simulate realistic AI processing time
            time.sleep(0.05)  # 50ms processing
            return Mock(
                pet_response=f"Realistic response to: {text}",
                success=True,
                attribute_changes={"happiness": 1}
            )
        
        mock_ai.process_interaction.side_effect = realistic_interaction
        mock_ai.memory_manager = Mock()
        mock_ai.memory_manager.get_interaction_summary.return_value = {}
        mock_ai.unload_all_models.return_value = None
        
        digipal_core = DigiPalCore(storage_manager, mock_ai)
        
        return {
            'core': digipal_core,
            'storage': storage_manager,
            'ai': mock_ai
        }
    
    def test_typical_user_session(self, realistic_system):
        """Test a typical user session with realistic interactions."""
        core = realistic_system['core']
        storage = realistic_system['storage']
        
        # Create user and pet
        storage.create_user("typical_user", "typical_user")
        pet = core.create_new_pet(EggType.GREEN, "typical_user", "TypicalPal")
        
        # Simulate typical user session (10-15 interactions)
        typical_interactions = [
            "hello", "how are you", "let's play", "train offense",
            "feed", "good job", "status", "train defense", "rest",
            "play again", "praise", "check stats", "goodbye"
        ]
        
        session_start = time.time()
        successful_interactions = 0
        
        for interaction_text in typical_interactions:
            try:
                success, interaction = core.process_interaction("typical_user", interaction_text)
                if success:
                    successful_interactions += 1
            except Exception as e:
                print(f"Error in typical session: {e}")
        
        session_end = time.time()
        session_duration = session_end - session_start
        
        success_rate = successful_interactions / len(typical_interactions)
        avg_response_time = session_duration / len(typical_interactions)
        
        print(f"Typical user session:")
        print(f"  Duration: {session_duration:.2f}s")
        print(f"  Interactions: {len(typical_interactions)}")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Avg response time: {avg_response_time:.3f}s")
        
        # Realistic expectations for user experience
        assert success_rate >= 0.95  # 95% success rate
        assert avg_response_time < 0.2  # Less than 200ms average response
        assert session_duration < 5.0   # Complete session under 5 seconds
    
    def test_long_running_session(self, realistic_system):
        """Test system stability during long-running sessions."""
        core = realistic_system['core']
        storage = realistic_system['storage']
        
        # Create user and pet
        storage.create_user("long_user", "long_user")
        pet = core.create_new_pet(EggType.RED, "long_user", "LongPal")
        
        # Simulate long session (100 interactions over time)
        num_interactions = 100
        start_time = time.time()
        
        response_times = []
        memory_samples = []
        process = psutil.Process()
        
        for i in range(num_interactions):
            interaction_start = time.time()
            
            success, interaction = core.process_interaction("long_user", f"long interaction {i}")
            
            interaction_end = time.time()
            response_times.append(interaction_end - interaction_start)
            
            # Sample memory every 20 interactions
            if i % 20 == 0:
                memory_mb = process.memory_info().rss / 1024 / 1024
                memory_samples.append(memory_mb)
            
            assert success == True
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Analyze performance over time
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        min_response_time = min(response_times)
        
        # Check for performance degradation
        early_responses = response_times[:20]
        late_responses = response_times[-20:]
        early_avg = sum(early_responses) / len(early_responses)
        late_avg = sum(late_responses) / len(late_responses)
        
        print(f"Long running session:")
        print(f"  Total duration: {total_duration:.2f}s")
        print(f"  Avg response time: {avg_response_time:.3f}s")
        print(f"  Min/Max response time: {min_response_time:.3f}s / {max_response_time:.3f}s")
        print(f"  Early vs Late avg: {early_avg:.3f}s vs {late_avg:.3f}s")
        print(f"  Memory samples: {[f'{m:.1f}MB' for m in memory_samples]}")
        
        # Performance should remain stable
        assert avg_response_time < 0.15  # Average under 150ms
        assert max_response_time < 0.5   # No response over 500ms
        assert late_avg < early_avg * 1.5  # No more than 50% degradation
        
        # Memory should not grow excessively
        if len(memory_samples) > 1:
            memory_growth = memory_samples[-1] - memory_samples[0]
            assert memory_growth < 50  # Less than 50MB growth


if __name__ == "__main__":
    pytest.main([__file__, "-v"])