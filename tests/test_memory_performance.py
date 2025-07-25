"""
Tests for memory management and performance optimization features.
"""

import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from digipal.core.memory_manager import (
    EmotionalMemory, MemoryCache, SimpleRAG, EnhancedMemoryManager
)
from digipal.core.performance_optimizer import (
    LazyModelLoader, BackgroundTaskManager, DatabaseOptimizer,
    PerformanceMonitor, ResourceCleanupManager,
    ModelLoadingConfig, BackgroundTaskConfig
)
from digipal.core.models import DigiPal, Interaction
from digipal.core.enums import EggType, LifeStage, InteractionResult
from digipal.storage.storage_manager import StorageManager


class TestEmotionalMemory:
    """Test emotional memory functionality."""
    
    def test_emotional_memory_creation(self):
        """Test creating emotional memory."""
        memory = EmotionalMemory(
            id="test_memory_1",
            timestamp=datetime.now(),
            content="User praised the pet",
            memory_type="interaction",
            emotional_value=0.8,
            importance=0.7,
            tags={"praise", "positive"}
        )
        
        assert memory.id == "test_memory_1"
        assert memory.emotional_value == 0.8
        assert memory.importance == 0.7
        assert "praise" in memory.tags
        assert memory.access_count == 0
    
    def test_emotional_memory_serialization(self):
        """Test memory serialization and deserialization."""
        original = EmotionalMemory(
            id="test_memory_2",
            timestamp=datetime.now(),
            content="Pet was fed",
            memory_type="action",
            emotional_value=0.3,
            importance=0.5,
            tags={"eating", "care"}
        )
        
        # Serialize to dict
        data = original.to_dict()
        
        # Deserialize from dict
        restored = EmotionalMemory.from_dict(data)
        
        assert restored.id == original.id
        assert restored.content == original.content
        assert restored.emotional_value == original.emotional_value
        assert restored.tags == original.tags


class TestMemoryCache:
    """Test memory cache functionality."""
    
    def test_cache_basic_operations(self):
        """Test basic cache operations."""
        cache = MemoryCache(max_size=3, ttl_seconds=1)
        
        # Test put and get
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Test cache miss
        assert cache.get("nonexistent") is None
        
        # Test cache size
        assert cache.size() == 1
        
        cache.shutdown()
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = MemoryCache(max_size=2, ttl_seconds=10)
        
        # Fill cache
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add third item, should evict key2
        cache.put("key3", "value3")
        
        assert cache.get("key1") == "value1"  # Should still exist
        assert cache.get("key2") is None      # Should be evicted
        assert cache.get("key3") == "value3"  # Should exist
        
        cache.shutdown()
    
    def test_cache_ttl_expiration(self):
        """Test TTL-based expiration."""
        cache = MemoryCache(max_size=10, ttl_seconds=0.1)  # 100ms TTL
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(0.2)
        
        assert cache.get("key1") is None  # Should be expired
        
        cache.shutdown()


class TestSimpleRAG:
    """Test simple RAG functionality."""
    
    def setup_method(self):
        """Set up test data."""
        self.rag = SimpleRAG(max_context_memories=3)
        
        # Create test memories
        self.memories = [
            EmotionalMemory(
                id="mem1",
                timestamp=datetime.now() - timedelta(hours=1),
                content="User said eat and pet was fed",
                memory_type="interaction",
                emotional_value=0.5,
                importance=0.6,
                tags={"eating", "care"}
            ),
            EmotionalMemory(
                id="mem2",
                timestamp=datetime.now() - timedelta(hours=2),
                content="Pet was praised for good behavior",
                memory_type="interaction",
                emotional_value=0.8,
                importance=0.7,
                tags={"praise", "positive"}
            ),
            EmotionalMemory(
                id="mem3",
                timestamp=datetime.now() - timedelta(hours=3),
                content="Training session completed successfully",
                memory_type="action",
                emotional_value=0.4,
                importance=0.8,
                tags={"training", "exercise"}
            )
        ]
    
    def test_retrieve_relevant_memories(self):
        """Test retrieving relevant memories."""
        # Query about eating
        relevant = self.rag.retrieve_relevant_memories(
            "I want to feed my pet",
            self.memories,
            {}
        )
        
        assert len(relevant) > 0
        # Should prioritize eating-related memory
        assert any("eat" in mem.content.lower() for mem in relevant)
    
    def test_retrieve_with_emotional_relevance(self):
        """Test that emotional memories are prioritized."""
        # Query that should match the high-emotion praise memory
        relevant = self.rag.retrieve_relevant_memories(
            "good job well done",
            self.memories,
            {}
        )
        
        assert len(relevant) > 0
        # Should include the praise memory due to high emotional value
        praise_memory = next((m for m in relevant if "praised" in m.content), None)
        assert praise_memory is not None
    
    def test_retrieve_with_recency_bias(self):
        """Test that recent memories are prioritized."""
        # All memories should be returned, but most recent first
        relevant = self.rag.retrieve_relevant_memories(
            "pet interaction",
            self.memories,
            {}
        )
        
        # Should return memories, with recent ones having higher scores
        assert len(relevant) <= 3  # Max context memories
        if len(relevant) > 1:
            # More recent memories should generally score higher
            timestamps = [m.timestamp for m in relevant]
            # At least some ordering by recency should be present
            assert any(timestamps[i] >= timestamps[i+1] for i in range(len(timestamps)-1))


class TestEnhancedMemoryManager:
    """Test enhanced memory manager."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_storage = Mock(spec=StorageManager)
        self.memory_manager = EnhancedMemoryManager(
            self.mock_storage,
            cache_size=100,
            max_memories_per_pet=50
        )
        
        # Create test pet
        self.test_pet = DigiPal(
            id="test_pet_1",
            user_id="test_user_1",
            name="TestPal",
            egg_type=EggType.RED,
            life_stage=LifeStage.CHILD
        )
    
    def test_add_memory(self):
        """Test adding memory."""
        memory_id = self.memory_manager.add_memory(
            pet_id=self.test_pet.id,
            content="Test memory content",
            memory_type="interaction",
            emotional_value=0.5,
            importance=0.6,
            tags={"test"}
        )
        
        assert memory_id is not None
        assert len(self.memory_manager.pet_memories[self.test_pet.id]) == 1
        
        memory = self.memory_manager.pet_memories[self.test_pet.id][0]
        assert memory.content == "Test memory content"
        assert memory.emotional_value == 0.5
    
    def test_add_interaction_memory(self):
        """Test adding interaction memory with emotional context."""
        interaction = Interaction(
            timestamp=datetime.now(),
            user_input="good job",
            interpreted_command="praise",
            pet_response="Thank you!",
            attribute_changes={"happiness": 10},
            success=True,
            result=InteractionResult.SUCCESS
        )
        
        memory_id = self.memory_manager.add_interaction_memory(self.test_pet, interaction)
        
        assert memory_id is not None
        memories = self.memory_manager.pet_memories[self.test_pet.id]
        assert len(memories) == 1
        
        memory = memories[0]
        assert memory.memory_type == "interaction"
        assert memory.emotional_value > 0  # Should be positive for praise
        assert "praise" in memory.tags
    
    def test_get_relevant_memories(self):
        """Test retrieving relevant memories."""
        # Add some test memories
        self.memory_manager.add_memory(
            self.test_pet.id, "Pet was fed", "action", 0.3, 0.5, {"eating"}
        )
        self.memory_manager.add_memory(
            self.test_pet.id, "Pet was trained", "action", 0.2, 0.7, {"training"}
        )
        
        # Query for eating-related memories
        relevant = self.memory_manager.get_relevant_memories(
            self.test_pet.id, "I want to feed my pet"
        )
        
        assert len(relevant) > 0
        assert any("fed" in mem.content for mem in relevant)
    
    def test_emotional_state_summary(self):
        """Test emotional state summary calculation."""
        # Add memories with different emotional values
        self.memory_manager.add_memory(
            self.test_pet.id, "Happy memory", "interaction", 0.8, 0.5
        )
        self.memory_manager.add_memory(
            self.test_pet.id, "Sad memory", "interaction", -0.6, 0.5
        )
        self.memory_manager.add_memory(
            self.test_pet.id, "Neutral memory", "interaction", 0.0, 0.5
        )
        
        summary = self.memory_manager.get_emotional_state_summary(self.test_pet.id)
        
        assert "overall_mood" in summary
        assert "recent_trend" in summary
        assert "positive_memories" in summary
        assert "negative_memories" in summary
    
    def test_memory_size_management(self):
        """Test that memory size is managed properly."""
        # Add more memories than the limit
        for i in range(60):  # More than max_memories_per_pet (50)
            self.memory_manager.add_memory(
                self.test_pet.id,
                f"Memory {i}",
                "interaction",
                0.1,
                0.5 if i < 30 else 0.8  # Make later memories more important
            )
        
        memories = self.memory_manager.pet_memories[self.test_pet.id]
        assert len(memories) <= 50  # Should not exceed limit
        
        # Should keep more important memories
        avg_importance = sum(m.importance for m in memories) / len(memories)
        assert avg_importance > 0.6  # Should be higher due to keeping important ones


class TestLazyModelLoader:
    """Test lazy model loading functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        config = ModelLoadingConfig(
            lazy_loading=True,
            quantization=False,  # Disable for testing
            model_cache_size=2,
            unload_after_idle_minutes=1
        )
        self.loader = LazyModelLoader(config)
    
    @patch('digipal.core.performance_optimizer.AutoTokenizer')
    @patch('digipal.core.performance_optimizer.AutoModelForCausalLM')
    def test_lazy_loading(self, mock_model_class, mock_tokenizer_class):
        """Test lazy model loading."""
        # Mock the model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # First call should load the model
        model, tokenizer = self.loader.get_language_model("test-model")
        
        assert model == mock_model
        assert tokenizer == mock_tokenizer
        assert len(self.loader.loaded_models) == 1
        
        # Second call should return cached model
        model2, tokenizer2 = self.loader.get_language_model("test-model")
        
        assert model2 == mock_model
        assert tokenizer2 == mock_tokenizer
        # Should only call from_pretrained once
        assert mock_model_class.from_pretrained.call_count == 1
    
    def test_cache_size_management(self):
        """Test that cache size is managed properly."""
        with patch('digipal.core.performance_optimizer.AutoTokenizer'), \
             patch('digipal.core.performance_optimizer.AutoModelForCausalLM'):
            
            # Load models up to cache limit
            self.loader.get_language_model("model1")
            self.loader.get_language_model("model2")
            
            assert len(self.loader.loaded_models) == 2
            
            # Loading third model should evict oldest
            self.loader.get_language_model("model3")
            
            assert len(self.loader.loaded_models) == 2
            # Should keep most recently used models
            assert "language_model3" in self.loader.loaded_models


class TestBackgroundTaskManager:
    """Test background task management."""
    
    def setup_method(self):
        """Set up test environment."""
        config = BackgroundTaskConfig()
        mock_storage = Mock(spec=StorageManager)
        self.task_manager = BackgroundTaskManager(config, mock_storage)
    
    def test_register_and_execute_task(self):
        """Test registering and executing background tasks."""
        # Create a mock callback
        callback = Mock()
        
        # Register task with short interval for testing
        self.task_manager.register_task("test_task", callback, 0.1)
        
        # Wait for task to execute
        time.sleep(0.2)
        
        # Callback should have been called
        assert callback.call_count >= 1
        
        # Stop the task
        self.task_manager.stop_task("test_task")
    
    def test_task_performance_tracking(self):
        """Test that task performance is tracked."""
        def slow_task():
            time.sleep(0.05)  # Simulate work
        
        self.task_manager.register_task("slow_task", slow_task, 0.1)
        
        # Wait for some executions
        time.sleep(0.3)
        
        # Check performance tracking
        performance = self.task_manager.get_task_performance()
        assert "slow_task" in performance
        assert performance["slow_task"]["total_executions"] > 0
        assert performance["slow_task"]["avg_execution_time"] > 0
        
        self.task_manager.stop_task("slow_task")
    
    def test_stop_all_tasks(self):
        """Test stopping all background tasks."""
        callback1 = Mock()
        callback2 = Mock()
        
        self.task_manager.register_task("task1", callback1, 0.1)
        self.task_manager.register_task("task2", callback2, 0.1)
        
        # Wait for some executions
        time.sleep(0.2)
        
        # Stop all tasks
        self.task_manager.stop_all_tasks()
        
        # Tasks should be stopped
        assert len(self.task_manager.active_tasks) == 0
        assert len(self.task_manager.task_stop_events) == 0


class TestPerformanceMonitor:
    """Test performance monitoring."""
    
    def setup_method(self):
        """Set up test environment."""
        self.monitor = PerformanceMonitor()
    
    @patch('digipal.core.performance_optimizer.psutil')
    @patch('digipal.core.performance_optimizer.torch')
    def test_collect_metrics(self, mock_torch, mock_psutil):
        """Test metrics collection."""
        # Mock system metrics
        mock_psutil.cpu_percent.return_value = 50.0
        mock_memory = Mock()
        mock_memory.percent = 60.0
        mock_psutil.virtual_memory.return_value = mock_memory
        
        # Mock CUDA
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 1024 * 1024 * 512  # 512MB
        mock_torch.cuda.max_memory_allocated.return_value = 1024 * 1024 * 1024  # 1GB
        
        # Collect metrics
        metrics = self.monitor.collect_metrics(
            active_pets=5,
            cached_models=2,
            response_time_avg=1.5
        )
        
        assert metrics.cpu_usage == 50.0
        assert metrics.memory_usage == 60.0
        assert metrics.gpu_memory_usage == 50.0  # 512MB / 1GB * 100
        assert metrics.active_pets == 5
        assert metrics.cached_models == 2
        assert metrics.response_time_avg == 1.5
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        # Add some test metrics
        for i in range(10):
            self.monitor.metrics_history.append(
                Mock(
                    timestamp=datetime.now() - timedelta(minutes=i),
                    cpu_usage=50.0 + i,
                    memory_usage=60.0 + i,
                    gpu_memory_usage=40.0 + i,
                    response_time_avg=1.0 + i * 0.1,
                    active_pets=5,
                    cached_models=2
                )
            )
        
        summary = self.monitor.get_performance_summary(hours=1)
        
        assert "averages" in summary
        assert "peaks" in summary
        assert "current" in summary
        assert summary["data_points"] == 10
    
    def test_performance_alerts(self):
        """Test performance alert generation."""
        # Add metrics that should trigger alerts
        self.monitor.metrics_history.append(
            Mock(
                timestamp=datetime.now(),
                cpu_usage=85.0,  # Should trigger warning
                memory_usage=90.0,  # Should trigger warning
                gpu_memory_usage=95.0,  # Should trigger warning
                response_time_avg=6.0,  # Should trigger warning
                active_pets=5,
                cached_models=2
            )
        )
        
        alerts = self.monitor.check_performance_alerts()
        
        assert len(alerts) > 0
        alert_types = [alert["type"] for alert in alerts]
        assert "high_cpu" in alert_types
        assert "high_memory" in alert_types
        assert "high_gpu_memory" in alert_types
        assert "slow_response" in alert_types


class TestResourceCleanupManager:
    """Test resource cleanup management."""
    
    def setup_method(self):
        """Set up test environment."""
        self.cleanup_manager = ResourceCleanupManager()
    
    def test_register_cleanup_callback(self):
        """Test registering cleanup callbacks."""
        callback = Mock()
        self.cleanup_manager.register_cleanup_callback(callback)
        
        assert len(self.cleanup_manager.cleanup_callbacks) == 1
    
    def test_perform_cleanup(self):
        """Test performing cleanup."""
        callback1 = Mock()
        callback2 = Mock()
        
        self.cleanup_manager.register_cleanup_callback(callback1)
        self.cleanup_manager.register_cleanup_callback(callback2)
        
        # Perform cleanup
        results = self.cleanup_manager.perform_cleanup(force_gc=False)
        
        # Callbacks should have been called
        callback1.assert_called_once()
        callback2.assert_called_once()
        
        # Results should contain callback results
        assert "callback_0" in results
        assert "callback_1" in results
        assert results["callback_0"] == "success"
        assert results["callback_1"] == "success"
    
    @patch('digipal.core.performance_optimizer.psutil')
    def test_get_memory_info(self, mock_psutil):
        """Test getting memory information."""
        # Mock system memory
        mock_memory = Mock()
        mock_memory.total = 8 * 1024 * 1024 * 1024  # 8GB
        mock_memory.available = 4 * 1024 * 1024 * 1024  # 4GB
        mock_memory.used = 4 * 1024 * 1024 * 1024  # 4GB
        mock_memory.percent = 50.0
        mock_psutil.virtual_memory.return_value = mock_memory
        
        memory_info = self.cleanup_manager.get_memory_info()
        
        assert "system_memory" in memory_info
        assert memory_info["system_memory"]["total_mb"] == 8192  # 8GB in MB
        assert memory_info["system_memory"]["percent"] == 50.0


if __name__ == "__main__":
    pytest.main([__file__])