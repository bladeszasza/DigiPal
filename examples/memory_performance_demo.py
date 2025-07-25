#!/usr/bin/env python3
"""
Demo script showcasing the enhanced memory management and performance optimization features.

This script demonstrates:
- Enhanced memory system with emotional values
- Simple RAG for memory retrieval
- Performance optimization with lazy loading
- Background task management
- Resource cleanup and monitoring
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import digipal
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from digipal.core.models import DigiPal, Interaction
from digipal.core.enums import EggType, LifeStage, InteractionResult
from digipal.core.memory_manager import EnhancedMemoryManager
from digipal.core.performance_optimizer import (
    LazyModelLoader, BackgroundTaskManager, PerformanceMonitor,
    ResourceCleanupManager, ModelLoadingConfig, BackgroundTaskConfig
)
from digipal.storage.storage_manager import StorageManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_pet() -> DigiPal:
    """Create a test DigiPal for demonstration."""
    pet = DigiPal(
        id="demo_pet_001",
        user_id="demo_user",
        name="MemoryPal",
        egg_type=EggType.BLUE,
        life_stage=LifeStage.CHILD,
        happiness=75,
        energy=80
    )
    
    # Add some conversation history
    interactions = [
        Interaction(
            timestamp=datetime.now() - timedelta(hours=2),
            user_input="good job",
            interpreted_command="praise",
            pet_response="Thank you! I'm so happy!",
            attribute_changes={"happiness": 10},
            success=True,
            result=InteractionResult.SUCCESS
        ),
        Interaction(
            timestamp=datetime.now() - timedelta(hours=1),
            user_input="let's train",
            interpreted_command="train",
            pet_response="Yes! Let's get stronger!",
            attribute_changes={"offense": 5, "energy": -10},
            success=True,
            result=InteractionResult.SUCCESS
        ),
        Interaction(
            timestamp=datetime.now() - timedelta(minutes=30),
            user_input="are you hungry?",
            interpreted_command="status",
            pet_response="I could eat something!",
            attribute_changes={},
            success=True,
            result=InteractionResult.SUCCESS
        )
    ]
    
    pet.conversation_history = interactions
    return pet


def demo_enhanced_memory_system():
    """Demonstrate the enhanced memory system."""
    print("\n" + "="*60)
    print("ENHANCED MEMORY SYSTEM DEMO")
    print("="*60)
    
    # Create storage manager (in-memory for demo)
    storage_manager = StorageManager("demo_memory.db", "demo_assets")
    
    # Create enhanced memory manager
    memory_manager = EnhancedMemoryManager(storage_manager, cache_size=100)
    
    # Create test pet
    pet = create_test_pet()
    
    print(f"\nCreated test pet: {pet.name} (ID: {pet.id})")
    print(f"Life stage: {pet.life_stage.value}")
    print(f"Happiness: {pet.happiness}, Energy: {pet.energy}")
    
    # Add interaction memories with emotional context
    print("\nAdding interaction memories...")
    for interaction in pet.conversation_history:
        memory_id = memory_manager.add_interaction_memory(pet, interaction)
        print(f"  Added memory {memory_id}: {interaction.user_input[:30]}...")
    
    # Add some action memories
    print("\nAdding action memories...")
    memory_manager.add_action_memory(
        pet.id, "feeding", "Pet was fed and became happy", {"happiness": 15, "energy": 10}
    )
    memory_manager.add_action_memory(
        pet.id, "scolding", "Pet was scolded for misbehavior", {"happiness": -5, "discipline": 10}
    )
    
    # Add life event memory
    memory_manager.add_life_event_memory(
        pet.id, "Pet evolved from baby to child stage!", 0.9
    )
    
    # Demonstrate RAG retrieval
    print("\nDemonstrating RAG memory retrieval:")
    
    queries = [
        "I want to train my pet",
        "How is my pet feeling?",
        "Let's praise the pet",
        "What happened recently?"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        relevant_memories = memory_manager.get_relevant_memories(pet.id, query)
        
        if relevant_memories:
            print("  Relevant memories:")
            for i, memory in enumerate(relevant_memories[:3], 1):
                emotional_indicator = ""
                if memory.emotional_value > 0.3:
                    emotional_indicator = " 😊"
                elif memory.emotional_value < -0.3:
                    emotional_indicator = " 😔"
                
                print(f"    {i}. {memory.content[:50]}...{emotional_indicator}")
                print(f"       Emotional value: {memory.emotional_value:.2f}, Importance: {memory.importance:.2f}")
        else:
            print("  No relevant memories found")
    
    # Get memory context for LLM
    print("\nMemory context for LLM:")
    context = memory_manager.get_memory_context_for_llm(pet.id, "How are you feeling?")
    print(context)
    
    # Get emotional state summary
    print("\nEmotional state summary:")
    emotional_state = memory_manager.get_emotional_state_summary(pet.id)
    print(f"  Overall mood: {emotional_state['overall_mood']}")
    print(f"  Recent trend: {emotional_state['recent_trend']}")
    print(f"  Positive memories: {emotional_state['positive_memories']}")
    print(f"  Negative memories: {emotional_state['negative_memories']}")
    
    # Get memory statistics
    print("\nMemory statistics:")
    stats = memory_manager.get_memory_statistics(pet.id)
    print(f"  Total memories: {stats['total_memories']}")
    print(f"  Happy memories: {stats['happy_memories']}")
    print(f"  Stressful memories: {stats['stressful_memories']}")
    print(f"  Neutral memories: {stats['neutral_memories']}")
    
    # Cleanup
    memory_manager.shutdown()
    
    # Clean up demo database
    try:
        os.remove("demo_memory.db")
    except:
        pass


def demo_performance_optimization():
    """Demonstrate performance optimization features."""
    print("\n" + "="*60)
    print("PERFORMANCE OPTIMIZATION DEMO")
    print("="*60)
    
    # Model loading configuration
    model_config = ModelLoadingConfig(
        lazy_loading=True,
        quantization=False,  # Disabled for demo
        model_cache_size=2,
        unload_after_idle_minutes=1
    )
    
    # Create lazy model loader
    print("\nCreating lazy model loader...")
    model_loader = LazyModelLoader(model_config)
    
    print("Model loader configuration:")
    print(f"  Lazy loading: {model_config.lazy_loading}")
    print(f"  Cache size: {model_config.model_cache_size}")
    print(f"  Unload after: {model_config.unload_after_idle_minutes} minutes")
    
    # Get cache info
    cache_info = model_loader.get_cache_info()
    print(f"\nInitial cache state:")
    print(f"  Loaded models: {cache_info['loaded_models']}")
    print(f"  Cache limit: {cache_info['cache_limit']}")
    
    # Background task management
    print("\nSetting up background task management...")
    task_config = BackgroundTaskConfig()
    storage_manager = StorageManager("demo_tasks.db", "demo_assets")
    task_manager = BackgroundTaskManager(task_config, storage_manager)
    
    # Register demo tasks
    task_counter = {'count': 0}
    
    def demo_task():
        task_counter['count'] += 1
        print(f"  Demo task executed {task_counter['count']} times")
    
    task_manager.register_task("demo_task", demo_task, 2)  # Every 2 seconds
    
    print("Registered demo background task (runs every 2 seconds)")
    print("Letting it run for 6 seconds...")
    time.sleep(6)
    
    # Get task performance
    performance = task_manager.get_task_performance()
    if "demo_task" in performance:
        task_perf = performance["demo_task"]
        print(f"\nTask performance:")
        print(f"  Total executions: {task_perf['total_executions']}")
        print(f"  Average execution time: {task_perf['avg_execution_time']:.4f}s")
    
    # Stop tasks
    task_manager.stop_all_tasks()
    print("Stopped all background tasks")
    
    # Performance monitoring
    print("\nDemonstrating performance monitoring...")
    monitor = PerformanceMonitor()
    
    # Collect some metrics
    for i in range(5):
        metrics = monitor.collect_metrics(
            active_pets=i + 1,
            cached_models=1,
            response_time_avg=1.0 + i * 0.1
        )
        print(f"  Collected metrics {i+1}: CPU={metrics.cpu_usage:.1f}%, Memory={metrics.memory_usage:.1f}%")
        time.sleep(0.5)
    
    # Get performance summary
    summary = monitor.get_performance_summary(hours=1)
    print(f"\nPerformance summary:")
    print(f"  Data points: {summary.get('data_points', 0)}")
    if 'averages' in summary:
        avg = summary['averages']
        print(f"  Average CPU: {avg.get('cpu_usage', 0):.1f}%")
        print(f"  Average Memory: {avg.get('memory_usage', 0):.1f}%")
        print(f"  Average Response Time: {avg.get('response_time', 0):.2f}s")
    
    # Check for alerts
    alerts = monitor.check_performance_alerts()
    if alerts:
        print(f"\nPerformance alerts:")
        for alert in alerts:
            print(f"  {alert['severity'].upper()}: {alert['message']}")
    else:
        print("\nNo performance alerts")
    
    # Get optimization suggestions
    suggestions = monitor.suggest_optimizations()
    if suggestions:
        print(f"\nOptimization suggestions:")
        for suggestion in suggestions:
            print(f"  - {suggestion}")
    else:
        print("\nNo optimization suggestions at this time")
    
    # Resource cleanup
    print("\nDemonstrating resource cleanup...")
    cleanup_manager = ResourceCleanupManager()
    
    # Register cleanup callback
    cleanup_called = {'called': False}
    def demo_cleanup():
        cleanup_called['called'] = True
        print("  Demo cleanup callback executed")
    
    cleanup_manager.register_cleanup_callback(demo_cleanup)
    
    # Perform cleanup
    results = cleanup_manager.perform_cleanup(force_gc=True)
    print(f"Cleanup results: {len(results)} operations completed")
    print(f"Demo cleanup called: {cleanup_called['called']}")
    
    # Get memory info
    memory_info = cleanup_manager.get_memory_info()
    if 'system_memory' in memory_info:
        sys_mem = memory_info['system_memory']
        print(f"\nSystem memory info:")
        print(f"  Total: {sys_mem.get('total_mb', 0):.0f} MB")
        print(f"  Available: {sys_mem.get('available_mb', 0):.0f} MB")
        print(f"  Usage: {sys_mem.get('percent', 0):.1f}%")
    
    # Cleanup
    model_loader.shutdown()
    
    # Clean up demo database
    try:
        os.remove("demo_tasks.db")
    except:
        pass


def demo_integration():
    """Demonstrate integration of memory and performance systems."""
    print("\n" + "="*60)
    print("INTEGRATION DEMO")
    print("="*60)
    
    print("This demo shows how the enhanced memory system and performance")
    print("optimization work together in a real DigiPal application.")
    
    # Create components
    storage_manager = StorageManager("demo_integration.db", "demo_assets")
    memory_manager = EnhancedMemoryManager(storage_manager)
    
    # Create test pet
    pet = create_test_pet()
    
    print(f"\nSimulating DigiPal interactions with {pet.name}...")
    
    # Simulate user interactions
    interactions = [
        ("Hello there!", "greeting", "Hi! Nice to see you!", {"happiness": 5}),
        ("Let's play!", "play", "Yay! I love playing!", {"happiness": 10, "energy": -5}),
        ("Good job!", "praise", "Thank you so much!", {"happiness": 15}),
        ("Time to rest", "sleep", "I am getting sleepy...", {"energy": 20}),
        ("How are you feeling?", "status", "I'm feeling great!", {})
    ]
    
    for user_input, command, response, changes in interactions:
        # Create interaction
        interaction = Interaction(
            timestamp=datetime.now(),
            user_input=user_input,
            interpreted_command=command,
            pet_response=response,
            attribute_changes=changes,
            success=True,
            result=InteractionResult.SUCCESS
        )
        
        # Add to pet's conversation history
        pet.conversation_history.append(interaction)
        
        # Add to enhanced memory system
        memory_id = memory_manager.add_interaction_memory(pet, interaction)
        
        print(f"  User: {user_input}")
        print(f"  Pet: {response}")
        print(f"  Memory ID: {memory_id}")
        print()
        
        time.sleep(0.5)  # Simulate time between interactions
    
    # Demonstrate memory retrieval during conversation
    print("Demonstrating contextual memory retrieval:")
    
    query = "What did we do together?"
    print(f"\nUser asks: '{query}'")
    
    # Get relevant memories
    relevant_memories = memory_manager.get_relevant_memories(pet.id, query)
    
    print("Pet recalls these memories:")
    for memory in relevant_memories[:3]:
        time_ago = (datetime.now() - memory.timestamp).total_seconds()
        if time_ago < 60:
            time_str = "just now"
        elif time_ago < 3600:
            time_str = f"{int(time_ago/60)} minutes ago"
        else:
            time_str = f"{int(time_ago/3600)} hours ago"
        
        emotional_context = ""
        if memory.emotional_value > 0.3:
            emotional_context = " (happy memory)"
        elif memory.emotional_value < -0.3:
            emotional_context = " (sad memory)"
        
        print(f"  - {time_str}: {memory.content}{emotional_context}")
    
    # Get emotional state
    emotional_state = memory_manager.get_emotional_state_summary(pet.id)
    print(f"\nPet's emotional state: {emotional_state['overall_mood']}")
    print(f"Recent trend: {emotional_state['recent_trend']}")
    
    # Show memory statistics
    stats = memory_manager.get_memory_statistics(pet.id)
    print(f"\nMemory statistics:")
    print(f"  Total memories: {stats['total_memories']}")
    print(f"  Happy memories: {stats['happy_memories']}")
    print(f"  Recent memories (24h): {stats['recent_memories']}")
    
    # Cleanup
    memory_manager.shutdown()
    
    # Clean up demo database
    try:
        os.remove("demo_integration.db")
    except:
        pass


def main():
    """Run all demonstrations."""
    print("DigiPal Memory Management and Performance Optimization Demo")
    print("=" * 60)
    
    try:
        # Run demonstrations
        demo_enhanced_memory_system()
        demo_performance_optimization()
        demo_integration()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey features demonstrated:")
        print("✓ Enhanced memory system with emotional values")
        print("✓ Simple RAG for relevant memory retrieval")
        print("✓ Lazy model loading with caching")
        print("✓ Background task management")
        print("✓ Performance monitoring and alerts")
        print("✓ Resource cleanup and optimization")
        print("✓ Integration of all systems")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())