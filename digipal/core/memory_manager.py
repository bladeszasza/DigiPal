"""
Enhanced memory management system for DigiPal with emotional values and RAG capabilities.

This module provides comprehensive memory management including:
- Memory caching for frequently accessed pet data
- Emotional memory system with happiness/stress values
- Simple RAG implementation for relevant memory retrieval
- Memory cleanup and optimization
"""

import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
import threading
import weakref
import gc

from .models import DigiPal, Interaction
from .enums import LifeStage, InteractionResult
from ..storage.storage_manager import StorageManager

logger = logging.getLogger(__name__)


@dataclass
class EmotionalMemory:
    """Represents a memory with emotional context and metadata."""
    id: str
    timestamp: datetime
    content: str
    memory_type: str  # 'interaction', 'action', 'event', 'detail'
    emotional_value: float  # -1.0 (very stressful) to 1.0 (very happy)
    importance: float  # 0.0 to 1.0, affects retention
    tags: Set[str] = field(default_factory=set)
    related_attributes: Dict[str, int] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'content': self.content,
            'memory_type': self.memory_type,
            'emotional_value': self.emotional_value,
            'importance': self.importance,
            'tags': list(self.tags),
            'related_attributes': self.related_attributes,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionalMemory':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        data['tags'] = set(data.get('tags', []))
        return cls(**data)


class MemoryCache:
    """LRU cache for frequently accessed pet data with automatic cleanup."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize memory cache.
        
        Args:
            max_size: Maximum number of items to cache
            ttl_seconds: Time-to-live for cached items in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.RLock()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._stop_cleanup = False
        self._cleanup_thread.start()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            current_time = time.time()
            
            # Check if item exists and is not expired
            if key in self._cache:
                if current_time - self._timestamps[key] < self.ttl_seconds:
                    # Move to end (most recently used)
                    self._cache.move_to_end(key)
                    return self._cache[key]
                else:
                    # Item expired, remove it
                    del self._cache[key]
                    del self._timestamps[key]
            
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        with self._lock:
            current_time = time.time()
            
            # If key exists, update it
            if key in self._cache:
                self._cache[key] = value
                self._timestamps[key] = current_time
                self._cache.move_to_end(key)
                return
            
            # If cache is full, remove least recently used item
            if len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]
            
            # Add new item
            self._cache[key] = value
            self._timestamps[key] = current_time
    
    def remove(self, key: str) -> bool:
        """Remove item from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._timestamps[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)
    
    def _cleanup_loop(self) -> None:
        """Background cleanup loop for expired items."""
        while not self._stop_cleanup:
            try:
                current_time = time.time()
                expired_keys = []
                
                with self._lock:
                    for key, timestamp in self._timestamps.items():
                        if current_time - timestamp >= self.ttl_seconds:
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        if key in self._cache:
                            del self._cache[key]
                        if key in self._timestamps:
                            del self._timestamps[key]
                
                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired cache items")
                
                # Sleep for cleanup interval (1/4 of TTL)
                time.sleep(max(60, self.ttl_seconds // 4))
                
            except Exception as e:
                logger.error(f"Error in cache cleanup loop: {e}")
                time.sleep(60)
    
    def shutdown(self) -> None:
        """Shutdown the cache and cleanup thread."""
        self._stop_cleanup = True
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)


class SimpleRAG:
    """Simple Retrieval-Augmented Generation for memory retrieval."""
    
    def __init__(self, max_context_memories: int = 5):
        """
        Initialize simple RAG system.
        
        Args:
            max_context_memories: Maximum memories to include in context
        """
        self.max_context_memories = max_context_memories
    
    def retrieve_relevant_memories(self, query: str, memories: List[EmotionalMemory], 
                                 current_context: Dict[str, Any]) -> List[EmotionalMemory]:
        """
        Retrieve relevant memories for a given query using simple similarity.
        
        Args:
            query: User input or context query
            memories: Available memories to search
            current_context: Current pet state and context
            
        Returns:
            List of relevant memories sorted by relevance
        """
        if not memories:
            return []
        
        query_lower = query.lower()
        scored_memories = []
        
        for memory in memories:
            score = self._calculate_relevance_score(memory, query_lower, current_context)
            if score > 0:
                scored_memories.append((memory, score))
        
        # Sort by score (descending) and take top memories
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        return [memory for memory, score in scored_memories[:self.max_context_memories]]
    
    def _calculate_relevance_score(self, memory: EmotionalMemory, query_lower: str, 
                                 context: Dict[str, Any]) -> float:
        """Calculate relevance score for a memory."""
        score = 0.0
        
        # Text similarity (simple keyword matching)
        memory_content_lower = memory.content.lower()
        query_words = set(query_lower.split())
        memory_words = set(memory_content_lower.split())
        
        # Keyword overlap
        common_words = query_words.intersection(memory_words)
        if common_words:
            score += len(common_words) / len(query_words) * 0.4
        
        # Tag matching
        query_tags = self._extract_tags_from_query(query_lower)
        tag_overlap = query_tags.intersection(memory.tags)
        if tag_overlap:
            score += len(tag_overlap) / max(len(query_tags), 1) * 0.3
        
        # Recency boost (more recent memories are more relevant)
        hours_ago = (datetime.now() - memory.timestamp).total_seconds() / 3600
        recency_score = max(0, 1 - (hours_ago / 168))  # Decay over a week
        score += recency_score * 0.2
        
        # Importance boost
        score += memory.importance * 0.1
        
        # Emotional relevance (memories with strong emotions are more memorable)
        emotional_strength = abs(memory.emotional_value)
        score += emotional_strength * 0.1
        
        # Access frequency (frequently accessed memories are more relevant)
        access_boost = min(memory.access_count / 10, 1.0) * 0.1
        score += access_boost
        
        return score
    
    def _extract_tags_from_query(self, query_lower: str) -> Set[str]:
        """Extract potential tags from query text."""
        # Simple tag extraction based on common patterns
        tags = set()
        
        # Action-based tags
        if any(word in query_lower for word in ['eat', 'food', 'hungry', 'feed']):
            tags.add('eating')
        if any(word in query_lower for word in ['sleep', 'rest', 'tired', 'nap']):
            tags.add('sleeping')
        if any(word in query_lower for word in ['train', 'exercise', 'workout']):
            tags.add('training')
        if any(word in query_lower for word in ['play', 'fun', 'game']):
            tags.add('playing')
        if any(word in query_lower for word in ['good', 'praise', 'well done']):
            tags.add('praise')
        if any(word in query_lower for word in ['bad', 'scold', 'no']):
            tags.add('discipline')
        
        # Emotional tags
        if any(word in query_lower for word in ['happy', 'joy', 'excited']):
            tags.add('positive')
        if any(word in query_lower for word in ['sad', 'upset', 'angry']):
            tags.add('negative')
        
        return tags


class EnhancedMemoryManager:
    """Enhanced memory manager with emotional values, caching, and RAG capabilities."""
    
    def __init__(self, storage_manager: StorageManager, cache_size: int = 1000, 
                 max_memories_per_pet: int = 500):
        """
        Initialize enhanced memory manager.
        
        Args:
            storage_manager: Storage manager for persistence
            cache_size: Size of memory cache
            max_memories_per_pet: Maximum memories to keep per pet
        """
        self.storage_manager = storage_manager
        self.max_memories_per_pet = max_memories_per_pet
        
        # Memory cache for frequently accessed data
        self.memory_cache = MemoryCache(max_size=cache_size)
        
        # Pet memories storage (pet_id -> List[EmotionalMemory])
        self.pet_memories: Dict[str, List[EmotionalMemory]] = defaultdict(list)
        
        # RAG system for memory retrieval
        self.rag_system = SimpleRAG()
        
        # Memory statistics
        self.memory_stats = defaultdict(lambda: {
            'total_memories': 0,
            'happy_memories': 0,
            'stressful_memories': 0,
            'neutral_memories': 0,
            'last_cleanup': datetime.now()
        })
        
        # Background cleanup
        self._cleanup_thread = None
        self._stop_cleanup = False
        
        logger.info("Enhanced memory manager initialized")
    
    def add_memory(self, pet_id: str, content: str, memory_type: str, 
                  emotional_value: float = 0.0, importance: float = 0.5,
                  tags: Optional[Set[str]] = None, 
                  related_attributes: Optional[Dict[str, int]] = None) -> str:
        """
        Add a new memory for a pet.
        
        Args:
            pet_id: Pet identifier
            content: Memory content
            memory_type: Type of memory ('interaction', 'action', 'event', 'detail')
            emotional_value: Emotional value (-1.0 to 1.0)
            importance: Importance level (0.0 to 1.0)
            tags: Optional tags for categorization
            related_attributes: Optional attribute changes related to this memory
            
        Returns:
            Memory ID
        """
        memory_id = f"{pet_id}_{int(time.time() * 1000)}"
        
        memory = EmotionalMemory(
            id=memory_id,
            timestamp=datetime.now(),
            content=content,
            memory_type=memory_type,
            emotional_value=max(-1.0, min(1.0, emotional_value)),
            importance=max(0.0, min(1.0, importance)),
            tags=tags or set(),
            related_attributes=related_attributes or {}
        )
        
        # Add to pet memories
        self.pet_memories[pet_id].append(memory)
        
        # Update statistics
        self._update_memory_stats(pet_id, memory)
        
        # Manage memory size
        self._manage_memory_size(pet_id)
        
        # Cache the memory
        self.memory_cache.put(f"memory_{memory_id}", memory)
        
        logger.debug(f"Added memory {memory_id} for pet {pet_id}: {content[:50]}...")
        return memory_id
    
    def add_interaction_memory(self, pet: DigiPal, interaction: Interaction) -> str:
        """
        Add memory from an interaction with emotional context.
        
        Args:
            pet: DigiPal instance
            interaction: Interaction to convert to memory
            
        Returns:
            Memory ID
        """
        # Calculate emotional value based on interaction
        emotional_value = self._calculate_emotional_value(interaction, pet)
        
        # Calculate importance based on success and attribute changes
        importance = self._calculate_importance(interaction)
        
        # Extract tags from interaction
        tags = self._extract_interaction_tags(interaction)
        
        # Create memory content
        content = f"User said: '{interaction.user_input}' - I responded: '{interaction.pet_response}'"
        
        return self.add_memory(
            pet_id=pet.id,
            content=content,
            memory_type='interaction',
            emotional_value=emotional_value,
            importance=importance,
            tags=tags,
            related_attributes=interaction.attribute_changes
        )
    
    def add_action_memory(self, pet_id: str, action: str, result: str, 
                         attribute_changes: Dict[str, int]) -> str:
        """
        Add memory from a care action.
        
        Args:
            pet_id: Pet identifier
            action: Action performed
            result: Result of the action
            attribute_changes: Attribute changes from action
            
        Returns:
            Memory ID
        """
        # Calculate emotional value based on attribute changes
        emotional_value = 0.0
        if 'happiness' in attribute_changes:
            emotional_value += attribute_changes['happiness'] / 100.0
        if 'energy' in attribute_changes:
            emotional_value += attribute_changes['energy'] / 200.0
        
        # Clamp emotional value
        emotional_value = max(-1.0, min(1.0, emotional_value))
        
        # Calculate importance based on magnitude of changes
        importance = min(1.0, sum(abs(v) for v in attribute_changes.values()) / 100.0)
        
        # Extract tags
        tags = {action, 'action'}
        if emotional_value > 0.3:
            tags.add('positive')
        elif emotional_value < -0.3:
            tags.add('negative')
        
        content = f"Action: {action} - Result: {result}"
        
        return self.add_memory(
            pet_id=pet_id,
            content=content,
            memory_type='action',
            emotional_value=emotional_value,
            importance=importance,
            tags=tags,
            related_attributes=attribute_changes
        )
    
    def add_life_event_memory(self, pet_id: str, event: str, emotional_impact: float = 0.0) -> str:
        """
        Add memory for significant life events (evolution, achievements, etc.).
        
        Args:
            pet_id: Pet identifier
            event: Event description
            emotional_impact: Emotional impact of the event
            
        Returns:
            Memory ID
        """
        return self.add_memory(
            pet_id=pet_id,
            content=event,
            memory_type='event',
            emotional_value=emotional_impact,
            importance=0.9,  # Life events are usually important
            tags={'life_event', 'milestone'}
        )
    
    def get_relevant_memories(self, pet_id: str, query: str, 
                            current_context: Optional[Dict[str, Any]] = None) -> List[EmotionalMemory]:
        """
        Get relevant memories for a query using RAG.
        
        Args:
            pet_id: Pet identifier
            query: Query text
            current_context: Current pet state context
            
        Returns:
            List of relevant memories
        """
        memories = self.pet_memories.get(pet_id, [])
        if not memories:
            return []
        
        context = current_context or {}
        relevant_memories = self.rag_system.retrieve_relevant_memories(query, memories, context)
        
        # Update access counts for retrieved memories
        for memory in relevant_memories:
            memory.access_count += 1
            memory.last_accessed = datetime.now()
        
        return relevant_memories
    
    def get_memory_context_for_llm(self, pet_id: str, query: str, 
                                  current_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Get formatted memory context for LLM input.
        
        Args:
            pet_id: Pet identifier
            query: Current query
            current_context: Current pet state
            
        Returns:
            Formatted memory context string
        """
        relevant_memories = self.get_relevant_memories(pet_id, query, current_context)
        
        if not relevant_memories:
            return ""
        
        context_parts = ["Recent relevant memories:"]
        
        for memory in relevant_memories:
            # Format memory with emotional context
            emotional_indicator = ""
            if memory.emotional_value > 0.3:
                emotional_indicator = " (happy memory)"
            elif memory.emotional_value < -0.3:
                emotional_indicator = " (stressful memory)"
            
            time_ago = self._format_time_ago(memory.timestamp)
            context_parts.append(f"- {time_ago}: {memory.content}{emotional_indicator}")
        
        return "\n".join(context_parts)
    
    def get_emotional_state_summary(self, pet_id: str) -> Dict[str, Any]:
        """
        Get emotional state summary based on recent memories.
        
        Args:
            pet_id: Pet identifier
            
        Returns:
            Emotional state summary
        """
        memories = self.pet_memories.get(pet_id, [])
        if not memories:
            return {'overall_mood': 'neutral', 'recent_trend': 'stable'}
        
        # Analyze recent memories (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_memories = [m for m in memories if m.timestamp > recent_cutoff]
        
        if not recent_memories:
            recent_memories = memories[-10:]  # Use last 10 if no recent ones
        
        # Calculate emotional metrics
        total_emotional_value = sum(m.emotional_value for m in recent_memories)
        avg_emotional_value = total_emotional_value / len(recent_memories)
        
        positive_memories = sum(1 for m in recent_memories if m.emotional_value > 0.1)
        negative_memories = sum(1 for m in recent_memories if m.emotional_value < -0.1)
        
        # Determine overall mood
        if avg_emotional_value > 0.3:
            overall_mood = 'very_happy'
        elif avg_emotional_value > 0.1:
            overall_mood = 'happy'
        elif avg_emotional_value < -0.3:
            overall_mood = 'stressed'
        elif avg_emotional_value < -0.1:
            overall_mood = 'unhappy'
        else:
            overall_mood = 'neutral'
        
        # Determine trend
        if len(recent_memories) >= 5:
            first_half = recent_memories[:len(recent_memories)//2]
            second_half = recent_memories[len(recent_memories)//2:]
            
            first_avg = sum(m.emotional_value for m in first_half) / len(first_half)
            second_avg = sum(m.emotional_value for m in second_half) / len(second_half)
            
            if second_avg - first_avg > 0.2:
                trend = 'improving'
            elif first_avg - second_avg > 0.2:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return {
            'overall_mood': overall_mood,
            'recent_trend': trend,
            'avg_emotional_value': avg_emotional_value,
            'positive_memories': positive_memories,
            'negative_memories': negative_memories,
            'total_recent_memories': len(recent_memories)
        }
    
    def cleanup_old_memories(self, pet_id: str, max_age_days: int = 30) -> int:
        """
        Clean up old memories while preserving important ones.
        
        Args:
            pet_id: Pet identifier
            max_age_days: Maximum age for memories in days
            
        Returns:
            Number of memories cleaned up
        """
        memories = self.pet_memories.get(pet_id, [])
        if not memories:
            return 0
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        # Separate memories into keep and remove lists
        keep_memories = []
        removed_count = 0
        
        for memory in memories:
            # Always keep important memories or recent ones
            if (memory.importance > 0.7 or 
                memory.timestamp > cutoff_date or
                abs(memory.emotional_value) > 0.5):
                keep_memories.append(memory)
            else:
                removed_count += 1
        
        # Update memories list
        self.pet_memories[pet_id] = keep_memories
        
        # Update statistics
        self.memory_stats[pet_id]['last_cleanup'] = datetime.now()
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old memories for pet {pet_id}")
        
        return removed_count
    
    def get_memory_statistics(self, pet_id: str) -> Dict[str, Any]:
        """Get memory statistics for a pet."""
        memories = self.pet_memories.get(pet_id, [])
        stats = self.memory_stats[pet_id].copy()
        
        # Update current counts
        stats['total_memories'] = len(memories)
        stats['happy_memories'] = sum(1 for m in memories if m.emotional_value > 0.1)
        stats['stressful_memories'] = sum(1 for m in memories if m.emotional_value < -0.1)
        stats['neutral_memories'] = stats['total_memories'] - stats['happy_memories'] - stats['stressful_memories']
        
        # Memory type breakdown
        type_counts = defaultdict(int)
        for memory in memories:
            type_counts[memory.memory_type] += 1
        stats['memory_types'] = dict(type_counts)
        
        # Recent activity
        recent_cutoff = datetime.now() - timedelta(hours=24)
        stats['recent_memories'] = sum(1 for m in memories if m.timestamp > recent_cutoff)
        
        return stats
    
    def _calculate_emotional_value(self, interaction: Interaction, pet: DigiPal) -> float:
        """Calculate emotional value for an interaction."""
        emotional_value = 0.0
        
        # Base emotional value from success/failure
        if interaction.success:
            emotional_value += 0.2
        else:
            emotional_value -= 0.3
        
        # Adjust based on interaction result
        if interaction.result == InteractionResult.SUCCESS:
            emotional_value += 0.1
        elif interaction.result == InteractionResult.FAILURE:
            emotional_value -= 0.2
        elif interaction.result == InteractionResult.STAGE_INAPPROPRIATE:
            emotional_value -= 0.1
        
        # Adjust based on command type
        command = interaction.interpreted_command.lower()
        if command in ['good', 'praise']:
            emotional_value += 0.4
        elif command in ['bad', 'scold']:
            emotional_value -= 0.3
        elif command in ['play', 'fun']:
            emotional_value += 0.2
        elif command in ['eat', 'food'] and pet.energy < 50:
            emotional_value += 0.3  # Food when hungry is very positive
        
        # Adjust based on attribute changes
        if 'happiness' in interaction.attribute_changes:
            emotional_value += interaction.attribute_changes['happiness'] / 100.0
        
        return max(-1.0, min(1.0, emotional_value))
    
    def _calculate_importance(self, interaction: Interaction) -> float:
        """Calculate importance level for an interaction."""
        importance = 0.5  # Base importance
        
        # Increase importance for successful interactions
        if interaction.success:
            importance += 0.2
        
        # Increase importance based on attribute changes
        total_change = sum(abs(v) for v in interaction.attribute_changes.values())
        importance += min(0.3, total_change / 100.0)
        
        # Special commands are more important
        special_commands = ['evolution', 'death', 'birth', 'milestone']
        if any(cmd in interaction.interpreted_command.lower() for cmd in special_commands):
            importance += 0.3
        
        return max(0.0, min(1.0, importance))
    
    def _extract_interaction_tags(self, interaction: Interaction) -> Set[str]:
        """Extract tags from an interaction."""
        tags = {'interaction'}
        
        command = interaction.interpreted_command.lower()
        
        # Command-based tags
        if command in ['eat', 'food', 'feed']:
            tags.add('eating')
        elif command in ['sleep', 'rest']:
            tags.add('sleeping')
        elif command in ['train', 'exercise']:
            tags.add('training')
        elif command in ['play', 'fun']:
            tags.add('playing')
        elif command in ['good', 'praise']:
            tags.add('praise')
        elif command in ['bad', 'scold']:
            tags.add('discipline')
        
        # Success/failure tags
        if interaction.success:
            tags.add('successful')
        else:
            tags.add('failed')
        
        return tags
    
    def _update_memory_stats(self, pet_id: str, memory: EmotionalMemory) -> None:
        """Update memory statistics."""
        stats = self.memory_stats[pet_id]
        stats['total_memories'] += 1
        
        if memory.emotional_value > 0.1:
            stats['happy_memories'] += 1
        elif memory.emotional_value < -0.1:
            stats['stressful_memories'] += 1
        else:
            stats['neutral_memories'] += 1
    
    def _manage_memory_size(self, pet_id: str) -> None:
        """Manage memory size to prevent unlimited growth."""
        memories = self.pet_memories[pet_id]
        
        if len(memories) > self.max_memories_per_pet:
            # Sort by importance and recency, keep the most important/recent
            memories.sort(key=lambda m: (m.importance, m.timestamp.timestamp()), reverse=True)
            
            # Keep top memories
            self.pet_memories[pet_id] = memories[:self.max_memories_per_pet]
            
            removed_count = len(memories) - self.max_memories_per_pet
            logger.debug(f"Removed {removed_count} old memories for pet {pet_id}")
    
    def _format_time_ago(self, timestamp: datetime) -> str:
        """Format timestamp as 'time ago' string."""
        delta = datetime.now() - timestamp
        
        if delta.days > 0:
            return f"{delta.days} day{'s' if delta.days != 1 else ''} ago"
        elif delta.seconds > 3600:
            hours = delta.seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif delta.seconds > 60:
            minutes = delta.seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        else:
            return "just now"
    
    def start_background_cleanup(self) -> None:
        """Start background cleanup thread."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            return
        
        self._stop_cleanup = False
        self._cleanup_thread = threading.Thread(target=self._background_cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        logger.info("Started background memory cleanup")
    
    def stop_background_cleanup(self) -> None:
        """Stop background cleanup thread."""
        self._stop_cleanup = True
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
    
    def _background_cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while not self._stop_cleanup:
            try:
                # Clean up old memories for all pets
                for pet_id in list(self.pet_memories.keys()):
                    self.cleanup_old_memories(pet_id)
                
                # Force garbage collection
                gc.collect()
                
                # Sleep for 1 hour
                time.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in background memory cleanup: {e}")
                time.sleep(300)  # Sleep 5 minutes on error
    
    def shutdown(self) -> None:
        """Shutdown the memory manager."""
        logger.info("Shutting down enhanced memory manager")
        
        # Stop background cleanup
        self.stop_background_cleanup()
        
        # Shutdown cache
        self.memory_cache.shutdown()
        
        # Clear memories to free memory
        self.pet_memories.clear()
        self.memory_stats.clear()
        
        logger.info("Enhanced memory manager shutdown complete")