#!/usr/bin/env python3
"""
DigiPal Core Engine Demo

This example demonstrates the complete DigiPal core engine functionality,
including pet creation, interaction processing, and lifecycle management.
"""

import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock

from digipal.core.digipal_core import DigiPalCore
from digipal.core.enums import EggType, LifeStage
from digipal.storage.storage_manager import StorageManager
from digipal.storage.database import DatabaseSchema
from digipal.ai.communication import AICommunication


def create_mock_ai_communication():
    """Create a mock AI communication system for demo purposes."""
    ai_comm = Mock(spec=AICommunication)
    
    # Mock speech processing
    ai_comm.process_speech.return_value = "hello there"
    
    # Mock interaction processing with contextual responses
    def mock_process_interaction(text, pet):
        from digipal.core.models import Interaction
        from digipal.core.enums import InteractionResult
        
        # Simple response logic based on input
        if "feed" in text.lower() or "eat" in text.lower():
            response = f"*munches happily* Thank you for the food!"
            command = "eat"
        elif "train" in text.lower():
            response = f"Let's get stronger together!"
            command = "train"
        elif "play" in text.lower():
            response = f"Yay! Playing is so much fun!"
            command = "play"
        elif "hello" in text.lower() or "hi" in text.lower():
            response = f"Hello! I'm {pet.name}, nice to meet you!"
            command = "greet"
        else:
            response = f"I'm not sure what you mean, but I'm happy to see you!"
            command = "unknown"
        
        return Interaction(
            user_input=text,
            interpreted_command=command,
            pet_response=response,
            success=True,
            result=InteractionResult.SUCCESS
        )
    
    ai_comm.process_interaction.side_effect = mock_process_interaction
    
    # Mock memory manager
    ai_comm.memory_manager = Mock()
    ai_comm.memory_manager.get_interaction_summary.return_value = {
        'total_interactions': 0,
        'successful_interactions': 0,
        'success_rate': 0.0,
        'most_common_commands': [],
        'last_interaction': None
    }
    
    # Mock model unloading
    ai_comm.unload_all_models.return_value = None
    
    return ai_comm


def main():
    """Run the DigiPal Core Engine demo."""
    print("🥚 DigiPal Core Engine Demo 🥚")
    print("=" * 50)
    
    # Create temporary database
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db.close()
    
    try:
        # Initialize database schema
        print("📊 Initializing database...")
        DatabaseSchema.create_database(temp_db.name)
        
        # Create storage manager and AI communication
        storage_manager = StorageManager(temp_db.name, "demo_assets")
        ai_communication = create_mock_ai_communication()
        
        # Create DigiPal core engine
        print("🤖 Initializing DigiPal Core Engine...")
        core = DigiPalCore(storage_manager, ai_communication)
        
        # Create a user (required for foreign key constraints)
        print("👤 Creating demo user...")
        storage_manager.create_user("demo_user", "Demo User", "demo_token")
        
        # Create a new DigiPal
        print("\n🥚 Creating new DigiPal...")
        pet = core.create_new_pet(EggType.RED, "demo_user", "Flamey")
        print(f"✅ Created {pet.name} (ID: {pet.id[:8]}...)")
        print(f"   Egg Type: {pet.egg_type.value}")
        print(f"   Life Stage: {pet.life_stage.value}")
        print(f"   Attributes: HP={pet.hp}, Offense={pet.offense}, Defense={pet.defense}")
        
        # Get initial pet state
        print("\n📊 Getting pet state...")
        state = core.get_pet_state("demo_user")
        print(f"   Status: {state.status_summary}")
        print(f"   Needs Attention: {state.needs_attention}")
        print(f"   Age: {state.age_hours:.2f} hours")
        
        # Process speech interaction to trigger hatching
        print("\n🗣️  Processing speech interaction (triggers hatching)...")
        success, interaction = core.process_speech_interaction("demo_user", b"hello_audio_data")
        print(f"   Success: {success}")
        print(f"   Response: {interaction.pet_response}")
        if interaction.attribute_changes:
            print(f"   Attribute Changes: {interaction.attribute_changes}")
        
        # Check if pet evolved
        updated_pet = core.active_pets["demo_user"]
        if updated_pet.life_stage != LifeStage.EGG:
            print(f"🎉 {updated_pet.name} has evolved to {updated_pet.life_stage.value}!")
        
        # Process text interactions
        print("\n💬 Processing text interactions...")
        interactions = [
            "Hello Flamey!",
            "Let's feed you some food",
            "Time for training!",
            "Want to play together?"
        ]
        
        for text in interactions:
            success, interaction = core.process_interaction("demo_user", text)
            print(f"   User: {text}")
            print(f"   {updated_pet.name}: {interaction.pet_response}")
            print()
        
        # Apply care actions
        print("🎯 Applying care actions...")
        care_actions = ["meat", "strength_training", "play", "rest"]
        
        for action in care_actions:
            success, interaction = core.apply_care_action("demo_user", action)
            if success:
                print(f"   ✅ {action}: {interaction.pet_response}")
                if interaction.attribute_changes:
                    changes = ", ".join([f"{k}={v:+d}" for k, v in interaction.attribute_changes.items()])
                    print(f"      Changes: {changes}")
            else:
                print(f"   ❌ {action}: {interaction.pet_response}")
        
        # Get available care actions
        print("\n🛠️  Available care actions:")
        available_actions = core.get_care_actions("demo_user")
        for action in available_actions[:5]:  # Show first 5
            print(f"   - {action}")
        if len(available_actions) > 5:
            print(f"   ... and {len(available_actions) - 5} more")
        
        # Update pet state (simulate time passing)
        print("\n⏰ Updating pet state (simulating time passage)...")
        # Manually adjust last interaction time to simulate time passing
        updated_pet.last_interaction = datetime.now() - timedelta(hours=1)
        core.update_pet_state("demo_user")
        
        # Get comprehensive statistics
        print("\n📈 Pet Statistics:")
        stats = core.get_pet_statistics("demo_user")
        
        print("   Basic Info:")
        for key, value in stats['basic_info'].items():
            print(f"     {key}: {value}")
        
        print("   Attributes:")
        for key, value in stats['attributes'].items():
            print(f"     {key}: {value}")
        
        print("   Care Assessment:")
        for key, value in stats['care_assessment'].items():
            print(f"     {key}: {value}")
        
        print("   Evolution Status:")
        for key, value in stats['evolution_status'].items():
            print(f"     {key}: {value}")
        
        # Test persistence by creating new core instance
        print("\n💾 Testing persistence...")
        core.shutdown()
        
        # Create new core instance
        core2 = DigiPalCore(storage_manager, ai_communication)
        loaded_pet = core2.load_existing_pet("demo_user")
        
        if loaded_pet:
            print(f"   ✅ Successfully loaded {loaded_pet.name} from storage")
            print(f"   Life Stage: {loaded_pet.life_stage.value}")
            print(f"   HP: {loaded_pet.hp}, Energy: {loaded_pet.energy}")
            print(f"   Conversation History: {len(loaded_pet.conversation_history)} interactions")
        else:
            print("   ❌ Failed to load pet from storage")
        
        core2.shutdown()
        
        print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up temporary database
        try:
            os.unlink(temp_db.name)
            print("🧹 Cleaned up temporary database")
        except:
            pass


if __name__ == "__main__":
    main()