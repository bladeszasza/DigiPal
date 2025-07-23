#!/usr/bin/env python3
"""
Demo script showing DigiPal storage functionality.
"""

import sys
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path to import digipal
sys.path.insert(0, str(Path(__file__).parent.parent))

from digipal.storage import StorageManager
from digipal.core.models import DigiPal, Interaction, CareAction, AttributeModifier
from digipal.core.enums import (
    EggType, LifeStage, CareActionType, AttributeType, 
    InteractionResult
)


def main():
    """Demonstrate storage functionality."""
    print("🥚 DigiPal Storage Demo")
    print("=" * 50)
    
    # Create temporary storage
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "demo.db"
    assets_path = Path(temp_dir) / "assets"
    
    print(f"📁 Using temporary storage: {temp_dir}")
    
    # Initialize storage manager
    storage = StorageManager(str(db_path), str(assets_path))
    
    # Create a user
    user_id = "demo_user_123"
    username = "demo_user"
    print(f"\n👤 Creating user: {username}")
    
    success = storage.create_user(user_id, username, "demo_token")
    if success:
        print("✅ User created successfully")
    else:
        print("❌ Failed to create user")
        return
    
    # Create a DigiPal
    print(f"\n🥚 Creating DigiPal egg...")
    pet = DigiPal(
        user_id=user_id,
        name="DemoPal",
        egg_type=EggType.RED,
        life_stage=LifeStage.EGG
    )
    
    success = storage.save_pet(pet)
    if success:
        print("✅ DigiPal egg saved successfully")
    else:
        print("❌ Failed to save DigiPal")
        return
    
    # Hatch the egg
    print(f"\n🐣 Hatching DigiPal...")
    pet.life_stage = LifeStage.BABY
    pet.learned_commands = {"eat", "sleep", "good", "bad"}
    pet.last_interaction = datetime.now()
    
    # Add first interaction
    first_interaction = Interaction(
        user_input="Hello little one!",
        interpreted_command="greet",
        pet_response="*chirp chirp*",
        attribute_changes={"happiness": 5},
        success=True,
        result=InteractionResult.SUCCESS
    )
    pet.conversation_history.append(first_interaction)
    
    storage.save_pet(pet)
    print("✅ DigiPal hatched and first interaction recorded")
    
    # Load the pet back
    print(f"\n📖 Loading DigiPal from storage...")
    loaded_pet = storage.load_pet(user_id)
    
    if loaded_pet:
        print(f"✅ Loaded DigiPal: {loaded_pet.name}")
        print(f"   Life Stage: {loaded_pet.life_stage.value}")
        print(f"   Happiness: {loaded_pet.happiness}")
        print(f"   Learned Commands: {loaded_pet.learned_commands}")
        print(f"   Conversation History: {len(loaded_pet.conversation_history)} interactions")
    else:
        print("❌ Failed to load DigiPal")
        return
    
    # Perform care actions
    print(f"\n🍼 Performing care actions...")
    
    # Feed the DigiPal
    care_action = CareAction(
        name="Feed Milk",
        action_type=CareActionType.FEED,
        energy_cost=0,
        happiness_change=10,
        attribute_modifiers=[
            AttributeModifier(AttributeType.HAPPINESS, 10),
            AttributeModifier(AttributeType.WEIGHT, 1)
        ]
    )
    
    # Apply care action effects
    loaded_pet.happiness += 10
    loaded_pet.weight += 1
    
    # Record care action
    attribute_changes = {"happiness": 10, "weight": 1}
    success = storage.save_care_action(
        loaded_pet.id, 
        user_id, 
        care_action, 
        attribute_changes
    )
    
    if success:
        print("✅ Care action recorded")
    
    # Save updated pet
    storage.save_pet(loaded_pet)
    
    # Get care history
    care_history = storage.get_care_history(loaded_pet.id)
    print(f"📋 Care History: {len(care_history)} actions")
    
    # Create backup
    print(f"\n💾 Creating backup...")
    success = storage.create_backup(user_id, "demo_backup")
    
    if success:
        print("✅ Backup created successfully")
        
        # List backups
        backups = storage.get_backups(user_id)
        print(f"📦 Available backups: {len(backups)}")
        for backup in backups:
            print(f"   - {backup['backup_type']} at {backup['created_at']}")
    
    # Simulate pet evolution
    print(f"\n🦋 Evolving DigiPal to child stage...")
    loaded_pet.life_stage = LifeStage.CHILD
    loaded_pet.learned_commands.update({"play", "train"})
    loaded_pet.evolution_timer = 0.0
    
    # Add evolution interaction
    evolution_interaction = Interaction(
        user_input="grow up",
        interpreted_command="evolve",
        pet_response="*grows bigger* I'm a child now!",
        attribute_changes={"hp": 20, "mp": 10},
        success=True,
        result=InteractionResult.SUCCESS
    )
    loaded_pet.conversation_history.append(evolution_interaction)
    loaded_pet.hp += 20
    loaded_pet.mp += 10
    
    storage.save_pet(loaded_pet)
    print("✅ DigiPal evolved to child stage")
    
    # Get interaction history
    print(f"\n💬 Getting interaction history...")
    history = storage.get_interaction_history(user_id, limit=10)
    print(f"📜 Interaction History: {len(history)} interactions")
    
    for i, interaction in enumerate(history, 1):
        print(f"   {i}. '{interaction['user_input']}' -> '{interaction['pet_response']}'")
    
    # Database statistics
    print(f"\n📊 Database Statistics:")
    stats = storage.get_database_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Asset management demo
    print(f"\n🖼️  Asset Management Demo:")
    test_image_data = b"fake_image_data_for_demo"
    asset_path = storage.save_asset(user_id, "demo_pet_image.png", test_image_data)
    
    if asset_path:
        print(f"✅ Asset saved: {Path(asset_path).name}")
        
        assets = storage.get_user_assets(user_id)
        print(f"📁 User assets: {len(assets)} files")
        for asset in assets:
            print(f"   - {asset['filename']} ({asset['size']} bytes)")
    
    # Cleanup demo
    print(f"\n🧹 Cleanup:")
    print(f"   Database file: {db_path}")
    print(f"   Assets directory: {assets_path}")
    print(f"   Temporary directory: {temp_dir}")
    print("   (Files will be cleaned up automatically)")
    
    print(f"\n🎉 Storage demo completed successfully!")


if __name__ == "__main__":
    main()