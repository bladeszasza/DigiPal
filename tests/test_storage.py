"""
Comprehensive tests for DigiPal storage and persistence layer.
"""

import json
import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from digipal.storage import StorageManager, DatabaseSchema, DatabaseConnection
from digipal.core.models import DigiPal, Interaction, CareAction, AttributeModifier
from digipal.core.enums import (
    EggType, LifeStage, CareActionType, AttributeType, 
    CommandType, InteractionResult
)


class TestDatabaseSchema:
    """Test database schema creation and migrations."""
    
    def test_create_database(self):
        """Test database creation with all tables and indexes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            
            # Create database
            success = DatabaseSchema.create_database(str(db_path))
            assert success
            assert db_path.exists()
            
            # Verify tables exist
            conn = DatabaseConnection(str(db_path))
            tables = conn.execute_query(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            table_names = {row['name'] for row in tables}
            
            expected_tables = {
                'users', 'digipals', 'interactions', 
                'care_actions', 'backups', 'schema_migrations'
            }
            assert expected_tables.issubset(table_names)
    
    def test_get_schema_version(self):
        """Test schema version retrieval."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            
            # Non-existent database should return version 0
            version = DatabaseSchema.get_schema_version(str(db_path))
            assert version == 0
            
            # Create database and check version
            DatabaseSchema.create_database(str(db_path))
            version = DatabaseSchema.get_schema_version(str(db_path))
            assert version == DatabaseSchema.CURRENT_VERSION
    
    def test_migrate_database(self):
        """Test database migration system."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            
            # Create database
            DatabaseSchema.create_database(str(db_path))
            
            # Migration should succeed (no-op for current version)
            success = DatabaseSchema.migrate_database(str(db_path))
            assert success


class TestDatabaseConnection:
    """Test database connection management."""
    
    def test_connection_initialization(self):
        """Test database connection initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            
            conn = DatabaseConnection(str(db_path))
            assert Path(db_path).exists()
            
            # Test connection works
            result = conn.execute_query("SELECT 1 as test")
            assert result[0]['test'] == 1
    
    def test_execute_query(self):
        """Test query execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            conn = DatabaseConnection(str(db_path))
            
            # Insert test data
            conn.execute_update(
                "INSERT INTO users (id, username) VALUES (?, ?)",
                ("test_user", "testuser")
            )
            
            # Query data
            results = conn.execute_query(
                "SELECT * FROM users WHERE id = ?",
                ("test_user",)
            )
            
            assert len(results) == 1
            assert results[0]['id'] == "test_user"
            assert results[0]['username'] == "testuser"
    
    def test_execute_transaction(self):
        """Test transaction execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            conn = DatabaseConnection(str(db_path))
            
            queries = [
                ("INSERT INTO users (id, username) VALUES (?, ?)", ("user1", "test1")),
                ("INSERT INTO users (id, username) VALUES (?, ?)", ("user2", "test2"))
            ]
            
            success = conn.execute_transaction(queries)
            assert success
            
            # Verify both records exist
            results = conn.execute_query("SELECT COUNT(*) as count FROM users")
            assert results[0]['count'] == 2


class TestStorageManager:
    """Test StorageManager CRUD operations."""
    
    @pytest.fixture
    def storage_manager(self):
        """Create a temporary StorageManager for testing."""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test.db"
        assets_path = Path(temp_dir) / "assets"
        
        manager = StorageManager(str(db_path), str(assets_path))
        
        yield manager
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_digipal(self):
        """Create a sample DigiPal for testing."""
        return DigiPal(
            id="test_pet_123",
            user_id="test_user_456",
            name="TestPal",
            egg_type=EggType.RED,
            life_stage=LifeStage.BABY,
            generation=1,
            hp=100,
            mp=50,
            offense=15,
            defense=8,
            speed=12,
            brains=8,
            discipline=10,
            happiness=60,
            weight=22,
            care_mistakes=0,
            energy=90,
            birth_time=datetime.now() - timedelta(hours=2),
            last_interaction=datetime.now() - timedelta(minutes=30),
            evolution_timer=1.5,
            learned_commands={"eat", "sleep", "good", "bad"},
            personality_traits={"playful": 0.8, "curious": 0.6},
            current_image_path="/path/to/image.png",
            image_generation_prompt="A cute baby DigiPal"
        )
    
    def test_user_management(self, storage_manager):
        """Test user creation, retrieval, and updates."""
        user_id = "test_user_123"
        username = "testuser"
        token = "hf_test_token"
        
        # Create user
        success = storage_manager.create_user(user_id, username, token)
        assert success
        
        # Retrieve user
        user = storage_manager.get_user(user_id)
        assert user is not None
        assert user['id'] == user_id
        assert user['username'] == username
        assert user['huggingface_token'] == token
        
        # Update session
        session_data = {"last_action": "login", "preferences": {"theme": "dark"}}
        success = storage_manager.update_user_session(user_id, session_data)
        assert success
        
        # Verify session update
        updated_user = storage_manager.get_user(user_id)
        assert updated_user['session_data'] == session_data
    
    def test_pet_crud_operations(self, storage_manager, sample_digipal):
        """Test DigiPal CRUD operations."""
        # Create user first
        storage_manager.create_user(sample_digipal.user_id, "testuser")
        
        # Save pet
        success = storage_manager.save_pet(sample_digipal)
        assert success
        
        # Load pet
        loaded_pet = storage_manager.load_pet(sample_digipal.user_id)
        assert loaded_pet is not None
        assert loaded_pet.id == sample_digipal.id
        assert loaded_pet.name == sample_digipal.name
        assert loaded_pet.egg_type == sample_digipal.egg_type
        assert loaded_pet.life_stage == sample_digipal.life_stage
        assert loaded_pet.hp == sample_digipal.hp
        assert loaded_pet.learned_commands == sample_digipal.learned_commands
        assert loaded_pet.personality_traits == sample_digipal.personality_traits
        
        # Get pet by ID
        pet_by_id = storage_manager.get_pet(sample_digipal.id)
        assert pet_by_id is not None
        assert pet_by_id.id == sample_digipal.id
        
        # Update pet
        sample_digipal.happiness = 80
        sample_digipal.energy = 70
        success = storage_manager.save_pet(sample_digipal)
        assert success
        
        # Verify update
        updated_pet = storage_manager.get_pet(sample_digipal.id)
        assert updated_pet.happiness == 80
        assert updated_pet.energy == 70
        
        # Get user pets
        user_pets = storage_manager.get_user_pets(sample_digipal.user_id)
        assert len(user_pets) == 1
        assert user_pets[0].id == sample_digipal.id
        
        # Delete pet (soft delete)
        success = storage_manager.delete_pet(sample_digipal.id)
        assert success
        
        # Verify pet is inactive
        active_pets = storage_manager.get_user_pets(sample_digipal.user_id)
        assert len(active_pets) == 0
        
        # But still exists when including inactive
        all_pets = storage_manager.get_user_pets(sample_digipal.user_id, include_inactive=True)
        assert len(all_pets) == 1
    
    def test_interaction_history(self, storage_manager, sample_digipal):
        """Test interaction history management."""
        # Create user and pet
        storage_manager.create_user(sample_digipal.user_id, "testuser")
        
        # Add some interactions to the pet
        interactions = [
            Interaction(
                timestamp=datetime.now() - timedelta(minutes=10),
                user_input="eat",
                interpreted_command="feed",
                pet_response="*munch munch* Happy!",
                attribute_changes={"happiness": 5, "energy": -10},
                success=True,
                result=InteractionResult.SUCCESS
            ),
            Interaction(
                timestamp=datetime.now() - timedelta(minutes=5),
                user_input="good boy",
                interpreted_command="praise",
                pet_response="*happy chirp*",
                attribute_changes={"happiness": 10, "discipline": 2},
                success=True,
                result=InteractionResult.SUCCESS
            )
        ]
        
        sample_digipal.conversation_history = interactions
        
        # Save pet with interactions
        success = storage_manager.save_pet(sample_digipal)
        assert success
        
        # Load pet and verify interactions
        loaded_pet = storage_manager.load_pet(sample_digipal.user_id)
        assert len(loaded_pet.conversation_history) == 2
        
        # Verify interaction details
        first_interaction = loaded_pet.conversation_history[0]
        assert first_interaction.user_input == "eat"
        assert first_interaction.interpreted_command == "feed"
        assert first_interaction.attribute_changes == {"happiness": 5, "energy": -10}
        
        # Get interaction history directly
        history = storage_manager.get_interaction_history(sample_digipal.user_id)
        assert len(history) == 2
    
    def test_care_actions(self, storage_manager, sample_digipal):
        """Test care action logging."""
        # Create user and pet
        storage_manager.create_user(sample_digipal.user_id, "testuser")
        storage_manager.save_pet(sample_digipal)
        
        # Create care action
        care_action = CareAction(
            name="Basic Training",
            action_type=CareActionType.TRAIN,
            energy_cost=20,
            happiness_change=5,
            attribute_modifiers=[
                AttributeModifier(AttributeType.OFFENSE, 2),
                AttributeModifier(AttributeType.SPEED, 1)
            ]
        )
        
        attribute_changes = {"offense": 2, "speed": 1, "energy": -20, "happiness": 5}
        
        # Save care action
        success = storage_manager.save_care_action(
            sample_digipal.id, 
            sample_digipal.user_id, 
            care_action, 
            attribute_changes
        )
        assert success
        
        # Get care history
        care_history = storage_manager.get_care_history(sample_digipal.id)
        assert len(care_history) == 1
        
        care_record = care_history[0]
        assert care_record['action_type'] == CareActionType.TRAIN.value
        assert care_record['action_name'] == "Basic Training"
        assert care_record['energy_cost'] == 20
        assert care_record['happiness_change'] == 5
        assert care_record['attribute_changes'] == attribute_changes
    
    def test_backup_and_recovery(self, storage_manager, sample_digipal):
        """Test backup and recovery system."""
        # Create user and pet with interactions
        storage_manager.create_user(sample_digipal.user_id, "testuser")
        
        # Add interaction history
        sample_digipal.conversation_history = [
            Interaction(
                user_input="test command",
                interpreted_command="test",
                pet_response="test response",
                attribute_changes={"happiness": 1}
            )
        ]
        
        storage_manager.save_pet(sample_digipal)
        
        # Create backup
        success = storage_manager.create_backup(sample_digipal.user_id)
        assert success
        
        # Verify backup exists
        backups = storage_manager.get_backups(sample_digipal.user_id)
        assert len(backups) == 1
        assert backups[0]['backup_type'] == "manual"
        
        # Modify pet data
        sample_digipal.happiness = 99
        sample_digipal.name = "ModifiedPal"
        storage_manager.save_pet(sample_digipal)
        
        # Restore from backup
        success = storage_manager.restore_backup(sample_digipal.user_id)
        assert success
        
        # Verify restoration
        restored_pet = storage_manager.load_pet(sample_digipal.user_id)
        assert restored_pet.name == "TestPal"  # Original name
        assert len(restored_pet.conversation_history) == 1
    
    def test_automatic_backup(self, storage_manager, sample_digipal):
        """Test automatic backup creation and cleanup."""
        # Create user and pet
        storage_manager.create_user(sample_digipal.user_id, "testuser")
        storage_manager.save_pet(sample_digipal)
        
        # Create multiple automatic backups
        for i in range(12):  # More than the keep_count of 10
            success = storage_manager.create_automatic_backup(sample_digipal.user_id)
            assert success
        
        # Verify only 10 backups are kept
        backups = storage_manager.get_backups(sample_digipal.user_id, limit=20)
        assert len(backups) <= 10
        
        # All should be automatic backups
        for backup in backups:
            assert backup['backup_type'] == "automatic"
    
    def test_asset_management(self, storage_manager):
        """Test asset file management."""
        user_id = "test_user"
        filename = "test_image.png"
        test_data = b"fake image data"
        
        # Save asset
        asset_path = storage_manager.save_asset(user_id, filename, test_data)
        assert asset_path != ""
        assert Path(asset_path).exists()
        
        # Get user assets
        assets = storage_manager.get_user_assets(user_id)
        assert len(assets) == 1
        assert assets[0]['filename'] == filename
        assert assets[0]['size'] == len(test_data)
    
    def test_database_maintenance(self, storage_manager):
        """Test database maintenance operations."""
        # Test vacuum
        success = storage_manager.vacuum_database()
        assert success
        
        # Test stats
        stats = storage_manager.get_database_stats()
        assert 'users_count' in stats
        assert 'digipals_count' in stats
        assert 'database_size_bytes' in stats
        assert 'database_size_mb' in stats
    
    def test_error_handling(self, storage_manager):
        """Test error handling in storage operations."""
        # Test loading non-existent pet
        pet = storage_manager.load_pet("non_existent_user")
        assert pet is None
        
        # Test getting non-existent user
        user = storage_manager.get_user("non_existent_user")
        assert user is None
        
        # Test deleting non-existent pet
        success = storage_manager.delete_pet("non_existent_pet")
        assert not success


class TestStorageIntegration:
    """Integration tests for storage layer."""
    
    def test_complete_pet_lifecycle(self):
        """Test complete pet lifecycle with storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            assets_path = Path(temp_dir) / "assets"
            
            storage = StorageManager(str(db_path), str(assets_path))
            
            # Create user
            user_id = "lifecycle_user"
            storage.create_user(user_id, "lifecycleuser")
            
            # Create egg
            pet = DigiPal(
                user_id=user_id,
                name="LifecyclePal",
                egg_type=EggType.BLUE,
                life_stage=LifeStage.EGG
            )
            
            # Save egg
            storage.save_pet(pet)
            
            # Hatch to baby
            pet.life_stage = LifeStage.BABY
            pet.learned_commands = {"eat", "sleep", "good", "bad"}
            storage.save_pet(pet)
            
            # Add interactions
            interaction = Interaction(
                user_input="eat",
                interpreted_command="feed",
                pet_response="*happy eating sounds*",
                attribute_changes={"happiness": 5}
            )
            pet.conversation_history.append(interaction)
            storage.save_pet(pet)
            
            # Evolve to child
            pet.life_stage = LifeStage.CHILD
            pet.learned_commands.update({"play", "train"})
            storage.save_pet(pet)
            
            # Create backup at child stage
            storage.create_backup(user_id, "child_stage")
            
            # Continue evolution
            pet.life_stage = LifeStage.TEEN
            pet.happiness = 90
            storage.save_pet(pet)
            
            # Load pet and verify complete state
            loaded_pet = storage.load_pet(user_id)
            assert loaded_pet.life_stage == LifeStage.TEEN
            assert loaded_pet.happiness == 90
            assert "train" in loaded_pet.learned_commands
            assert len(loaded_pet.conversation_history) == 1
            
            # Verify backups exist
            backups = storage.get_backups(user_id)
            assert len(backups) >= 1
            
            # Test restoration to child stage
            child_backup = next(b for b in backups if b['backup_type'] == 'child_stage')
            storage.restore_backup(user_id, child_backup['id'])
            
            restored_pet = storage.load_pet(user_id)
            assert restored_pet.life_stage == LifeStage.CHILD


if __name__ == "__main__":
    pytest.main([__file__, "-v"])