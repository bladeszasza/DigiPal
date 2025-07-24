"""
StorageManager class for DigiPal data persistence with CRUD operations.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from ..core.models import DigiPal, Interaction, CareAction
from ..core.enums import EggType, LifeStage, CareActionType, InteractionResult
from ..core.exceptions import StorageError, RecoveryError
from ..core.error_handler import with_error_handling, with_retry, RetryConfig
from .database import DatabaseConnection
from .backup_recovery import BackupRecoveryManager, BackupConfig

logger = logging.getLogger(__name__)


class StorageManager:
    """
    Manages all data persistence operations for DigiPal application.
    Provides CRUD operations for pets, users, interactions, and backup/recovery.
    """
    
    def __init__(self, db_path: str, assets_path: str = "assets"):
        """
        Initialize StorageManager with database and assets paths.
        
        Args:
            db_path: Path to SQLite database file
            assets_path: Path to assets directory for images and backups
        """
        self.db_path = db_path
        self.assets_path = Path(assets_path)
        self.assets_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database connection
        self.db = DatabaseConnection(db_path)
        
        # Create subdirectories for assets
        (self.assets_path / "images").mkdir(exist_ok=True)
        (self.assets_path / "backups").mkdir(exist_ok=True)
        
        # Initialize backup and recovery manager
        backup_config = BackupConfig(
            backup_interval_hours=6,
            max_backups=10,
            verify_backups=True,
            backup_on_critical_operations=True
        )
        self.backup_manager = BackupRecoveryManager(
            db_path, 
            str(self.assets_path / "backups"), 
            backup_config
        )
        
        # Start automatic backups
        self.backup_manager.start_automatic_backups()
        
        logger.info(f"StorageManager initialized with db: {db_path}, assets: {assets_path}")
    
    # User Management
    def create_user(self, user_id: str, username: str, huggingface_token: str = "") -> bool:
        """Create a new user record."""
        try:
            query = '''
                INSERT INTO users (id, username, huggingface_token, last_login, session_data)
                VALUES (?, ?, ?, ?, ?)
            '''
            session_data = json.dumps({"created": datetime.now().isoformat()})
            
            affected = self.db.execute_update(
                query, 
                (user_id, username, huggingface_token, datetime.now(), session_data)
            )
            
            if affected > 0:
                logger.info(f"Created user: {user_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to create user {user_id}: {e}")
            return False
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user information by ID."""
        try:
            query = 'SELECT * FROM users WHERE id = ?'
            results = self.db.execute_query(query, (user_id,))
            
            if results:
                row = results[0]
                return {
                    'id': row['id'],
                    'username': row['username'],
                    'huggingface_token': row['huggingface_token'],
                    'created_at': row['created_at'],
                    'last_login': row['last_login'],
                    'session_data': json.loads(row['session_data']) if row['session_data'] else {}
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user {user_id}: {e}")
            return None
    
    def update_user_session(self, user_id: str, session_data: Dict[str, Any]) -> bool:
        """Update user session data and last login time."""
        try:
            query = '''
                UPDATE users 
                SET last_login = ?, session_data = ?
                WHERE id = ?
            '''
            
            affected = self.db.execute_update(
                query,
                (datetime.now(), json.dumps(session_data), user_id)
            )
            
            return affected > 0
            
        except Exception as e:
            logger.error(f"Failed to update user session {user_id}: {e}")
            return False
    
    # DigiPal CRUD Operations
    @with_error_handling(fallback_value=False, context={'operation': 'save_pet'})
    @with_retry(RetryConfig(max_attempts=3, retry_on=[StorageError]))
    def save_pet(self, pet: DigiPal) -> bool:
        """Save or update a DigiPal to the database."""
        try:
            # Create pre-operation backup for critical pet data changes
            backup_id = self.backup_manager.create_pre_operation_backup(
                "save_pet", 
                {"pet_id": pet.id, "user_id": pet.user_id}
            )
            
            # Check if pet exists
            existing = self.get_pet(pet.id)
            
            if existing:
                result = self._update_pet(pet)
            else:
                result = self._insert_pet(pet)
            
            if not result:
                raise StorageError(f"Failed to save pet {pet.id}")
            
            return result
                
        except Exception as e:
            logger.error(f"Failed to save pet {pet.id}: {e}")
            raise StorageError(f"Pet save operation failed: {str(e)}")
    
    def _insert_pet(self, pet: DigiPal) -> bool:
        """Insert a new DigiPal record."""
        query = '''
            INSERT INTO digipals (
                id, user_id, name, egg_type, life_stage, generation,
                hp, mp, offense, defense, speed, brains,
                discipline, happiness, weight, care_mistakes, energy,
                birth_time, last_interaction, evolution_timer,
                learned_commands, personality_traits,
                current_image_path, image_generation_prompt,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        
        params = (
            pet.id, pet.user_id, pet.name, pet.egg_type.value, pet.life_stage.value, pet.generation,
            pet.hp, pet.mp, pet.offense, pet.defense, pet.speed, pet.brains,
            pet.discipline, pet.happiness, pet.weight, pet.care_mistakes, pet.energy,
            pet.birth_time, pet.last_interaction, pet.evolution_timer,
            json.dumps(list(pet.learned_commands)), json.dumps(pet.personality_traits),
            pet.current_image_path, pet.image_generation_prompt,
            datetime.now()
        )
        
        affected = self.db.execute_update(query, params)
        
        if affected > 0:
            # Save conversation history as interactions
            self._save_conversation_history(pet)
            logger.info(f"Inserted pet: {pet.id}")
            return True
        
        return False
    
    def _update_pet(self, pet: DigiPal) -> bool:
        """Update an existing DigiPal record."""
        query = '''
            UPDATE digipals SET
                name = ?, egg_type = ?, life_stage = ?, generation = ?,
                hp = ?, mp = ?, offense = ?, defense = ?, speed = ?, brains = ?,
                discipline = ?, happiness = ?, weight = ?, care_mistakes = ?, energy = ?,
                birth_time = ?, last_interaction = ?, evolution_timer = ?,
                learned_commands = ?, personality_traits = ?,
                current_image_path = ?, image_generation_prompt = ?,
                updated_at = ?
            WHERE id = ?
        '''
        
        params = (
            pet.name, pet.egg_type.value, pet.life_stage.value, pet.generation,
            pet.hp, pet.mp, pet.offense, pet.defense, pet.speed, pet.brains,
            pet.discipline, pet.happiness, pet.weight, pet.care_mistakes, pet.energy,
            pet.birth_time, pet.last_interaction, pet.evolution_timer,
            json.dumps(list(pet.learned_commands)), json.dumps(pet.personality_traits),
            pet.current_image_path, pet.image_generation_prompt,
            datetime.now(), pet.id
        )
        
        affected = self.db.execute_update(query, params)
        
        if affected > 0:
            # Update conversation history
            self._save_conversation_history(pet)
            logger.info(f"Updated pet: {pet.id}")
            return True
        
        return False
    
    def load_pet(self, user_id: str) -> Optional[DigiPal]:
        """Load the active DigiPal for a user."""
        try:
            query = '''
                SELECT * FROM digipals 
                WHERE user_id = ? AND is_active = 1 
                ORDER BY updated_at DESC 
                LIMIT 1
            '''
            
            results = self.db.execute_query(query, (user_id,))
            
            if not results:
                return None
            
            row = results[0]
            
            # Convert database row to DigiPal object
            pet_data = {
                'id': row['id'],
                'user_id': row['user_id'],
                'name': row['name'],
                'egg_type': EggType(row['egg_type']),
                'life_stage': LifeStage(row['life_stage']),
                'generation': row['generation'],
                'hp': row['hp'],
                'mp': row['mp'],
                'offense': row['offense'],
                'defense': row['defense'],
                'speed': row['speed'],
                'brains': row['brains'],
                'discipline': row['discipline'],
                'happiness': row['happiness'],
                'weight': row['weight'],
                'care_mistakes': row['care_mistakes'],
                'energy': row['energy'],
                'birth_time': datetime.fromisoformat(row['birth_time']),
                'last_interaction': datetime.fromisoformat(row['last_interaction']),
                'evolution_timer': row['evolution_timer'],
                'learned_commands': set(json.loads(row['learned_commands']) if row['learned_commands'] else []),
                'personality_traits': json.loads(row['personality_traits']) if row['personality_traits'] else {},
                'current_image_path': row['current_image_path'] or "",
                'image_generation_prompt': row['image_generation_prompt'] or "",
                'conversation_history': []
            }
            
            pet = DigiPal(**pet_data)
            
            # Load conversation history
            pet.conversation_history = self._load_conversation_history(pet.id)
            
            logger.info(f"Loaded pet: {pet.id} for user: {user_id}")
            return pet
            
        except Exception as e:
            logger.error(f"Failed to load pet for user {user_id}: {e}")
            return None
    
    def get_pet(self, pet_id: str) -> Optional[DigiPal]:
        """Get a specific DigiPal by ID."""
        try:
            query = 'SELECT * FROM digipals WHERE id = ?'
            results = self.db.execute_query(query, (pet_id,))
            
            if not results:
                return None
            
            row = results[0]
            
            pet_data = {
                'id': row['id'],
                'user_id': row['user_id'],
                'name': row['name'],
                'egg_type': EggType(row['egg_type']),
                'life_stage': LifeStage(row['life_stage']),
                'generation': row['generation'],
                'hp': row['hp'],
                'mp': row['mp'],
                'offense': row['offense'],
                'defense': row['defense'],
                'speed': row['speed'],
                'brains': row['brains'],
                'discipline': row['discipline'],
                'happiness': row['happiness'],
                'weight': row['weight'],
                'care_mistakes': row['care_mistakes'],
                'energy': row['energy'],
                'birth_time': datetime.fromisoformat(row['birth_time']),
                'last_interaction': datetime.fromisoformat(row['last_interaction']),
                'evolution_timer': row['evolution_timer'],
                'learned_commands': set(json.loads(row['learned_commands']) if row['learned_commands'] else []),
                'personality_traits': json.loads(row['personality_traits']) if row['personality_traits'] else {},
                'current_image_path': row['current_image_path'] or "",
                'image_generation_prompt': row['image_generation_prompt'] or "",
                'conversation_history': []
            }
            
            pet = DigiPal(**pet_data)
            pet.conversation_history = self._load_conversation_history(pet.id)
            
            return pet
            
        except Exception as e:
            logger.error(f"Failed to get pet {pet_id}: {e}")
            return None
    
    def delete_pet(self, pet_id: str) -> bool:
        """Soft delete a DigiPal (mark as inactive)."""
        try:
            query = 'UPDATE digipals SET is_active = 0, updated_at = ? WHERE id = ?'
            affected = self.db.execute_update(query, (datetime.now(), pet_id))
            
            if affected > 0:
                logger.info(f"Deleted pet: {pet_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete pet {pet_id}: {e}")
            return False
    
    def get_user_pets(self, user_id: str, include_inactive: bool = False) -> List[DigiPal]:
        """Get all pets for a user."""
        try:
            if include_inactive:
                query = 'SELECT * FROM digipals WHERE user_id = ? ORDER BY updated_at DESC'
            else:
                query = 'SELECT * FROM digipals WHERE user_id = ? AND is_active = 1 ORDER BY updated_at DESC'
            
            results = self.db.execute_query(query, (user_id,))
            pets = []
            
            for row in results:
                pet_data = {
                    'id': row['id'],
                    'user_id': row['user_id'],
                    'name': row['name'],
                    'egg_type': EggType(row['egg_type']),
                    'life_stage': LifeStage(row['life_stage']),
                    'generation': row['generation'],
                    'hp': row['hp'],
                    'mp': row['mp'],
                    'offense': row['offense'],
                    'defense': row['defense'],
                    'speed': row['speed'],
                    'brains': row['brains'],
                    'discipline': row['discipline'],
                    'happiness': row['happiness'],
                    'weight': row['weight'],
                    'care_mistakes': row['care_mistakes'],
                    'energy': row['energy'],
                    'birth_time': datetime.fromisoformat(row['birth_time']),
                    'last_interaction': datetime.fromisoformat(row['last_interaction']),
                    'evolution_timer': row['evolution_timer'],
                    'learned_commands': set(json.loads(row['learned_commands']) if row['learned_commands'] else []),
                    'personality_traits': json.loads(row['personality_traits']) if row['personality_traits'] else {},
                    'current_image_path': row['current_image_path'] or "",
                    'image_generation_prompt': row['image_generation_prompt'] or "",
                    'conversation_history': []
                }
                
                pet = DigiPal(**pet_data)
                # Note: Not loading full conversation history for performance
                pets.append(pet)
            
            return pets
            
        except Exception as e:
            logger.error(f"Failed to get pets for user {user_id}: {e}")
            return []
    
    # Interaction History Management
    def save_interaction_history(self, interactions: List[Interaction]) -> bool:
        """Save multiple interactions to the database."""
        if not interactions:
            return True
        
        try:
            queries = []
            for interaction in interactions:
                query = '''
                    INSERT INTO interactions (
                        digipal_id, user_id, timestamp, user_input, interpreted_command,
                        pet_response, attribute_changes, success, result
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                '''
                
                # Extract digipal_id and user_id from context (would need to be passed)
                # For now, we'll handle this in _save_conversation_history
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save interaction history: {e}")
            return False
    
    def _save_conversation_history(self, pet: DigiPal) -> bool:
        """Save pet's conversation history to interactions table."""
        if not pet.conversation_history:
            return True
        
        try:
            # First, delete existing interactions for this pet to avoid duplicates
            delete_query = 'DELETE FROM interactions WHERE digipal_id = ?'
            self.db.execute_update(delete_query, (pet.id,))
            
            # Insert all interactions
            queries = []
            for interaction in pet.conversation_history:
                query = '''
                    INSERT INTO interactions (
                        digipal_id, user_id, timestamp, user_input, interpreted_command,
                        pet_response, attribute_changes, success, result
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                '''
                
                params = (
                    pet.id, pet.user_id, interaction.timestamp, interaction.user_input,
                    interaction.interpreted_command, interaction.pet_response,
                    json.dumps(interaction.attribute_changes), interaction.success,
                    interaction.result.value
                )
                
                queries.append((query, params))
            
            return self.db.execute_transaction(queries)
            
        except Exception as e:
            logger.error(f"Failed to save conversation history for pet {pet.id}: {e}")
            return False
    
    def _load_conversation_history(self, pet_id: str, limit: int = 100) -> List[Interaction]:
        """Load conversation history for a pet."""
        try:
            query = '''
                SELECT * FROM interactions 
                WHERE digipal_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            '''
            
            results = self.db.execute_query(query, (pet_id, limit))
            interactions = []
            
            for row in results:
                interaction = Interaction(
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    user_input=row['user_input'],
                    interpreted_command=row['interpreted_command'] or "",
                    pet_response=row['pet_response'] or "",
                    attribute_changes=json.loads(row['attribute_changes']) if row['attribute_changes'] else {},
                    success=bool(row['success']),
                    result=InteractionResult(row['result']) if row['result'] else InteractionResult.SUCCESS
                )
                interactions.append(interaction)
            
            # Reverse to get chronological order
            return list(reversed(interactions))
            
        except Exception as e:
            logger.error(f"Failed to load conversation history for pet {pet_id}: {e}")
            return []
    
    def get_interaction_history(self, user_id: str, pet_id: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get interaction history for a user or specific pet."""
        try:
            if pet_id:
                query = '''
                    SELECT * FROM interactions 
                    WHERE user_id = ? AND digipal_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                '''
                params = (user_id, pet_id, limit)
            else:
                query = '''
                    SELECT * FROM interactions 
                    WHERE user_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                '''
                params = (user_id, limit)
            
            results = self.db.execute_query(query, params)
            
            return [
                {
                    'id': row['id'],
                    'digipal_id': row['digipal_id'],
                    'user_id': row['user_id'],
                    'timestamp': row['timestamp'],
                    'user_input': row['user_input'],
                    'interpreted_command': row['interpreted_command'],
                    'pet_response': row['pet_response'],
                    'attribute_changes': json.loads(row['attribute_changes']) if row['attribute_changes'] else {},
                    'success': bool(row['success']),
                    'result': row['result']
                }
                for row in results
            ]
            
        except Exception as e:
            logger.error(f"Failed to get interaction history: {e}")
            return []
    
    # Care Actions Management
    def save_care_action(self, pet_id: str, user_id: str, care_action: CareAction, 
                        attribute_changes: Dict[str, int], success: bool = True) -> bool:
        """Save a care action to the database."""
        try:
            query = '''
                INSERT INTO care_actions (
                    digipal_id, user_id, action_type, action_name, timestamp,
                    energy_cost, happiness_change, attribute_changes, success
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            params = (
                pet_id, user_id, care_action.action_type.value, care_action.name,
                datetime.now(), care_action.energy_cost, care_action.happiness_change,
                json.dumps(attribute_changes), success
            )
            
            affected = self.db.execute_update(query, params)
            return affected > 0
            
        except Exception as e:
            logger.error(f"Failed to save care action: {e}")
            return False
    
    def get_care_history(self, pet_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get care action history for a pet."""
        try:
            query = '''
                SELECT * FROM care_actions 
                WHERE digipal_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            '''
            
            results = self.db.execute_query(query, (pet_id, limit))
            
            return [
                {
                    'id': row['id'],
                    'digipal_id': row['digipal_id'],
                    'user_id': row['user_id'],
                    'action_type': row['action_type'],
                    'action_name': row['action_name'],
                    'timestamp': row['timestamp'],
                    'energy_cost': row['energy_cost'],
                    'happiness_change': row['happiness_change'],
                    'attribute_changes': json.loads(row['attribute_changes']) if row['attribute_changes'] else {},
                    'success': bool(row['success'])
                }
                for row in results
            ]
            
        except Exception as e:
            logger.error(f"Failed to get care history for pet {pet_id}: {e}")
            return []
    
    # Asset Management
    def get_user_assets(self, user_id: str) -> List[Dict[str, str]]:
        """Get list of asset references for a user."""
        try:
            user_assets_path = self.assets_path / "images" / user_id
            if not user_assets_path.exists():
                return []
            
            assets = []
            for asset_file in user_assets_path.iterdir():
                if asset_file.is_file() and asset_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif']:
                    assets.append({
                        'filename': asset_file.name,
                        'path': str(asset_file),
                        'size': asset_file.stat().st_size,
                        'modified': datetime.fromtimestamp(asset_file.stat().st_mtime).isoformat()
                    })
            
            return assets
            
        except Exception as e:
            logger.error(f"Failed to get assets for user {user_id}: {e}")
            return []
    
    def save_asset(self, user_id: str, filename: str, data: bytes) -> str:
        """Save an asset file and return the path."""
        try:
            user_assets_path = self.assets_path / "images" / user_id
            user_assets_path.mkdir(parents=True, exist_ok=True)
            
            asset_path = user_assets_path / filename
            asset_path.write_bytes(data)
            
            logger.info(f"Saved asset: {asset_path}")
            return str(asset_path)
            
        except Exception as e:
            logger.error(f"Failed to save asset {filename} for user {user_id}: {e}")
            return ""  
  
    # Backup and Recovery System
    def create_backup(self, user_id: str, backup_type: str = "manual", pet_id: str = None) -> bool:
        """Create a backup of user data."""
        try:
            backup_data = {
                'timestamp': datetime.now().isoformat(),
                'backup_type': backup_type,
                'user_data': self.get_user(user_id),
                'pets': [],
                'interactions': [],
                'care_actions': []
            }
            
            # Get pets data
            if pet_id:
                pet = self.get_pet(pet_id)
                if pet:
                    backup_data['pets'] = [pet.to_dict()]
                    backup_data['interactions'] = self.get_interaction_history(user_id, pet_id, limit=1000)
                    backup_data['care_actions'] = self.get_care_history(pet_id, limit=1000)
            else:
                # Backup all user pets
                pets = self.get_user_pets(user_id, include_inactive=True)
                backup_data['pets'] = [pet.to_dict() for pet in pets]
                backup_data['interactions'] = self.get_interaction_history(user_id, limit=1000)
                
                # Get care actions for all pets
                for pet in pets:
                    pet_care_actions = self.get_care_history(pet.id, limit=1000)
                    backup_data['care_actions'].extend(pet_care_actions)
            
            # Save backup to database
            backup_json = json.dumps(backup_data, indent=2)
            
            query = '''
                INSERT INTO backups (user_id, digipal_id, backup_type, backup_data)
                VALUES (?, ?, ?, ?)
            '''
            
            affected = self.db.execute_update(query, (user_id, pet_id, backup_type, backup_json))
            
            if affected > 0:
                # Also save to file for additional safety
                backup_filename = f"backup_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                backup_file_path = self.assets_path / "backups" / backup_filename
                
                backup_file_path.write_text(backup_json)
                
                # Update backup record with file path
                update_query = 'UPDATE backups SET file_path = ? WHERE user_id = ? AND created_at = (SELECT MAX(created_at) FROM backups WHERE user_id = ?)'
                self.db.execute_update(update_query, (str(backup_file_path), user_id, user_id))
                
                logger.info(f"Created backup for user {user_id}: {backup_filename}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to create backup for user {user_id}: {e}")
            return False
    
    def restore_backup(self, user_id: str, backup_id: int = None) -> bool:
        """Restore user data from backup."""
        try:
            # Get backup data
            if backup_id:
                query = 'SELECT * FROM backups WHERE id = ? AND user_id = ?'
                results = self.db.execute_query(query, (backup_id, user_id))
            else:
                # Get most recent backup
                query = '''
                    SELECT * FROM backups 
                    WHERE user_id = ? 
                    ORDER BY created_at DESC 
                    LIMIT 1
                '''
                results = self.db.execute_query(query, (user_id,))
            
            if not results:
                logger.error(f"No backup found for user {user_id}")
                return False
            
            backup_row = results[0]
            backup_data = json.loads(backup_row['backup_data'])
            
            # Start transaction for restoration
            queries = []
            
            # Restore user data
            if backup_data.get('user_data'):
                user_data = backup_data['user_data']
                user_query = '''
                    INSERT OR REPLACE INTO users (id, username, huggingface_token, created_at, last_login, session_data)
                    VALUES (?, ?, ?, ?, ?, ?)
                '''
                user_params = (
                    user_data['id'], user_data['username'], user_data['huggingface_token'],
                    user_data['created_at'], user_data['last_login'], 
                    json.dumps(user_data['session_data'])
                )
                queries.append((user_query, user_params))
            
            # Restore pets
            for pet_data in backup_data.get('pets', []):
                pet = DigiPal.from_dict(pet_data)
                
                pet_query = '''
                    INSERT OR REPLACE INTO digipals (
                        id, user_id, name, egg_type, life_stage, generation,
                        hp, mp, offense, defense, speed, brains,
                        discipline, happiness, weight, care_mistakes, energy,
                        birth_time, last_interaction, evolution_timer,
                        learned_commands, personality_traits,
                        current_image_path, image_generation_prompt,
                        created_at, updated_at, is_active
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                '''
                
                pet_params = (
                    pet.id, pet.user_id, pet.name, pet.egg_type.value, pet.life_stage.value, pet.generation,
                    pet.hp, pet.mp, pet.offense, pet.defense, pet.speed, pet.brains,
                    pet.discipline, pet.happiness, pet.weight, pet.care_mistakes, pet.energy,
                    pet.birth_time, pet.last_interaction, pet.evolution_timer,
                    json.dumps(list(pet.learned_commands)), json.dumps(pet.personality_traits),
                    pet.current_image_path, pet.image_generation_prompt,
                    datetime.now(), datetime.now(), 1
                )
                queries.append((pet_query, pet_params))
            
            # Clear existing interactions before restoring
            clear_interactions_query = 'DELETE FROM interactions WHERE user_id = ?'
            queries.append((clear_interactions_query, (user_id,)))
            
            # Restore interactions
            for interaction_data in backup_data.get('interactions', []):
                interaction_query = '''
                    INSERT INTO interactions (
                        digipal_id, user_id, timestamp, user_input, interpreted_command,
                        pet_response, attribute_changes, success, result
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                '''
                
                interaction_params = (
                    interaction_data['digipal_id'], interaction_data['user_id'],
                    interaction_data['timestamp'], interaction_data['user_input'],
                    interaction_data['interpreted_command'], interaction_data['pet_response'],
                    json.dumps(interaction_data['attribute_changes']),
                    interaction_data['success'], interaction_data['result']
                )
                queries.append((interaction_query, interaction_params))
            
            # Restore care actions
            for care_data in backup_data.get('care_actions', []):
                care_query = '''
                    INSERT OR REPLACE INTO care_actions (
                        digipal_id, user_id, action_type, action_name, timestamp,
                        energy_cost, happiness_change, attribute_changes, success
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                '''
                
                care_params = (
                    care_data['digipal_id'], care_data['user_id'],
                    care_data['action_type'], care_data['action_name'],
                    care_data['timestamp'], care_data['energy_cost'],
                    care_data['happiness_change'], json.dumps(care_data['attribute_changes']),
                    care_data['success']
                )
                queries.append((care_query, care_params))
            
            # Execute all restoration queries in transaction
            success = self.db.execute_transaction(queries)
            
            if success:
                logger.info(f"Successfully restored backup for user {user_id}")
                return True
            else:
                logger.error(f"Failed to restore backup for user {user_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to restore backup for user {user_id}: {e}")
            return False
    
    def get_backups(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get list of available backups for a user."""
        try:
            query = '''
                SELECT id, user_id, digipal_id, backup_type, created_at, file_path
                FROM backups 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            '''
            
            results = self.db.execute_query(query, (user_id, limit))
            
            return [
                {
                    'id': row['id'],
                    'user_id': row['user_id'],
                    'digipal_id': row['digipal_id'],
                    'backup_type': row['backup_type'],
                    'created_at': row['created_at'],
                    'file_path': row['file_path']
                }
                for row in results
            ]
            
        except Exception as e:
            logger.error(f"Failed to get backups for user {user_id}: {e}")
            return []
    
    def delete_old_backups(self, user_id: str, keep_count: int = 5) -> bool:
        """Delete old backups, keeping only the most recent ones."""
        try:
            # Get all backups for user, ordered by creation time (newest first)
            query = '''
                SELECT id, file_path FROM backups 
                WHERE user_id = ? 
                ORDER BY created_at DESC
            '''
            
            results = self.db.execute_query(query, (user_id,))
            
            if len(results) <= keep_count:
                return True  # No old backups to delete
            
            # Get backups to delete (skip the first keep_count)
            backups_to_delete = results[keep_count:]
            
            # Delete backup files
            for row in backups_to_delete:
                if row['file_path']:
                    try:
                        Path(row['file_path']).unlink(missing_ok=True)
                    except Exception as e:
                        logger.warning(f"Failed to delete backup file {row['file_path']}: {e}")
            
            # Delete database records
            backup_ids = [str(row['id']) for row in backups_to_delete]
            if backup_ids:
                placeholders = ','.join(['?' for _ in backup_ids])
                delete_query = f'DELETE FROM backups WHERE id IN ({placeholders})'
                affected = self.db.execute_update(delete_query, tuple(backup_ids))
                
                logger.info(f"Deleted {affected} old backups for user {user_id}")
                return affected > 0
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete old backups for user {user_id}: {e}")
            return False
    
    def create_automatic_backup(self, user_id: str) -> bool:
        """Create an automatic backup and clean up old ones."""
        try:
            # Create backup
            success = self.create_backup(user_id, backup_type="automatic")
            
            if success:
                # Clean up old automatic backups (keep last 10)
                self.delete_old_backups(user_id, keep_count=10)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to create automatic backup for user {user_id}: {e}")
            return False
    
    # Database Maintenance
    def vacuum_database(self) -> bool:
        """Optimize database by running VACUUM."""
        try:
            with self.db.get_connection() as conn:
                conn.execute('VACUUM')
                conn.commit()
            
            logger.info("Database vacuum completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to vacuum database: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            stats = {}
            
            # Table row counts
            tables = ['users', 'digipals', 'interactions', 'care_actions', 'backups']
            for table in tables:
                query = f'SELECT COUNT(*) as count FROM {table}'
                result = self.db.execute_query(query)
                stats[f'{table}_count'] = result[0]['count'] if result else 0
            
            # Database file size
            db_path = Path(self.db_path)
            if db_path.exists():
                stats['database_size_bytes'] = db_path.stat().st_size
                stats['database_size_mb'] = round(stats['database_size_bytes'] / (1024 * 1024), 2)
            
            # Assets directory size
            if self.assets_path.exists():
                total_size = sum(f.stat().st_size for f in self.assets_path.rglob('*') if f.is_file())
                stats['assets_size_bytes'] = total_size
                stats['assets_size_mb'] = round(total_size / (1024 * 1024), 2)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
    
    def close(self):
        """Close database connections and cleanup."""
        # SQLite connections are closed automatically with context managers
        # This method is here for future cleanup if needed
        logger.info("StorageManager closed")