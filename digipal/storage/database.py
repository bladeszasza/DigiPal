"""
Database schema and migration system for DigiPal storage.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class DatabaseSchema:
    """Manages database schema creation and migrations."""
    
    # Current schema version
    CURRENT_VERSION = 1
    
    @staticmethod
    def get_schema_sql() -> Dict[str, str]:
        """Get SQL statements for creating all tables."""
        return {
            'users': '''
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    huggingface_token TEXT,
                    username TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    session_data TEXT  -- JSON blob for session information
                )
            ''',
            
            'digipals': '''
                CREATE TABLE IF NOT EXISTS digipals (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    egg_type TEXT NOT NULL,
                    life_stage TEXT NOT NULL,
                    generation INTEGER DEFAULT 1,
                    
                    -- Primary Attributes
                    hp INTEGER DEFAULT 100,
                    mp INTEGER DEFAULT 50,
                    offense INTEGER DEFAULT 10,
                    defense INTEGER DEFAULT 10,
                    speed INTEGER DEFAULT 10,
                    brains INTEGER DEFAULT 10,
                    
                    -- Secondary Attributes
                    discipline INTEGER DEFAULT 0,
                    happiness INTEGER DEFAULT 50,
                    weight INTEGER DEFAULT 20,
                    care_mistakes INTEGER DEFAULT 0,
                    energy INTEGER DEFAULT 100,
                    
                    -- Lifecycle Management
                    birth_time TIMESTAMP NOT NULL,
                    last_interaction TIMESTAMP NOT NULL,
                    evolution_timer REAL DEFAULT 0.0,
                    
                    -- Memory and Context (JSON blobs)
                    learned_commands TEXT,  -- JSON array
                    personality_traits TEXT,  -- JSON object
                    
                    -- Visual Representation
                    current_image_path TEXT,
                    image_generation_prompt TEXT,
                    
                    -- Metadata
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''',
            
            'interactions': '''
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    digipal_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    user_input TEXT NOT NULL,
                    interpreted_command TEXT,
                    pet_response TEXT,
                    attribute_changes TEXT,  -- JSON object
                    success BOOLEAN DEFAULT 1,
                    result TEXT,  -- InteractionResult enum value
                    
                    FOREIGN KEY (digipal_id) REFERENCES digipals (id),
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''',
            
            'care_actions': '''
                CREATE TABLE IF NOT EXISTS care_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    digipal_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    action_name TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    energy_cost INTEGER DEFAULT 0,
                    happiness_change INTEGER DEFAULT 0,
                    attribute_changes TEXT,  -- JSON object
                    success BOOLEAN DEFAULT 1,
                    
                    FOREIGN KEY (digipal_id) REFERENCES digipals (id),
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''',
            
            'backups': '''
                CREATE TABLE IF NOT EXISTS backups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    digipal_id TEXT,
                    backup_type TEXT NOT NULL,  -- 'full', 'incremental', 'manual'
                    backup_data TEXT NOT NULL,  -- JSON blob
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    file_path TEXT,  -- Optional file path for large backups
                    
                    FOREIGN KEY (user_id) REFERENCES users (id),
                    FOREIGN KEY (digipal_id) REFERENCES digipals (id)
                )
            ''',
            
            'schema_migrations': '''
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT
                )
            '''
        }
    
    @staticmethod
    def get_indexes_sql() -> List[str]:
        """Get SQL statements for creating database indexes."""
        return [
            'CREATE INDEX IF NOT EXISTS idx_digipals_user_id ON digipals (user_id)',
            'CREATE INDEX IF NOT EXISTS idx_digipals_active ON digipals (user_id, is_active)',
            'CREATE INDEX IF NOT EXISTS idx_interactions_digipal ON interactions (digipal_id, timestamp)',
            'CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions (user_id, timestamp)',
            'CREATE INDEX IF NOT EXISTS idx_care_actions_digipal ON care_actions (digipal_id, timestamp)',
            'CREATE INDEX IF NOT EXISTS idx_backups_user ON backups (user_id, created_at)',
            'CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username ON users (username)'
        ]
    
    @staticmethod
    def create_database(db_path: str) -> bool:
        """Create database with all tables and indexes."""
        try:
            # Ensure directory exists
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(db_path) as conn:
                # Enable foreign key constraints
                conn.execute('PRAGMA foreign_keys = ON')
                
                # Create all tables
                schema_sql = DatabaseSchema.get_schema_sql()
                for table_name, sql in schema_sql.items():
                    conn.execute(sql)
                    logger.info(f"Created table: {table_name}")
                
                # Create indexes
                for index_sql in DatabaseSchema.get_indexes_sql():
                    conn.execute(index_sql)
                
                # Record schema version
                conn.execute(
                    'INSERT OR REPLACE INTO schema_migrations (version, description) VALUES (?, ?)',
                    (DatabaseSchema.CURRENT_VERSION, 'Initial schema creation')
                )
                
                conn.commit()
                logger.info(f"Database created successfully at {db_path}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create database: {e}")
            return False
    
    @staticmethod
    def get_schema_version(db_path: str) -> int:
        """Get current schema version from database."""
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute('SELECT MAX(version) FROM schema_migrations')
                result = cursor.fetchone()
                return result[0] if result[0] is not None else 0
        except Exception:
            return 0
    
    @staticmethod
    def migrate_database(db_path: str) -> bool:
        """Apply any pending database migrations."""
        current_version = DatabaseSchema.get_schema_version(db_path)
        target_version = DatabaseSchema.CURRENT_VERSION
        
        if current_version >= target_version:
            logger.info("Database is up to date")
            return True
        
        logger.info(f"Migrating database from version {current_version} to {target_version}")
        
        # Future migrations would be implemented here
        # For now, we only have version 1
        
        return True


class DatabaseConnection:
    """Manages database connections with proper error handling."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_database_exists()
    
    def _ensure_database_exists(self):
        """Ensure database exists and is properly initialized."""
        if not Path(self.db_path).exists():
            DatabaseSchema.create_database(self.db_path)
        else:
            # Check if database has required tables
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='digipals'")
                    if not cursor.fetchone():
                        # Database exists but doesn't have required tables, create them
                        DatabaseSchema.create_database(self.db_path)
                    else:
                        # Check if migration is needed
                        DatabaseSchema.migrate_database(self.db_path)
            except Exception as e:
                logger.error(f"Error checking database structure: {e}")
                # Recreate database if there's an error
                DatabaseSchema.create_database(self.db_path)
    
    def get_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper configuration."""
        conn = sqlite3.connect(self.db_path)
        conn.execute('PRAGMA foreign_keys = ON')
        conn.row_factory = sqlite3.Row  # Enable dict-like access to rows
        return conn
    
    def execute_query(self, query: str, params: tuple = ()) -> List[sqlite3.Row]:
        """Execute a SELECT query and return results."""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return cursor.fetchall()
    
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Execute an INSERT/UPDATE/DELETE query and return affected rows."""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.rowcount
    
    def execute_transaction(self, queries: List[tuple]) -> bool:
        """Execute multiple queries in a transaction."""
        try:
            with self.get_connection() as conn:
                for query, params in queries:
                    conn.execute(query, params)
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            return False