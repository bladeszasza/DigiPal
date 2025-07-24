"""
Backup and recovery system for DigiPal data.

This module provides comprehensive backup and recovery functionality
with automatic scheduling, data validation, and recovery mechanisms.
"""

import json
import logging
import shutil
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import sqlite3
import hashlib

from ..core.exceptions import StorageError, RecoveryError, DigiPalException, ErrorSeverity
from ..core.error_handler import with_error_handling, with_retry, RetryConfig
from ..core.models import DigiPal

logger = logging.getLogger(__name__)


class BackupConfig:
    """Configuration for backup operations."""
    
    def __init__(
        self,
        backup_interval_hours: int = 6,
        max_backups: int = 10,
        compress_backups: bool = True,
        verify_backups: bool = True,
        auto_cleanup: bool = True,
        backup_on_critical_operations: bool = True
    ):
        """
        Initialize backup configuration.
        
        Args:
            backup_interval_hours: Hours between automatic backups
            max_backups: Maximum number of backups to keep
            compress_backups: Whether to compress backup files
            verify_backups: Whether to verify backup integrity
            auto_cleanup: Whether to automatically clean old backups
            backup_on_critical_operations: Whether to backup before critical operations
        """
        self.backup_interval_hours = backup_interval_hours
        self.max_backups = max_backups
        self.compress_backups = compress_backups
        self.verify_backups = verify_backups
        self.auto_cleanup = auto_cleanup
        self.backup_on_critical_operations = backup_on_critical_operations


class BackupMetadata:
    """Metadata for backup files."""
    
    def __init__(
        self,
        backup_id: str,
        timestamp: datetime,
        backup_type: str,
        file_path: str,
        checksum: str,
        size_bytes: int,
        user_id: Optional[str] = None,
        pet_id: Optional[str] = None,
        description: Optional[str] = None
    ):
        """Initialize backup metadata."""
        self.backup_id = backup_id
        self.timestamp = timestamp
        self.backup_type = backup_type
        self.file_path = file_path
        self.checksum = checksum
        self.size_bytes = size_bytes
        self.user_id = user_id
        self.pet_id = pet_id
        self.description = description
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'backup_id': self.backup_id,
            'timestamp': self.timestamp.isoformat(),
            'backup_type': self.backup_type,
            'file_path': self.file_path,
            'checksum': self.checksum,
            'size_bytes': self.size_bytes,
            'user_id': self.user_id,
            'pet_id': self.pet_id,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BackupMetadata':
        """Create from dictionary."""
        return cls(
            backup_id=data['backup_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            backup_type=data['backup_type'],
            file_path=data['file_path'],
            checksum=data['checksum'],
            size_bytes=data['size_bytes'],
            user_id=data.get('user_id'),
            pet_id=data.get('pet_id'),
            description=data.get('description')
        )


class BackupRecoveryManager:
    """Manages backup and recovery operations for DigiPal data."""
    
    def __init__(self, db_path: str, backup_dir: str, config: Optional[BackupConfig] = None):
        """
        Initialize backup and recovery manager.
        
        Args:
            db_path: Path to main database
            backup_dir: Directory for backup files
            config: Backup configuration
        """
        self.db_path = Path(db_path)
        self.backup_dir = Path(backup_dir)
        self.config = config or BackupConfig()
        
        # Create backup directory
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata storage
        self.metadata_file = self.backup_dir / "backup_metadata.json"
        self.metadata: Dict[str, BackupMetadata] = {}
        
        # Background backup thread
        self._backup_thread = None
        self._stop_backup_thread = False
        
        # Load existing metadata
        self._load_metadata()
        
        logger.info(f"BackupRecoveryManager initialized: {backup_dir}")
    
    def _load_metadata(self):
        """Load backup metadata from file."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                    self.metadata = {
                        k: BackupMetadata.from_dict(v) 
                        for k, v in metadata_dict.items()
                    }
                logger.info(f"Loaded {len(self.metadata)} backup metadata entries")
        except Exception as e:
            logger.error(f"Failed to load backup metadata: {e}")
            self.metadata = {}
    
    def _save_metadata(self):
        """Save backup metadata to file."""
        try:
            metadata_dict = {k: v.to_dict() for k, v in self.metadata.items()}
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save backup metadata: {e}")
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    @with_error_handling(fallback_value=False, context={'operation': 'create_backup'})
    @with_retry(RetryConfig(max_attempts=3, retry_on=[StorageError]))
    def create_backup(
        self, 
        backup_type: str = "manual", 
        user_id: Optional[str] = None,
        pet_id: Optional[str] = None,
        description: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Create a backup of the database.
        
        Args:
            backup_type: Type of backup (manual, automatic, pre_operation)
            user_id: Specific user ID to backup (None for full backup)
            pet_id: Specific pet ID to backup (None for user/full backup)
            description: Optional description for the backup
            
        Returns:
            Tuple of (success, backup_id)
        """
        try:
            # Generate backup ID
            backup_id = f"{backup_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(time.time()) % 10000:04d}"
            
            # Create backup filename
            backup_filename = f"backup_{backup_id}.db"
            backup_path = self.backup_dir / backup_filename
            
            logger.info(f"Creating backup: {backup_id}")
            
            if user_id or pet_id:
                # Partial backup
                success = self._create_partial_backup(backup_path, user_id, pet_id)
            else:
                # Full database backup
                success = self._create_full_backup(backup_path)
            
            if not success:
                raise StorageError(f"Failed to create backup: {backup_id}")
            
            # Calculate checksum and size
            checksum = self._calculate_checksum(backup_path)
            size_bytes = backup_path.stat().st_size
            
            # Verify backup if configured
            if self.config.verify_backups:
                if not self._verify_backup(backup_path):
                    backup_path.unlink()  # Delete corrupted backup
                    raise StorageError(f"Backup verification failed: {backup_id}")
            
            # Compress if configured
            if self.config.compress_backups:
                compressed_path = self._compress_backup(backup_path)
                if compressed_path:
                    backup_path.unlink()  # Remove uncompressed version
                    backup_path = compressed_path
                    checksum = self._calculate_checksum(backup_path)
                    size_bytes = backup_path.stat().st_size
            
            # Create metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=datetime.now(),
                backup_type=backup_type,
                file_path=str(backup_path),
                checksum=checksum,
                size_bytes=size_bytes,
                user_id=user_id,
                pet_id=pet_id,
                description=description
            )
            
            # Store metadata
            self.metadata[backup_id] = metadata
            self._save_metadata()
            
            # Cleanup old backups if configured
            if self.config.auto_cleanup:
                self._cleanup_old_backups()
            
            logger.info(f"Backup created successfully: {backup_id} ({size_bytes} bytes)")
            return True, backup_id
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise StorageError(f"Backup creation failed: {str(e)}")
    
    def _create_full_backup(self, backup_path: Path) -> bool:
        """Create a full database backup."""
        try:
            # Use SQLite backup API for consistent backup
            source_conn = sqlite3.connect(str(self.db_path))
            backup_conn = sqlite3.connect(str(backup_path))
            
            source_conn.backup(backup_conn)
            
            source_conn.close()
            backup_conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Full backup failed: {e}")
            return False
    
    def _create_partial_backup(self, backup_path: Path, user_id: Optional[str], pet_id: Optional[str]) -> bool:
        """Create a partial backup for specific user or pet."""
        try:
            # Create new database with same schema
            source_conn = sqlite3.connect(str(self.db_path))
            backup_conn = sqlite3.connect(str(backup_path))
            
            # Copy schema
            source_conn.backup(backup_conn)
            
            # Clear all data
            backup_conn.execute("DELETE FROM digipals")
            backup_conn.execute("DELETE FROM interactions")
            backup_conn.execute("DELETE FROM care_actions")
            backup_conn.execute("DELETE FROM users")
            
            # Copy specific data
            if pet_id:
                # Backup specific pet
                cursor = source_conn.execute("SELECT * FROM digipals WHERE id = ?", (pet_id,))
                pet_data = cursor.fetchone()
                if pet_data:
                    placeholders = ','.join(['?' for _ in pet_data])
                    backup_conn.execute(f"INSERT INTO digipals VALUES ({placeholders})", pet_data)
                    
                    # Copy related data
                    backup_conn.execute("INSERT INTO interactions SELECT * FROM interactions WHERE digipal_id = ?", (pet_id,))
                    backup_conn.execute("INSERT INTO care_actions SELECT * FROM care_actions WHERE digipal_id = ?", (pet_id,))
                    
                    # Copy user data
                    user_cursor = source_conn.execute("SELECT * FROM users WHERE id = (SELECT user_id FROM digipals WHERE id = ?)", (pet_id,))
                    user_data = user_cursor.fetchone()
                    if user_data:
                        placeholders = ','.join(['?' for _ in user_data])
                        backup_conn.execute(f"INSERT INTO users VALUES ({placeholders})", user_data)
            
            elif user_id:
                # Backup all pets for user
                backup_conn.execute("INSERT INTO users SELECT * FROM users WHERE id = ?", (user_id,))
                backup_conn.execute("INSERT INTO digipals SELECT * FROM digipals WHERE user_id = ?", (user_id,))
                backup_conn.execute("INSERT INTO interactions SELECT * FROM interactions WHERE user_id = ?", (user_id,))
                backup_conn.execute("INSERT INTO care_actions SELECT * FROM care_actions WHERE user_id = ?", (user_id,))
            
            backup_conn.commit()
            source_conn.close()
            backup_conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Partial backup failed: {e}")
            return False
    
    def _verify_backup(self, backup_path: Path) -> bool:
        """Verify backup integrity."""
        try:
            # Try to open and query the backup database
            conn = sqlite3.connect(str(backup_path))
            cursor = conn.cursor()
            
            # Check if main tables exist and are accessible
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            
            if table_count == 0:
                return False
            
            # Try to query main tables
            cursor.execute("SELECT COUNT(*) FROM digipals")
            cursor.execute("SELECT COUNT(*) FROM users")
            cursor.execute("SELECT COUNT(*) FROM interactions")
            
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return False
    
    def _compress_backup(self, backup_path: Path) -> Optional[Path]:
        """Compress backup file."""
        try:
            import gzip
            
            compressed_path = backup_path.with_suffix(backup_path.suffix + '.gz')
            
            with open(backup_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            return compressed_path
            
        except Exception as e:
            logger.error(f"Backup compression failed: {e}")
            return None
    
    def _cleanup_old_backups(self):
        """Clean up old backups based on configuration."""
        try:
            # Sort backups by timestamp
            sorted_backups = sorted(
                self.metadata.values(),
                key=lambda x: x.timestamp,
                reverse=True
            )
            
            # Keep only the most recent backups
            backups_to_delete = sorted_backups[self.config.max_backups:]
            
            for backup in backups_to_delete:
                try:
                    backup_path = Path(backup.file_path)
                    if backup_path.exists():
                        backup_path.unlink()
                    
                    # Remove from metadata
                    del self.metadata[backup.backup_id]
                    
                    logger.info(f"Cleaned up old backup: {backup.backup_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to cleanup backup {backup.backup_id}: {e}")
            
            if backups_to_delete:
                self._save_metadata()
                
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
    
    @with_error_handling(fallback_value=False, context={'operation': 'restore_backup'})
    def restore_backup(self, backup_id: str, target_db_path: Optional[str] = None) -> bool:
        """
        Restore from a backup.
        
        Args:
            backup_id: ID of backup to restore
            target_db_path: Target database path (None to restore to original)
            
        Returns:
            True if restore successful
        """
        try:
            if backup_id not in self.metadata:
                raise RecoveryError(f"Backup not found: {backup_id}")
            
            backup_metadata = self.metadata[backup_id]
            backup_path = Path(backup_metadata.file_path)
            
            if not backup_path.exists():
                raise RecoveryError(f"Backup file not found: {backup_path}")
            
            # Verify backup integrity
            if self.config.verify_backups:
                current_checksum = self._calculate_checksum(backup_path)
                if current_checksum != backup_metadata.checksum:
                    raise RecoveryError(f"Backup checksum mismatch: {backup_id}")
            
            target_path = Path(target_db_path) if target_db_path else self.db_path
            
            logger.info(f"Restoring backup {backup_id} to {target_path}")
            
            # Create backup of current database before restore
            if target_path.exists():
                current_backup_path = target_path.with_suffix('.pre_restore_backup')
                shutil.copy2(target_path, current_backup_path)
                logger.info(f"Created pre-restore backup: {current_backup_path}")
            
            # Decompress if needed
            restore_source = backup_path
            if backup_path.suffix == '.gz':
                restore_source = self._decompress_backup(backup_path)
                if not restore_source:
                    raise RecoveryError(f"Failed to decompress backup: {backup_id}")
            
            # Restore database
            shutil.copy2(restore_source, target_path)
            
            # Clean up temporary decompressed file
            if restore_source != backup_path:
                restore_source.unlink()
            
            # Verify restored database
            if not self._verify_backup(target_path):
                raise RecoveryError(f"Restored database verification failed: {backup_id}")
            
            logger.info(f"Backup restored successfully: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Backup restore failed: {e}")
            raise RecoveryError(f"Failed to restore backup {backup_id}: {str(e)}")
    
    def _decompress_backup(self, compressed_path: Path) -> Optional[Path]:
        """Decompress a backup file."""
        try:
            import gzip
            
            decompressed_path = compressed_path.with_suffix('')
            
            with gzip.open(compressed_path, 'rb') as f_in:
                with open(decompressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            return decompressed_path
            
        except Exception as e:
            logger.error(f"Backup decompression failed: {e}")
            return None
    
    def list_backups(self, user_id: Optional[str] = None, backup_type: Optional[str] = None) -> List[BackupMetadata]:
        """
        List available backups.
        
        Args:
            user_id: Filter by user ID
            backup_type: Filter by backup type
            
        Returns:
            List of backup metadata
        """
        backups = list(self.metadata.values())
        
        if user_id:
            backups = [b for b in backups if b.user_id == user_id]
        
        if backup_type:
            backups = [b for b in backups if b.backup_type == backup_type]
        
        # Sort by timestamp (newest first)
        backups.sort(key=lambda x: x.timestamp, reverse=True)
        
        return backups
    
    def delete_backup(self, backup_id: str) -> bool:
        """
        Delete a backup.
        
        Args:
            backup_id: ID of backup to delete
            
        Returns:
            True if deletion successful
        """
        try:
            if backup_id not in self.metadata:
                return False
            
            backup_metadata = self.metadata[backup_id]
            backup_path = Path(backup_metadata.file_path)
            
            if backup_path.exists():
                backup_path.unlink()
            
            del self.metadata[backup_id]
            self._save_metadata()
            
            logger.info(f"Deleted backup: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete backup {backup_id}: {e}")
            return False
    
    def start_automatic_backups(self):
        """Start automatic backup thread."""
        if self._backup_thread and self._backup_thread.is_alive():
            logger.warning("Automatic backups already running")
            return
        
        self._stop_backup_thread = False
        self._backup_thread = threading.Thread(target=self._backup_loop, daemon=True)
        self._backup_thread.start()
        
        logger.info("Automatic backups started")
    
    def stop_automatic_backups(self):
        """Stop automatic backup thread."""
        self._stop_backup_thread = True
        if self._backup_thread:
            self._backup_thread.join(timeout=5)
        
        logger.info("Automatic backups stopped")
    
    def _backup_loop(self):
        """Background loop for automatic backups."""
        logger.info("Automatic backup loop started")
        
        while not self._stop_backup_thread:
            try:
                # Check if it's time for a backup
                last_backup_time = self._get_last_automatic_backup_time()
                now = datetime.now()
                
                if not last_backup_time or (now - last_backup_time).total_seconds() >= (self.config.backup_interval_hours * 3600):
                    logger.info("Creating automatic backup")
                    success, backup_id = self.create_backup(
                        backup_type="automatic",
                        description="Scheduled automatic backup"
                    )
                    
                    if success:
                        logger.info(f"Automatic backup created: {backup_id}")
                    else:
                        logger.error("Automatic backup failed")
                
                # Sleep for 1 hour before checking again
                time.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in automatic backup loop: {e}")
                time.sleep(3600)  # Continue after error
        
        logger.info("Automatic backup loop stopped")
    
    def _get_last_automatic_backup_time(self) -> Optional[datetime]:
        """Get timestamp of last automatic backup."""
        automatic_backups = [
            b for b in self.metadata.values() 
            if b.backup_type == "automatic"
        ]
        
        if not automatic_backups:
            return None
        
        return max(b.timestamp for b in automatic_backups)
    
    def get_backup_statistics(self) -> Dict[str, Any]:
        """Get backup statistics."""
        backups = list(self.metadata.values())
        
        if not backups:
            return {
                'total_backups': 0,
                'total_size_bytes': 0,
                'backup_types': {},
                'oldest_backup': None,
                'newest_backup': None
            }
        
        total_size = sum(b.size_bytes for b in backups)
        backup_types = {}
        
        for backup in backups:
            backup_types[backup.backup_type] = backup_types.get(backup.backup_type, 0) + 1
        
        oldest = min(backups, key=lambda x: x.timestamp)
        newest = max(backups, key=lambda x: x.timestamp)
        
        return {
            'total_backups': len(backups),
            'total_size_bytes': total_size,
            'backup_types': backup_types,
            'oldest_backup': {
                'id': oldest.backup_id,
                'timestamp': oldest.timestamp.isoformat(),
                'type': oldest.backup_type
            },
            'newest_backup': {
                'id': newest.backup_id,
                'timestamp': newest.timestamp.isoformat(),
                'type': newest.backup_type
            }
        }
    
    def create_pre_operation_backup(self, operation_name: str, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Create a backup before a critical operation.
        
        Args:
            operation_name: Name of the operation
            context: Additional context
            
        Returns:
            Backup ID if successful, None otherwise
        """
        if not self.config.backup_on_critical_operations:
            return None
        
        try:
            description = f"Pre-operation backup for: {operation_name}"
            if context:
                description += f" (context: {json.dumps(context)})"
            
            success, backup_id = self.create_backup(
                backup_type="pre_operation",
                description=description
            )
            
            if success:
                logger.info(f"Pre-operation backup created: {backup_id} for {operation_name}")
                return backup_id
            
        except Exception as e:
            logger.error(f"Failed to create pre-operation backup for {operation_name}: {e}")
        
        return None