"""
File Storage System for Optional Persistence

Provides long-term storage for files when users check "Save for later use".
Files are stored with user-specified retention periods (max 90 days) and
automatically cleaned up after expiration.
"""

import os
import json
import uuid
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class FileStore:
    """
    Persistent file storage with automatic cleanup.
    
    Files are stored on disk with metadata tracking:
    - Upload date
    - Expiry date (based on retention period)
    - Original filename
    - File size
    
    A background thread runs daily to remove expired files.
    """
    
    def __init__(self, storage_dir: str = "./stored_files"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.storage_dir / "metadata.json"
        self.files: Dict[str, Dict] = {}
        
        # Load existing metadata
        self._load_metadata()
        
        # Start cleanup thread
        self._start_cleanup_thread()
        
        logger.info(f"FileStore initialized at {self.storage_dir}")
    
    def _load_metadata(self):
        """Load metadata from disk"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.files = json.load(f)
                logger.info(f"Loaded metadata for {len(self.files)} files")
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
                self.files = {}
        else:
            self.files = {}
    
    def _save_metadata(self):
        """Save metadata to disk"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.files, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def save_file(self, file_bytes: bytes, filename: str, retention_days: int) -> str:
        """
        Save file with expiration date.
        
        Args:
            file_bytes: File content as bytes
            filename: Original filename
            retention_days: Number of days to retain (max 90)
        
        Returns:
            file_id: Unique identifier for the saved file
        """
        # Validate retention period
        if retention_days > 90:
            logger.warning(f"Retention period {retention_days} exceeds max, capping at 90 days")
            retention_days = 90
        if retention_days < 1:
            retention_days = 1
        
        # Generate unique file ID
        file_id = uuid.uuid4().hex
        
        # Calculate expiry date
        upload_date = datetime.now()
        expiry_date = upload_date + timedelta(days=retention_days)
        
        # Save file to disk
        file_path = self.storage_dir / file_id
        try:
            file_path.write_bytes(file_bytes)
        except Exception as e:
            logger.error(f"Failed to save file {file_id}: {e}")
            raise
        
        # Store metadata
        self.files[file_id] = {
            "filename": filename,
            "upload_date": upload_date.isoformat(),
            "expiry_date": expiry_date.isoformat(),
            "retention_days": retention_days,
            "size_bytes": len(file_bytes)
        }
        self._save_metadata()
        
        logger.info(f"Saved file {file_id} ({filename}), expires {expiry_date.date()}")
        return file_id
    
    def get_file(self, file_id: str) -> bytes:
        """
        Retrieve file by ID.
        
        Args:
            file_id: File identifier
        
        Returns:
            File content as bytes
        
        Raises:
            FileNotFoundError: If file doesn't exist or has expired
        """
        if file_id not in self.files:
            raise FileNotFoundError(f"File {file_id} not found")
        
        metadata = self.files[file_id]
        
        # Check if expired
        expiry_date = datetime.fromisoformat(metadata["expiry_date"])
        if datetime.now() > expiry_date:
            logger.warning(f"File {file_id} has expired, removing")
            self.delete_file(file_id)
            raise FileNotFoundError(f"File {file_id} has expired, you may need to re-upload it.")
        
        # Read file
        file_path = self.storage_dir / file_id
        if not file_path.exists():
            logger.error(f"File {file_id} metadata exists but file is missing")
            del self.files[file_id]
            self._save_metadata()
            raise FileNotFoundError(f"File {file_id} data is missing")
        
        return file_path.read_bytes()
    
    def get_metadata(self, file_id: str) -> Dict:
        """Get file metadata without reading file"""
        if file_id not in self.files:
            raise FileNotFoundError(f"File {file_id} not found")
        return self.files[file_id].copy()
    
    def delete_file(self, file_id: str):
        """
        Delete file and metadata.
        
        Args:
            file_id: File identifier
        """
        if file_id not in self.files:
            return
        
        # Remove file from disk
        file_path = self.storage_dir / file_id
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception as e:
                logger.error(f"Failed to delete file {file_id}: {e}")
        
        # Remove metadata
        filename = self.files[file_id].get("filename", "unknown")
        del self.files[file_id]
        self._save_metadata()
        
        logger.info(f"Deleted file {file_id} ({filename})")
    
    def list_active_files(self) -> List[Dict]:
        """
        List all non-expired files with metadata.
        
        Returns:
            List of file metadata dictionaries with file_id included
        """
        now = datetime.now()
        active_files = []
        
        for file_id, metadata in self.files.items():
            expiry_date = datetime.fromisoformat(metadata["expiry_date"])
            if now <= expiry_date:
                file_info = metadata.copy()
                file_info["file_id"] = file_id
                
                # Add days remaining
                days_remaining = (expiry_date - now).days
                file_info["days_remaining"] = days_remaining
                
                active_files.append(file_info)
        
        # Sort by expiry date (soonest first)
        active_files.sort(key=lambda x: x["expiry_date"])
        
        return active_files
    
    def cleanup_expired(self):
        """Remove all expired files"""
        now = datetime.now()
        expired_ids = []
        
        for file_id, metadata in self.files.items():
            expiry_date = datetime.fromisoformat(metadata["expiry_date"])
            if now > expiry_date:
                expired_ids.append(file_id)
        
        if expired_ids:
            logger.info(f"Cleaning up {len(expired_ids)} expired files")
            for file_id in expired_ids:
                self.delete_file(file_id)
        else:
            logger.debug("No expired files to clean up")
    
    def _cleanup_loop(self):
        """Background thread that runs daily cleanup"""
        while True:
            try:
                # Run cleanup
                self.cleanup_expired()
                
                # Sleep for 24 hours
                time.sleep(86400)
            except Exception as e:
                logger.error(f"Cleanup thread error: {e}")
                # Sleep for 1 hour on error, then retry
                time.sleep(3600)
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="FileStoreCleanup"
        )
        cleanup_thread.start()
        logger.info("Started background cleanup thread")
    
    def get_stats(self) -> Dict:
        """Get storage statistics"""
        total_files = len(self.files)
        active_files = len(self.list_active_files())
        
        total_size = sum(m["size_bytes"] for m in self.files.values())
        
        return {
            "total_files": total_files,
            "active_files": active_files,
            "expired_files": total_files - active_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2)
        }
