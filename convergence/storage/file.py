"""
File-based storage backend for The Convergence framework.

Provides persistent storage using individual files per key.
Good for small datasets, debugging, and version control scenarios.
"""

import aiofiles
import pickle
import json
from typing import Any, List, Optional
from pathlib import Path
import asyncio

from convergence.storage.base import (
    StorageProtocol,
    StorageError,
    StorageSerializationError,
)


class FileStorage:
    """
    File-based storage implementation.
    
    Each key is stored as a separate file in the base directory.
    Keys are sanitized to safe filenames.
    
    Features:
    - Async file I/O via aiofiles
    - File locking for concurrent access safety
    - Automatic directory creation
    - Configurable serialization (pickle or json)
    
    Good for:
    - Small datasets (< 1000 keys)
    - Debugging (easy to inspect files)
    - Version control (can track individual changes)
    - Development/testing
    
    Not recommended for:
    - Large datasets (filesystem overhead)
    - High-frequency updates (disk I/O bottleneck)
    
    Usage:
        storage = FileStorage("./data/storage")
        await storage.save("agent:1", agent_data)
        data = await storage.load("agent:1")
    """
    
    def __init__(
        self,
        base_dir: str | Path,
        serializer: str = "pickle",
        file_extension: str = ".pkl"
    ):
        """
        Initialize file-based storage.
        
        Args:
            base_dir: Base directory for storage files
            serializer: Serialization method ("pickle" or "json")
            file_extension: File extension for storage files
        """
        self.base_dir = Path(base_dir)
        self.serializer = serializer
        self.file_extension = file_extension
        self._locks: dict[str, asyncio.Lock] = {}
        
        # Create base directory if it doesn't exist
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _sanitize_key(self, key: str) -> str:
        """
        Sanitize key to safe filename.
        
        Replaces unsafe characters with underscores.
        
        Args:
            key: Original key
            
        Returns:
            Sanitized filename-safe key
        """
        # Replace common separators and unsafe chars
        safe_key = key.replace("/", "_")
        safe_key = safe_key.replace("\\", "_")
        safe_key = safe_key.replace(":", "_")
        safe_key = safe_key.replace("*", "_")
        safe_key = safe_key.replace("?", "_")
        safe_key = safe_key.replace("\"", "_")
        safe_key = safe_key.replace("<", "_")
        safe_key = safe_key.replace(">", "_")
        safe_key = safe_key.replace("|", "_")
        return safe_key
    
    def _key_to_path(self, key: str) -> Path:
        """
        Convert key to file path.
        
        Args:
            key: Storage key
            
        Returns:
            Path to file for this key
        """
        safe_key = self._sanitize_key(key)
        return self.base_dir / f"{safe_key}{self.file_extension}"
    
    def _get_lock(self, key: str) -> asyncio.Lock:
        """
        Get lock for a key (for concurrent access safety).
        
        Args:
            key: Storage key
            
        Returns:
            Lock for this key
        """
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]
    
    def _serialize(self, value: Any) -> bytes:
        """
        Serialize value to bytes.
        
        Args:
            value: Python object to serialize
            
        Returns:
            Serialized bytes
            
        Raises:
            StorageSerializationError: If serialization fails
        """
        try:
            if self.serializer == "pickle":
                return pickle.dumps(value)
            elif self.serializer == "json":
                return json.dumps(value, indent=2).encode("utf-8")
            else:
                raise ValueError(f"Unknown serializer: {self.serializer}")
        except Exception as e:
            raise StorageSerializationError(
                f"Failed to serialize value: {e}"
            ) from e
    
    def _deserialize(self, data: bytes) -> Any:
        """
        Deserialize bytes to Python object.
        
        Args:
            data: Serialized bytes
            
        Returns:
            Deserialized Python object
            
        Raises:
            StorageSerializationError: If deserialization fails
        """
        try:
            if self.serializer == "pickle":
                return pickle.loads(data)
            elif self.serializer == "json":
                return json.loads(data.decode("utf-8"))
            else:
                raise ValueError(f"Unknown serializer: {self.serializer}")
        except Exception as e:
            raise StorageSerializationError(
                f"Failed to deserialize value: {e}"
            ) from e
    
    async def save(self, key: str, value: Any) -> None:
        """
        Save a value with a key.
        
        Args:
            key: Unique identifier
            value: Python object to store
            
        Raises:
            StorageError: If save operation fails
        """
        lock = self._get_lock(key)
        async with lock:
            try:
                path = self._key_to_path(key)
                serialized = self._serialize(value)
                
                # Write atomically using temp file + rename
                temp_path = path.with_suffix(path.suffix + ".tmp")
                
                async with aiofiles.open(temp_path, "wb") as f:
                    await f.write(serialized)
                
                # Atomic rename (POSIX guarantees atomicity)
                temp_path.rename(path)
                
            except StorageSerializationError:
                raise  # Re-raise as-is
            except Exception as e:
                raise StorageError(
                    f"Failed to save key '{key}': {e}"
                ) from e
    
    async def load(self, key: str) -> Any:
        """
        Load a value by key.
        
        Args:
            key: Unique identifier
            
        Returns:
            Stored Python object
            
        Raises:
            KeyError: If key doesn't exist
            StorageError: If load operation fails
        """
        lock = self._get_lock(key)
        async with lock:
            try:
                path = self._key_to_path(key)
                
                if not path.exists():
                    raise KeyError(f"Key not found: {key}")
                
                async with aiofiles.open(path, "rb") as f:
                    data = await f.read()
                
                return self._deserialize(data)
                
            except KeyError:
                raise  # Re-raise as-is
            except StorageSerializationError:
                raise  # Re-raise as-is
            except Exception as e:
                raise StorageError(
                    f"Failed to load key '{key}': {e}"
                ) from e
    
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists.
        
        Args:
            key: Unique identifier
            
        Returns:
            True if key exists, False otherwise
        """
        try:
            path = self._key_to_path(key)
            return path.exists()
        except Exception as e:
            raise StorageError(
                f"Failed to check existence of key '{key}': {e}"
            ) from e
    
    async def delete(self, key: str) -> None:
        """
        Delete a key.
        
        Args:
            key: Unique identifier
            
        Note:
            Silently succeeds if key doesn't exist (idempotent).
        """
        lock = self._get_lock(key)
        async with lock:
            try:
                path = self._key_to_path(key)
                
                if path.exists():
                    path.unlink()
                
                # Clean up lock
                if key in self._locks:
                    del self._locks[key]
                    
            except Exception as e:
                raise StorageError(
                    f"Failed to delete key '{key}': {e}"
                ) from e
    
    async def list_keys(self, prefix: str = "") -> List[str]:
        """
        List all keys, optionally filtered by prefix.
        
        Args:
            prefix: Optional prefix to filter keys
            
        Returns:
            List of matching keys
            
        Note:
            Returns original keys (before sanitization).
            Keys are reconstructed from filenames.
        """
        try:
            keys = []
            
            # Iterate through all files in base directory
            for path in self.base_dir.glob(f"*{self.file_extension}"):
                # Extract key from filename
                key = path.stem  # Filename without extension
                
                # Filter by prefix if specified
                if not prefix or key.startswith(prefix):
                    keys.append(key)
            
            return sorted(keys)
            
        except Exception as e:
            raise StorageError(
                f"Failed to list keys with prefix '{prefix}': {e}"
            ) from e
    
    async def close(self) -> None:
        """
        Clean up resources.
        
        For file storage, this clears the lock registry.
        Idempotent - safe to call multiple times.
        """
        self._locks.clear()
    
    # Additional utility methods
    
    async def count_keys(self, prefix: str = "") -> int:
        """
        Count keys, optionally filtered by prefix.
        
        Args:
            prefix: Optional prefix to filter keys
            
        Returns:
            Number of matching keys
        """
        keys = await self.list_keys(prefix)
        return len(keys)
    
    async def clear(self, prefix: str = "") -> int:
        """
        Delete all keys, optionally filtered by prefix.
        
        Args:
            prefix: Optional prefix to filter keys
            
        Returns:
            Number of keys deleted
        """
        keys = await self.list_keys(prefix)
        
        for key in keys:
            await self.delete(key)
        
        return len(keys)
    
    def get_storage_size(self) -> int:
        """
        Get total storage size in bytes.
        
        Returns:
            Total size of all storage files
        """
        total_size = 0
        
        for path in self.base_dir.glob(f"*{self.file_extension}"):
            try:
                total_size += path.stat().st_size
            except Exception:
                pass  # Ignore errors for individual files
        
        return total_size


# Verify protocol implementation at module level
def _verify_protocol() -> None:
    """Verify that FileStorage implements StorageProtocol."""
    from convergence.storage.base import StorageProtocol
    import tempfile
    
    # This will raise TypeError at import time if protocol is not implemented
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = FileStorage(tmpdir)
        assert isinstance(storage, StorageProtocol), \
            "FileStorage must implement StorageProtocol"


_verify_protocol()

