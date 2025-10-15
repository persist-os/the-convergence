"""
SQLite storage backend for The Convergence framework.

Provides persistent storage using SQLite database with async operations.
Default storage backend - requires no external dependencies beyond Python stdlib.
"""

import aiosqlite
import pickle
import json
from typing import Any, List, Optional
from pathlib import Path
from datetime import datetime

from convergence.storage.base import (
    StorageProtocol,
    StorageError,
    StorageConnectionError,
    StorageSerializationError,
)


class SQLiteStorage:
    """
    SQLite-based storage implementation.
    
    Features:
    - Async operations via aiosqlite
    - Automatic serialization (pickle with JSON fallback)
    - Connection pooling via context manager
    - Automatic table creation
    - Timestamp tracking
    
    Usage:
        async with SQLiteStorage("./data/civilization.db") as storage:
            await storage.save("agent:1", agent_data)
            data = await storage.load("agent:1")
    
    Or with StorageRegistry:
        storage = StorageRegistry.get("sqlite", db_path="./data/civilization.db")
    """
    
    def __init__(
        self,
        db_path: str | Path,
        table_name: str = "storage",
        serializer: str = "pickle"
    ):
        """
        Initialize SQLite storage.
        
        Args:
            db_path: Path to SQLite database file
            table_name: Name of the storage table (default: "storage")
            serializer: Serialization method ("pickle" or "json")
        """
        self.db_path = Path(db_path)
        self.table_name = table_name
        self.serializer = serializer
        self._conn: Optional[aiosqlite.Connection] = None
        
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    async def __aenter__(self):
        """Async context manager entry - establishes connection."""
        await self._connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - closes connection."""
        await self.close()
    
    async def _connect(self) -> None:
        """
        Establish database connection and create table if needed.
        
        Raises:
            StorageConnectionError: If connection fails
        """
        if self._conn is not None:
            return  # Already connected
        
        try:
            self._conn = await aiosqlite.connect(self.db_path)
            
            # Create table if not exists
            await self._conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    key TEXT PRIMARY KEY,
                    value BLOB NOT NULL,
                    serializer TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index on key for faster lookups
            await self._conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_key
                ON {self.table_name}(key)
            """)
            
            await self._conn.commit()
            
        except Exception as e:
            raise StorageConnectionError(
                f"Failed to connect to SQLite database at {self.db_path}: {e}"
            ) from e
    
    async def _ensure_connected(self) -> None:
        """Ensure connection is established."""
        if self._conn is None:
            await self._connect()
    
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
                return json.dumps(value).encode("utf-8")
            else:
                raise ValueError(f"Unknown serializer: {self.serializer}")
        except Exception as e:
            raise StorageSerializationError(
                f"Failed to serialize value: {e}"
            ) from e
    
    def _deserialize(self, data: bytes, serializer: str) -> Any:
        """
        Deserialize bytes to Python object.
        
        Args:
            data: Serialized bytes
            serializer: Serialization method used
            
        Returns:
            Deserialized Python object
            
        Raises:
            StorageSerializationError: If deserialization fails
        """
        try:
            if serializer == "pickle":
                return pickle.loads(data)
            elif serializer == "json":
                return json.loads(data.decode("utf-8"))
            else:
                raise ValueError(f"Unknown serializer: {serializer}")
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
        await self._ensure_connected()
        
        try:
            serialized = self._serialize(value)
            
            await self._conn.execute(f"""
                INSERT OR REPLACE INTO {self.table_name}
                (key, value, serializer, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (key, serialized, self.serializer))
            
            await self._conn.commit()
            
        except StorageSerializationError:
            raise  # Re-raise serialization errors as-is
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
        await self._ensure_connected()
        
        try:
            async with self._conn.execute(f"""
                SELECT value, serializer FROM {self.table_name}
                WHERE key = ?
            """, (key,)) as cursor:
                row = await cursor.fetchone()
                
                if row is None:
                    raise KeyError(f"Key not found: {key}")
                
                value_bytes, serializer = row
                return self._deserialize(value_bytes, serializer)
                
        except KeyError:
            raise  # Re-raise KeyError as-is
        except StorageSerializationError:
            raise  # Re-raise deserialization errors as-is
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
        await self._ensure_connected()
        
        try:
            async with self._conn.execute(f"""
                SELECT 1 FROM {self.table_name}
                WHERE key = ?
                LIMIT 1
            """, (key,)) as cursor:
                row = await cursor.fetchone()
                return row is not None
                
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
        await self._ensure_connected()
        
        try:
            await self._conn.execute(f"""
                DELETE FROM {self.table_name}
                WHERE key = ?
            """, (key,))
            
            await self._conn.commit()
            
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
        """
        await self._ensure_connected()
        
        try:
            if prefix:
                # Use LIKE for prefix matching
                async with self._conn.execute(f"""
                    SELECT key FROM {self.table_name}
                    WHERE key LIKE ?
                    ORDER BY key
                """, (f"{prefix}%",)) as cursor:
                    rows = await cursor.fetchall()
            else:
                # Get all keys
                async with self._conn.execute(f"""
                    SELECT key FROM {self.table_name}
                    ORDER BY key
                """) as cursor:
                    rows = await cursor.fetchall()
            
            return [row[0] for row in rows]
            
        except Exception as e:
            raise StorageError(
                f"Failed to list keys with prefix '{prefix}': {e}"
            ) from e
    
    async def close(self) -> None:
        """
        Close the database connection.
        
        Idempotent - safe to call multiple times.
        """
        if self._conn is not None:
            try:
                await self._conn.close()
            except Exception:
                pass  # Ignore errors during close
            finally:
                self._conn = None
    
    # Additional utility methods
    
    async def count_keys(self, prefix: str = "") -> int:
        """
        Count keys, optionally filtered by prefix.
        
        Args:
            prefix: Optional prefix to filter keys
            
        Returns:
            Number of matching keys
        """
        await self._ensure_connected()
        
        try:
            if prefix:
                async with self._conn.execute(f"""
                    SELECT COUNT(*) FROM {self.table_name}
                    WHERE key LIKE ?
                """, (f"{prefix}%",)) as cursor:
                    row = await cursor.fetchone()
            else:
                async with self._conn.execute(f"""
                    SELECT COUNT(*) FROM {self.table_name}
                """) as cursor:
                    row = await cursor.fetchone()
            
            return row[0] if row else 0
            
        except Exception as e:
            raise StorageError(
                f"Failed to count keys with prefix '{prefix}': {e}"
            ) from e
    
    async def clear(self, prefix: str = "") -> int:
        """
        Delete all keys, optionally filtered by prefix.
        
        Args:
            prefix: Optional prefix to filter keys
            
        Returns:
            Number of keys deleted
        """
        await self._ensure_connected()
        
        try:
            if prefix:
                async with self._conn.execute(f"""
                    DELETE FROM {self.table_name}
                    WHERE key LIKE ?
                """, (f"{prefix}%",)) as cursor:
                    deleted = cursor.rowcount
            else:
                async with self._conn.execute(f"""
                    DELETE FROM {self.table_name}
                """) as cursor:
                    deleted = cursor.rowcount
            
            await self._conn.commit()
            return deleted if deleted is not None else 0
            
        except Exception as e:
            raise StorageError(
                f"Failed to clear keys with prefix '{prefix}': {e}"
            ) from e


# Verify protocol implementation at module level
# This ensures SQLiteStorage implements StorageProtocol correctly
def _verify_protocol() -> None:
    """Verify that SQLiteStorage implements StorageProtocol."""
    from convergence.storage.base import StorageProtocol
    
    # This will raise TypeError at import time if protocol is not implemented
    storage = SQLiteStorage(":memory:")
    assert isinstance(storage, StorageProtocol), \
        "SQLiteStorage must implement StorageProtocol"


_verify_protocol()

