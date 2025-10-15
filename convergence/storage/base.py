"""
Base storage protocol and configuration for The Convergence framework.

Defines the StorageProtocol interface that all storage backends must implement.
Uses Python's Protocol for structural subtyping (PEP 544).
"""

from typing import Protocol, Any, List, Optional, runtime_checkable
from pydantic import BaseModel, Field
from pathlib import Path


# ============================================================================
# STORAGE CONFIGURATION
# ============================================================================

class StorageConfig(BaseModel):
    """
    Configuration for storage backends.
    
    This is a base config that can be extended by specific storage implementations.
    """
    
    backend: str = Field(
        default="sqlite",
        description="Storage backend to use (sqlite, file, memory, postgres, or custom)"
    )
    
    # Common configuration options
    base_path: Optional[Path] = Field(
        default=None,
        description="Base path for file-based storage"
    )
    
    # Additional config as key-value pairs
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional backend-specific configuration"
    )
    
    class Config:
        arbitrary_types_allowed = True


# ============================================================================
# STORAGE PROTOCOL
# ============================================================================

@runtime_checkable
class StorageProtocol(Protocol):
    """
    Protocol for storage backends in The Convergence framework.
    
    Any class implementing these methods can be used as storage.
    No inheritance required - this is structural subtyping (PEP 544).
    
    Design Philosophy:
    - Simple async interface
    - Key-value semantics
    - Serialization handled by storage
    - Resource management via async context manager
    
    Example Implementation:
    ```python
    class MyStorage:
        async def save(self, key: str, value: Any) -> None:
            # Store the value
            pass
        
        async def load(self, key: str) -> Any:
            # Retrieve the value
            pass
        
        # ... implement other methods
    
    # Register and use
    StorageRegistry.register("mystorage", MyStorage)
    storage = StorageRegistry.get("mystorage", **config)
    ```
    """
    
    async def save(self, key: str, value: Any) -> None:
        """
        Save a value with a key.
        
        The storage backend is responsible for serialization.
        Values can be any Python object (must be serializable).
        
        Args:
            key: Unique identifier for the value
            value: Any Python object to store
            
        Raises:
            StorageError: If save operation fails
            
        Note:
            If key exists, it should be overwritten.
        """
        ...
    
    async def load(self, key: str) -> Any:
        """
        Load a value by key.
        
        Args:
            key: Unique identifier for the value
            
        Returns:
            The stored object (deserialized)
            
        Raises:
            KeyError: If key doesn't exist
            StorageError: If load operation fails
        """
        ...
    
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in storage.
        
        Args:
            key: Unique identifier to check
            
        Returns:
            True if key exists, False otherwise
            
        Note:
            Should not raise KeyError - returns False instead.
        """
        ...
    
    async def delete(self, key: str) -> None:
        """
        Delete a value by key.
        
        Args:
            key: Unique identifier to delete
            
        Raises:
            KeyError: If key doesn't exist (optional - can be silent)
            StorageError: If delete operation fails
            
        Note:
            Some implementations may silently ignore missing keys.
        """
        ...
    
    async def list_keys(self, prefix: str = "") -> List[str]:
        """
        List all keys, optionally filtered by prefix.
        
        Args:
            prefix: Optional prefix to filter keys
                   Empty string returns all keys
            
        Returns:
            List of matching keys
            
        Example:
            await storage.list_keys("user:")  # Returns ["user:1", "user:2", ...]
            await storage.list_keys()         # Returns all keys
        """
        ...
    
    async def close(self) -> None:
        """
        Clean up resources (connections, file handles, etc.).
        
        Should be idempotent - safe to call multiple times.
        
        This is typically called in the __aexit__ method when used
        as an async context manager.
        
        Example:
            async with storage:
                await storage.save("key", "value")
            # close() called automatically
        """
        ...


# ============================================================================
# STORAGE EXCEPTIONS
# ============================================================================

class StorageError(Exception):
    """Base exception for storage operations."""
    pass


class StorageConnectionError(StorageError):
    """Raised when storage connection fails."""
    pass


class StorageSerializationError(StorageError):
    """Raised when serialization/deserialization fails."""
    pass


class StorageCapacityError(StorageError):
    """Raised when storage is full or exceeds limits."""
    pass


class StorageNotFoundError(StorageError):
    """Raised when a key is not found in storage."""
    pass

