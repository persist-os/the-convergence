"""
Storage registry for managing storage backends.

Provides centralized registration and instantiation of storage backends.
"""

from typing import Dict, Type, Any, Optional
from convergence.storage.base import StorageProtocol, StorageError


class StorageRegistry:
    """
    Registry for storage backends.
    
    Allows registration of custom storage implementations and
    instantiation by name.
    
    Usage:
        # Register a custom storage
        StorageRegistry.register("redis", RedisStorage)
        
        # Get an instance
        storage = StorageRegistry.get("redis", host="localhost", port=6379)
    """
    
    _backends: Dict[str, Type] = {}
    
    @classmethod
    def register(cls, name: str, storage_class: Type) -> None:
        """
        Register a storage backend.
        
        Args:
            name: Backend name (e.g., "redis", "mongo", "s3")
            storage_class: Class implementing StorageProtocol
            
        Raises:
            TypeError: If storage_class is not a class
            
        Example:
            from my_storage import RedisStorage
            StorageRegistry.register("redis", RedisStorage)
        """
        if not isinstance(storage_class, type):
            raise TypeError(
                f"storage_class must be a class, got {type(storage_class)}"
            )
        
        cls._backends[name] = storage_class
        print(f"✅ Registered storage backend: {name}")
    
    @classmethod
    def get(
        cls,
        name: str,
        **config: Any
    ) -> StorageProtocol:
        """
        Get a storage instance by name.
        
        Args:
            name: Backend name
            **config: Configuration passed to storage constructor
            
        Returns:
            Storage instance implementing StorageProtocol
            
        Raises:
            ValueError: If backend not registered
            
        Example:
            storage = StorageRegistry.get(
                "sqlite",
                db_path="./data/civilization.db"
            )
        """
        if name not in cls._backends:
            available = ", ".join(cls._backends.keys())
            raise ValueError(
                f"Storage backend '{name}' not registered. "
                f"Available backends: {available or 'none'}"
            )
        
        storage_class = cls._backends[name]
        try:
            instance = storage_class(**config)
            return instance  # type: ignore
        except Exception as e:
            raise StorageError(
                f"Failed to instantiate storage backend '{name}': {e}"
            ) from e
    
    @classmethod
    def list_backends(cls) -> list[str]:
        """
        List all registered storage backends.
        
        Returns:
            List of backend names
        """
        return list(cls._backends.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a backend is registered.
        
        Args:
            name: Backend name
            
        Returns:
            True if registered, False otherwise
        """
        return name in cls._backends
    
    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Unregister a storage backend.
        
        Args:
            name: Backend name
            
        Note:
            Used primarily for testing. Be careful with this in production.
        """
        if name in cls._backends:
            del cls._backends[name]
            print(f"❌ Unregistered storage backend: {name}")


# ============================================================================
# AUTO-REGISTRATION OF BUILT-IN BACKENDS
# ============================================================================

def _register_builtin_backends() -> None:
    """
    Auto-register built-in storage backends.
    
    This is called when the module is imported.
    Backends are only registered if their dependencies are available.
    """
    
    # Try to register SQLite (should always be available)
    try:
        from convergence.storage.sqlite import SQLiteStorage
        StorageRegistry.register("sqlite", SQLiteStorage)
    except ImportError:
        pass
    
    # Try to register file storage (should always be available)
    try:
        from convergence.storage.file import FileStorage
        StorageRegistry.register("file", FileStorage)
    except ImportError:
        pass
    
    # Try to register memory storage (should always be available)
    try:
        from convergence.storage.memory import MemoryStorage
        StorageRegistry.register("memory", MemoryStorage)
    except ImportError:
        pass
    
    # Try to register multi-backend storage (should always be available)
    try:
        from convergence.storage.multi_backend import MultiBackendStorage
        StorageRegistry.register("multi", MultiBackendStorage)
    except ImportError:
        pass
    
    # Try to register Postgres (optional dependency)
    try:
        from convergence.storage.postgres import PostgresStorage
        StorageRegistry.register("postgres", PostgresStorage)
    except ImportError:
        pass  # Postgres is optional
    
    # Try to register Convex (requires backend integration)
    try:
        from convergence.storage.convex import ConvexStorage
        StorageRegistry.register("convex", ConvexStorage)
    except ImportError:
        pass  # Convex is optional


# Register built-in backends on module import
_register_builtin_backends()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_storage_registry() -> Type[StorageRegistry]:
    """
    Get the StorageRegistry class.
    
    Returns:
        StorageRegistry class (not an instance)
        
    Usage:
        registry = get_storage_registry()
        storage = registry.get("sqlite", db_path="./data/test.db")
    """
    return StorageRegistry


def reset_storage_registry() -> None:
    """
    Reset the storage registry (clear all registrations).
    
    This is useful for testing to ensure a clean state.
    After reset, you must re-register backends manually
    or call _register_builtin_backends().
    
    Usage:
        reset_storage_registry()
        # Now registry is empty
    """
    StorageRegistry._backends.clear()
    print("🔄 Storage registry reset")

