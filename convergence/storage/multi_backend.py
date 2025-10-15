"""
Multi-Backend Storage Manager for The Convergence.

CRITICAL INFRASTRUCTURE: This ensures data is NEVER lost.

Default Strategy:
- SQLite: Fast queries, relations, RL training data
- File: Human-readable backup, version control, audit trail
- Memory: Hot cache for active data (optional)

Why Multi-Backend?
- Redundancy: If one fails, we have another
- Legacy: File storage creates permanent audit trail
- Performance: Memory cache for hot data
- RL Training: SQLite optimized for trajectory queries
- Human Inspection: File storage for debugging/auditing

Usage:
    storage = MultiBackendStorage()
    await storage.save("agent:1", data)
    # Data is now in BOTH SQLite AND File storage
    # If either fails, the other succeeds
    # Nothing is lost
"""

from __future__ import annotations
from typing import Any, List, Dict, Optional, Set
import asyncio
from datetime import datetime
from pathlib import Path

from convergence.storage.base import StorageProtocol, StorageError
from convergence.storage.sqlite import SQLiteStorage
from convergence.storage.file import FileStorage
from convergence.storage.memory import MemoryStorage


class MultiBackendStorage:
    """
    Write to multiple storage backends simultaneously.
    
    Features:
    - Parallel writes to all backends (fast)
    - Read from fastest backend first (cache hierarchy)
    - Automatic failover if one backend fails
    - Guaranteed data persistence (redundancy)
    - Optimized for RL training data patterns
    
    Default Configuration:
    - Primary: SQLite (for queries and RL training)
    - Backup: File storage (for audit trail and human inspection)
    - Cache: Memory (optional, for hot data)
    
    Example:
        storage = MultiBackendStorage(
            backends=["sqlite", "file"],  # Dual write
            cache_enabled=True  # Memory cache for reads
        )
        
        await storage.save("episode:1", episode_data)
        # Written to BOTH SQLite and File
        
        data = await storage.load("episode:1")
        # Read from memory cache (if enabled) or SQLite (fastest)
    """
    
    def __init__(
        self,
        backends: Optional[List[str]] = None,
        cache_enabled: bool = True,
        sqlite_path: str = "./data/convergence_legacy.db",
        file_base_dir: str = "./data/convergence_legacy",
        cache_ttl_seconds: Optional[int] = 300,  # 5 min default
        cache_max_size: Optional[int] = 10000,
    ):
        """
        Initialize multi-backend storage.
        
        Args:
            backends: List of backends to use (default: ["sqlite", "file"])
            cache_enabled: Enable memory cache for reads
            sqlite_path: Path to SQLite database
            file_base_dir: Base directory for file storage
            cache_ttl_seconds: Cache TTL (None = no expiration)
            cache_max_size: Max cache entries (None = unlimited)
        """
        # Default to dual-write (SQLite + File)
        self.backends: List[StorageProtocol] = []
        self.backend_names = backends or ["sqlite", "file"]
        
        # Initialize backends
        for backend_name in self.backend_names:
            if backend_name == "sqlite":
                self.backends.append(SQLiteStorage(db_path=sqlite_path))
            elif backend_name == "file":
                self.backends.append(FileStorage(base_dir=file_base_dir))
            else:
                raise ValueError(f"Unknown backend: {backend_name}")
        
        # Optional memory cache for hot data
        self.cache_enabled = cache_enabled
        self.cache: Optional[MemoryStorage] = None
        if cache_enabled:
            self.cache = MemoryStorage(
                ttl_seconds=cache_ttl_seconds,
                max_size=cache_max_size
            )
        
        # Track write failures for monitoring
        self.write_failures: Dict[str, int] = {name: 0 for name in self.backend_names}
        
        # Track which backend is fastest for reads
        self.read_performance: Dict[str, List[float]] = {name: [] for name in self.backend_names}
        
        print(f"🔐 Multi-Backend Storage initialized:")
        print(f"   Backends: {', '.join(self.backend_names)}")
        print(f"   Cache: {'Enabled' if cache_enabled else 'Disabled'}")
        print(f"   Strategy: Dual-write for redundancy, fastest-first for reads")
    
    async def save(self, key: str, value: Any) -> Dict[str, bool]:
        """
        Save to ALL backends in parallel.
        
        This is the key method that ensures data is never lost.
        Even if one backend fails, others succeed.
        
        Args:
            key: Storage key
            value: Value to store
            
        Returns:
            Dict mapping backend name to success status
            
        Raises:
            StorageError: Only if ALL backends fail
        """
        # Write to cache immediately (synchronous, fast)
        if self.cache:
            await self.cache.save(key, value)
        
        # Write to all persistent backends in parallel
        tasks = []
        for backend, name in zip(self.backends, self.backend_names):
            tasks.append(self._safe_write(backend, name, key, value))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        success_map = {}
        all_failed = True
        
        for name, result in zip(self.backend_names, results):
            if isinstance(result, Exception):
                success_map[name] = False
                self.write_failures[name] += 1
                print(f"⚠️ Write to {name} failed: {result}")
            else:
                success_map[name] = True
                all_failed = False
        
        # If ALL backends failed, that's a critical error
        if all_failed:
            raise StorageError(
                f"All backends failed for key '{key}': {results}"
            )
        
        return success_map
    
    async def _safe_write(
        self,
        backend: StorageProtocol,
        name: str,
        key: str,
        value: Any
    ) -> None:
        """Safely write to a backend with error handling."""
        try:
            await backend.save(key, value)
        except Exception as e:
            # Log but don't raise - we want other backends to succeed
            print(f"⚠️ Backend {name} write failed: {e}")
            raise  # Re-raise so gather() catches it
    
    async def load(self, key: str) -> Any:
        """
        Load from fastest available backend.
        
        Read hierarchy:
        1. Memory cache (if enabled) - microseconds
        2. SQLite - milliseconds
        3. File storage - milliseconds
        
        Args:
            key: Storage key
            
        Returns:
            Stored value
            
        Raises:
            KeyError: If key not found in any backend
        """
        # Try cache first (fastest)
        if self.cache:
            try:
                return await self.cache.load(key)
            except KeyError:
                pass  # Not in cache, try persistent storage
        
        # Try backends in order (SQLite should be fastest)
        last_error = None
        for backend, name in zip(self.backends, self.backend_names):
            try:
                import time
                start = time.time()
                value = await backend.load(key)
                duration = time.time() - start
                
                # Track read performance
                self.read_performance[name].append(duration)
                
                # Warm cache for future reads
                if self.cache:
                    await self.cache.save(key, value)
                
                return value
                
            except KeyError:
                last_error = KeyError(f"Key not found: {key}")
                continue
            except Exception as e:
                print(f"⚠️ Backend {name} read failed: {e}")
                last_error = e
                continue
        
        # All backends failed
        raise last_error or KeyError(f"Key not found: {key}")
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in any backend."""
        # Check cache first
        if self.cache and await self.cache.exists(key):
            return True
        
        # Check any backend
        for backend in self.backends:
            try:
                if await backend.exists(key):
                    return True
            except Exception:
                continue
        
        return False
    
    async def delete(self, key: str) -> Dict[str, bool]:
        """Delete from all backends."""
        # Delete from cache
        if self.cache:
            await self.cache.delete(key)
        
        # Delete from all backends in parallel
        tasks = []
        for backend in self.backends:
            tasks.append(backend.delete(key))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            name: not isinstance(result, Exception)
            for name, result in zip(self.backend_names, results)
        }
    
    async def list_keys(self, prefix: str = "") -> List[str]:
        """
        List keys from primary backend (SQLite).
        
        SQLite is most efficient for listing/querying.
        """
        # Use first backend (should be SQLite)
        return await self.backends[0].list_keys(prefix)
    
    async def count_keys(self, prefix: str = "") -> int:
        """Count keys from primary backend."""
        return await self.backends[0].count_keys(prefix)
    
    async def clear(self, prefix: str = "") -> Dict[str, int]:
        """Clear keys from all backends."""
        # Clear cache
        cache_count = 0
        if self.cache:
            cache_count = await self.cache.clear(prefix)
        
        # Clear all backends in parallel
        tasks = []
        for backend in self.backends:
            tasks.append(backend.clear(prefix))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        clear_map = {"cache": cache_count}
        for name, result in zip(self.backend_names, results):
            if isinstance(result, Exception):
                clear_map[name] = 0
            else:
                clear_map[name] = result
        
        return clear_map
    
    async def close(self) -> None:
        """Close all backends."""
        if self.cache:
            await self.cache.close()
        
        for backend in self.backends:
            await backend.close()
    
    def get_health_stats(self) -> Dict[str, Any]:
        """
        Get health statistics for all backends.
        
        Returns:
            Dict with health metrics
        """
        stats = {
            "backends": self.backend_names,
            "cache_enabled": self.cache_enabled,
            "write_failures": self.write_failures,
            "read_performance": {}
        }
        
        # Calculate average read performance
        for name, durations in self.read_performance.items():
            if durations:
                avg = sum(durations) / len(durations)
                stats["read_performance"][name] = {
                    "avg_ms": avg * 1000,
                    "reads": len(durations)
                }
        
        # Cache stats
        if self.cache:
            stats["cache"] = self.cache.get_storage_stats()
        
        return stats
    
    async def verify_integrity(self) -> Dict[str, Any]:
        """
        Verify data integrity across backends.
        
        Checks that all backends have consistent data.
        Useful for auditing and debugging.
        
        Returns:
            Integrity report
        """
        print("🔍 Verifying storage integrity...")
        
        # Get keys from all backends
        backend_keys: Dict[str, Set[str]] = {}
        for backend, name in zip(self.backends, self.backend_names):
            try:
                keys = await backend.list_keys()
                backend_keys[name] = set(keys)
            except Exception as e:
                print(f"⚠️ Failed to list keys from {name}: {e}")
                backend_keys[name] = set()
        
        # Find inconsistencies
        all_keys = set().union(*backend_keys.values())
        missing_keys = {}
        
        for name, keys in backend_keys.items():
            missing = all_keys - keys
            if missing:
                missing_keys[name] = list(missing)
        
        # Calculate consistency
        if len(backend_keys) > 1:
            intersection = set.intersection(*backend_keys.values())
            consistency = len(intersection) / len(all_keys) if all_keys else 1.0
        else:
            consistency = 1.0
        
        report = {
            "total_keys": len(all_keys),
            "consistency": consistency,
            "backends": {
                name: {
                    "key_count": len(keys),
                    "missing_keys": len(missing_keys.get(name, []))
                }
                for name, keys in backend_keys.items()
            },
            "missing_keys_detail": missing_keys if missing_keys else None
        }
        
        print(f"✅ Integrity check complete:")
        print(f"   Total keys: {len(all_keys)}")
        print(f"   Consistency: {consistency * 100:.1f}%")
        
        return report


# Convenience function for getting default multi-backend storage
def get_legacy_storage(
    cache_enabled: bool = True,
    data_dir: str = "./data"
) -> MultiBackendStorage:
    """
    Get default legacy storage with recommended configuration.
    
    Args:
        cache_enabled: Enable memory cache (True for dev, False for production)
        data_dir: Base directory for data storage
        
    Returns:
        Configured multi-backend storage
        
    Example:
        # Development (with cache)
        storage = get_legacy_storage(cache_enabled=True)
        
        # Production (no cache, use Redis instead)
        storage = get_legacy_storage(cache_enabled=False)
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    return MultiBackendStorage(
        backends=["sqlite", "file"],
        cache_enabled=cache_enabled,
        sqlite_path=str(data_path / "convergence_legacy.db"),
        file_base_dir=str(data_path / "convergence_legacy"),
        cache_ttl_seconds=300 if cache_enabled else None,
        cache_max_size=10000
    )

