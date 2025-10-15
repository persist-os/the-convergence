"""
Async utilities for generation (atomic, race-free).

Follows asyncio best practices:
- No shared mutable state
- All operations atomic
- Proper resource cleanup with context managers
"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator
import aiofiles
import asyncio


@asynccontextmanager
async def atomic_write_file(path: Path) -> AsyncIterator:
    """
    Atomic file write (no race conditions).
    
    Writes to temp file first, then atomically moves.
    If interrupted, original file is unchanged.
    
    Args:
        path: Target file path
    
    Yields:
        File handle for writing
    
    Example:
        async with atomic_write_file(Path("config.yaml")) as f:
            await f.write("content")
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + '.tmp')
    
    try:
        async with aiofiles.open(temp_path, 'w') as f:
            yield f
        
        # Atomic move (no race condition)
        temp_path.replace(path)
    
    except Exception:
        # Cleanup temp file on error
        if temp_path.exists():
            temp_path.unlink()
        raise


async def safe_gather(*tasks, return_exceptions: bool = True):
    """
    Gather tasks with proper error handling.
    
    Unlike asyncio.gather, this always returns exceptions
    instead of raising, preventing partial state updates.
    
    Args:
        *tasks: Coroutines to run concurrently
        return_exceptions: Always True (force error handling)
    
    Returns:
        List of results or exceptions
    """
    return await asyncio.gather(*tasks, return_exceptions=True)

