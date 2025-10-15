"""
Automatic environment variable loading for The Convergence.

This module provides automatic loading of .env files without requiring
manual export commands. It integrates with python-dotenv to load
environment variables from .env files when needed.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import logging

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

logger = logging.getLogger(__name__)

class EnvironmentLoader:
    """Automatically loads environment variables from .env files."""
    
    def __init__(self, env_file: Optional[str] = None, override: bool = False):
        """
        Initialize the environment loader.
        
        Args:
            env_file: Path to .env file. If None, searches for .env in current and parent directories.
            override: Whether to override existing environment variables.
        """
        self.env_file = env_file
        self.override = override
        self._loaded = False
        
    def load_env(self, env_file: Optional[str] = None) -> Dict[str, str]:
        """
        Load environment variables from .env file.
        
        Args:
            env_file: Override the env file path for this call.
            
        Returns:
            Dictionary of loaded environment variables.
        """
        if not DOTENV_AVAILABLE:
            logger.warning("python-dotenv not available. Install with: pip install python-dotenv")
            return {}
            
        target_file = env_file or self.env_file
        
        # If no specific file provided, search for .env files
        if target_file is None:
            target_file = self._find_env_file()
            
        if target_file is None:
            logger.debug("No .env file found")
            return {}
            
        # Load the .env file
        try:
            # Load into environment
            load_dotenv(target_file, override=self.override)
            
            # Also return the variables for inspection
            env_vars = {}
            with open(target_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        # Remove quotes if present
                        value = value.strip('"\'')
                        env_vars[key] = value
                        
            self._loaded = True
            logger.info(f"✅ Loaded environment variables from {target_file}")
            return env_vars
            
        except Exception as e:
            logger.error(f"Failed to load .env file {target_file}: {e}")
            return {}
    
    def _find_env_file(self) -> Optional[str]:
        """Find .env file in current directory or parent directories."""
        current_path = Path.cwd()
        
        # Check current directory first
        env_file = current_path / '.env'
        if env_file.exists():
            return str(env_file)
            
        # Check parent directories (up to 3 levels)
        for i in range(3):
            current_path = current_path.parent
            env_file = current_path / '.env'
            if env_file.exists():
                return str(env_file)
                
        return None
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Get API key for a specific provider.
        
        Args:
            provider: Provider name (e.g., 'gemini', 'openai', 'browserbase')
            
        Returns:
            API key if found, None otherwise.
        """
        if not self._loaded:
            self.load_env()
            
        # Try multiple possible environment variable names
        possible_keys = [
            f"{provider.upper()}_API_KEY",
            f"{provider.upper()}_KEY", 
            f"{provider}_api_key",
            f"{provider}_key"
        ]
        
        # Special cases for common providers
        if provider.lower() == 'gemini':
            possible_keys.extend(['GOOGLE_API_KEY', 'google_api_key'])
        elif provider.lower() == 'openai':
            possible_keys.extend(['OPENAI_API_KEY'])
        elif provider.lower() == 'browserbase':
            possible_keys.extend(['BROWSERBASE_API_KEY'])
            
        for key in possible_keys:
            value = os.getenv(key)
            if value:
                return value
                
        return None
    
    def ensure_api_key(self, provider: str) -> str:
        """
        Ensure API key is available, loading .env if needed.
        
        Args:
            provider: Provider name
            
        Returns:
            API key
            
        Raises:
            ValueError: If API key not found after loading .env
        """
        api_key = self.get_api_key(provider)
        if not api_key:
            raise ValueError(
                f"API key for {provider} not found. "
                f"Please set {provider.upper()}_API_KEY in your .env file or environment."
            )
        return api_key

# Global instance for easy access
_env_loader = EnvironmentLoader()

def load_env_automatically(env_file: Optional[str] = None, override: bool = False) -> Dict[str, str]:
    """
    Convenience function to automatically load .env file.
    
    Args:
        env_file: Path to .env file
        override: Whether to override existing environment variables
        
    Returns:
        Dictionary of loaded environment variables
    """
    return _env_loader.load_env(env_file)

def get_api_key(provider: str) -> Optional[str]:
    """
    Get API key for provider, loading .env automatically if needed.
    
    Args:
        provider: Provider name
        
    Returns:
        API key if found
    """
    return _env_loader.get_api_key(provider)

def ensure_api_key(provider: str) -> str:
    """
    Ensure API key is available, loading .env automatically if needed.
    
    Args:
        provider: Provider name
        
    Returns:
        API key
        
    Raises:
        ValueError: If API key not found
    """
    return _env_loader.ensure_api_key(provider)

# Auto-load on import if .env file exists
if DOTENV_AVAILABLE:
    _env_loader.load_env()

__all__ = [
    'EnvironmentLoader',
    'load_env_automatically', 
    'get_api_key',
    'ensure_api_key'
]
