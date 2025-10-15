"""
Configuration loader for optimization.yaml files.

This module provides functionality to load, validate, and parse
optimization configuration files.
"""
import yaml
import json
from pathlib import Path
from typing import Union, Dict, Any
import os

from .models import OptimizationSchema
from convergence.core.env_loader import get_api_key


class ConfigLoader:
    """
    Loads and validates optimization configuration files.
    
    Supports YAML and JSON formats. Validates against OptimizationSchema
    and resolves environment variables for auth tokens.
    """
    
    @staticmethod
    def load(path: Union[str, Path]) -> OptimizationSchema:
        """
        Load and validate an optimization configuration file.
        
        Args:
            path: Path to YAML or JSON config file
            
        Returns:
            Validated OptimizationSchema instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        # Load file based on extension
        if path.suffix in ['.yaml', '.yml']:
            config_dict = ConfigLoader._load_yaml(path)
        elif path.suffix == '.json':
            config_dict = ConfigLoader._load_json(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}. Use .yaml, .yml, or .json")
        
        # Resolve environment variables
        ConfigLoader._resolve_env_vars(config_dict)
        
        # Validate with Pydantic
        try:
            schema = OptimizationSchema(**config_dict)
        except Exception as e:
            raise ValueError(f"Invalid configuration: {e}") from e
        
        return schema
    
    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        """Load YAML file."""
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        """Load JSON file."""
        with open(path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def _resolve_env_vars(config: Dict[str, Any]) -> None:
        """
        Resolve environment variables in config.
        
        Replaces token_env references with actual environment variable values.
        Modifies config in-place.
        """
        # Resolve auth token
        if 'api' in config and 'auth' in config['api']:
            auth = config['api']['auth']
            
            # Bearer token
            if auth.get('type') == 'bearer' and 'token_env' in auth:
                env_var = auth['token_env']
                token = os.getenv(env_var)
                if not token:
                    # Try to get from automatic env loading
                    provider = env_var.replace('_API_KEY', '').replace('_KEY', '').lower()
                    token = get_api_key(provider)
                if not token:
                    raise ValueError(f"Environment variable {env_var} not set for bearer token")
                auth['token'] = token
            
            # API key
            elif auth.get('type') == 'api_key' and 'token_env' in auth:
                env_var = auth['token_env']
                api_key = os.getenv(env_var)
                if not api_key:
                    # Try to get from automatic env loading
                    provider = env_var.replace('_API_KEY', '').replace('_KEY', '').lower()
                    api_key = get_api_key(provider)
                if not api_key:
                    raise ValueError(f"Environment variable {env_var} not set for API key")
                auth['api_key'] = api_key
            
            # Basic auth password
            elif auth.get('type') == 'basic' and 'password_env' in auth:
                env_var = auth['password_env']
                password = os.getenv(env_var)
                if not password:
                    raise ValueError(f"Environment variable {env_var} not set for basic auth password")
                auth['password'] = password
    
    @staticmethod
    def validate_file(path: Union[str, Path]) -> bool:
        """
        Validate a configuration file without loading it fully.
        
        Args:
            path: Path to config file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            ConfigLoader.load(path)
            return True
        except Exception:
            return False

