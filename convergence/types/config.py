"""
Configuration models for Convergence SDK.
"""

from typing import Dict, Any, Optional, Union, Callable
from pydantic import BaseModel, Field


class ApiConfig(BaseModel):
    """API endpoint configuration."""
    name: str
    kind: str = "callable"  # "http" or "callable"
    endpoint: Optional[str] = None
    request_timeout: Optional[float] = 30.0
    auth_type: Optional[str] = None  # "api_key", "bearer", etc.
    auth_token_env: Optional[str] = None  # Environment variable name for auth token
    auth_header_name: Optional[str] = None  # Header name for API key type


class SearchSpaceConfig(BaseModel):
    """Search space configuration."""
    parameters: Dict[str, Dict[str, Any]]


class RunnerConfig(BaseModel):
    """Runner configuration."""
    generations: int = 10
    population: int = 20
    seed: Optional[int] = None
    early_stopping: Optional[Dict[str, Any]] = None


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""
    required_metrics: list[str]
    weights: Dict[str, float]
    thresholds: Optional[Dict[str, float]] = None


class StorageConfig(BaseModel):
    """Storage configuration (optional)."""
    provider: Optional[str] = None  # "local", "s3", "gcs"
    path: Optional[str] = None


class AdaptersConfig(BaseModel):
    """Adapters configuration for data transformation."""
    input_adapter: Optional[Union[Callable, str]] = None
    output_adapter: Optional[Union[Callable, str]] = None
    case_adapter: Optional[Union[Callable, str]] = None
    batch_adapter: Optional[Union[Callable, str]] = None


class AgentConfig(BaseModel):
    """Agent configuration for Agno-based optimizations."""
    models: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    discord_auth: Optional[Dict[str, Any]] = None


class ConvergenceConfig(BaseModel):
    """Top-level Convergence configuration."""
    api: ApiConfig
    search_space: SearchSpaceConfig
    runner: RunnerConfig
    evaluation: EvaluationConfig
    storage: Optional[StorageConfig] = None
    adapters: Optional[AdaptersConfig] = None
    agent: Optional[AgentConfig] = None

