"""
Configuration management for The Convergence framework.

Uses Pydantic Settings for validation, env var loading, and type safety.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConvergenceConfig(BaseSettings):
    """
    Main configuration for The Convergence framework.
    
    Configuration sources (in order of precedence):
    1. Environment variables (CONVERGENCE_*)
    2. .env file
    3. Config file (JSON/YAML)
    4. Defaults in code
    """
    
    model_config = SettingsConfigDict(
        env_prefix="",  # Allow any env var (for API keys)
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra env vars
    )
    
    # ========================================================================
    # GENERAL SETTINGS
    # ========================================================================
    
    project_name: str = Field(
        default="the-convergence",
        description="Project name"
    )
    
    version: str = Field(
        default="0.1.0",
        description="Framework version"
    )
    
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    
    # ========================================================================
    # LLM PROVIDER SETTINGS
    # ========================================================================
    
    llm_provider: str = Field(
        default="litellm",
        description="Default LLM provider"
    )
    
    llm_model: str = Field(
        default="gpt-4",
        description="Default LLM model"
    )
    
    llm_api_key: Optional[str] = Field(
        default=None,
        description="API key for LLM provider"
    )
    
    llm_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Default temperature for LLM generation"
    )
    
    llm_max_tokens: int = Field(
        default=1000,
        ge=1,
        description="Default max tokens for LLM generation"
    )
    
    # API Keys (loaded from env vars directly, not prefixed)
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )
    
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key"
    )
    
    gemini_api_key: Optional[str] = Field(
        default=None,
        description="Google Gemini API key"
    )
    
    cohere_api_key: Optional[str] = Field(
        default=None,
        description="Cohere API key"
    )
    
    huggingface_api_key: Optional[str] = Field(
        default=None,
        description="Hugging Face API key"
    )
    
    # ========================================================================
    # LEARNING SETTINGS
    # ========================================================================
    
    # MAB Settings
    mab_algorithm: str = Field(
        default="thompson_sampling",
        description="Default MAB algorithm (thompson_sampling, ucb1, successive_elimination)"
    )
    
    mab_exploration_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Exploration rate for MAB"
    )
    
    # RLP Settings (NVIDIA research)
    rlp_enabled: bool = Field(
        default=True,
        description="Enable Reinforcement Learning Pretraining"
    )
    
    rlp_thought_depth: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Depth of reasoning chains in RLP"
    )
    
    rlp_reward_fn: str = Field(
        default="information_gain",
        description="Reward function for RLP (information_gain, accuracy)"
    )
    
    # SAO Settings (Hugging Face research)
    sao_enabled: bool = Field(
        default=True,
        description="Enable Self-Alignment Optimization"
    )
    
    sao_n_personas: int = Field(
        default=100,
        ge=1,
        description="Number of personas for synthetic data generation"
    )
    
    sao_synthetic_data_size: int = Field(
        default=10000,
        ge=100,
        description="Size of synthetic dataset"
    )
    
    # ========================================================================
    # MEMORY SETTINGS
    # ========================================================================
    
    # Procedural Memory (Memp research)
    procedural_memory_enabled: bool = Field(
        default=True,
        description="Enable procedural memory"
    )
    
    procedural_memory_distillation: List[str] = Field(
        default=["step_by_step", "script_level"],
        description="Distillation strategies for procedural memory"
    )
    
    # Semantic Memory (Mem0 research)
    semantic_memory_enabled: bool = Field(
        default=True,
        description="Enable semantic/graph memory"
    )
    
    semantic_memory_type: str = Field(
        default="graph",
        description="Type of semantic memory (graph, vector)"
    )
    
    # Episodic Memory (Nemori research)
    episodic_memory_enabled: bool = Field(
        default=True,
        description="Enable episodic memory"
    )
    
    episodic_segmentation: str = Field(
        default="semantic_boundaries",
        description="Segmentation strategy for episodic memory"
    )
    
    # ========================================================================
    # EVOLUTION SETTINGS
    # ========================================================================
    
    evolution_strategy: str = Field(
        default="darwin",
        description="Evolution strategy (darwin, lamarckian, island)"
    )
    
    evolution_population_size: int = Field(
        default=50,
        ge=2,
        description="Population size for evolution"
    )
    
    evolution_survival_rate: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Fraction of population that survives each generation"
    )
    
    evolution_mutation_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Mutation rate for evolution"
    )
    
    # ========================================================================
    # OBSERVABILITY SETTINGS
    # ========================================================================
    
    weave_project: str = Field(
        default="the-convergence",
        description="Weave project name for observability"
    )
    
    weave_enabled: bool = Field(
        default=True,
        description="Enable Weave tracing"
    )
    
    # ========================================================================
    # STORAGE SETTINGS
    # ========================================================================
    
    storage_backend: str = Field(
        default="sqlite",
        description="Storage backend (sqlite, file, memory, postgres, or custom)"
    )
    
    storage_path: Optional[Path] = Field(
        default=Path("./data/convergence.db"),
        description="Path to storage file (for sqlite/file backends)"
    )
    
    storage_serializer: str = Field(
        default="pickle",
        description="Serialization method (pickle or json)"
    )
    
    storage_ttl_seconds: Optional[int] = Field(
        default=None,
        description="Time-to-live for storage entries in seconds (memory backend only)"
    )
    
    storage_max_size: Optional[int] = Field(
        default=None,
        description="Maximum number of entries (memory backend only)"
    )
    
    # Database-specific settings
    storage_db_host: Optional[str] = Field(
        default="localhost",
        description="Database host (postgres backend)"
    )
    
    storage_db_port: Optional[int] = Field(
        default=5432,
        description="Database port (postgres backend)"
    )
    
    storage_db_name: Optional[str] = Field(
        default="convergence",
        description="Database name (postgres backend)"
    )
    
    storage_db_user: Optional[str] = Field(
        default=None,
        description="Database user (postgres backend)"
    )
    
    storage_db_password: Optional[str] = Field(
        default=None,
        description="Database password (postgres backend)"
    )
    
    # ========================================================================
    # PLUGIN SETTINGS
    # ========================================================================
    
    plugin_dirs: List[Path] = Field(
        default_factory=lambda: [Path("./plugins")],
        description="Directories to search for plugins"
    )
    
    # ========================================================================
    # GENERATION SETTINGS
    # ========================================================================
    
    generator_template_dir: Path = Field(
        default=Path("./convergence/templates"),
        description="Directory containing Jinja2 templates"
    )
    
    generator_output_dir: Path = Field(
        default=Path("./generated"),
        description="Directory for generated code"
    )
    
    # ========================================================================
    # METHODS
    # ========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.model_dump()
    
    @classmethod
    def from_yaml(cls, path: Path) -> "ConvergenceConfig":
        """Load config from YAML file."""
        import yaml
        
        with open(path) as f:
            data = yaml.safe_load(f)
        
        return cls(**data)
    
    @classmethod
    def from_json(cls, path: Path) -> "ConvergenceConfig":
        """Load config from JSON file."""
        import json
        
        with open(path) as f:
            data = json.load(f)
        
        return cls(**data)
    
    def save_yaml(self, path: Path) -> None:
        """Save config to YAML file."""
        import yaml
        
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def save_json(self, path: Path) -> None:
        """Save config to JSON file."""
        import json
        
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ============================================================================
# GLOBAL CONFIG
# ============================================================================

_global_config: Optional[ConvergenceConfig] = None


def get_config() -> ConvergenceConfig:
    """Get the global configuration."""
    global _global_config
    if _global_config is None:
        _global_config = ConvergenceConfig()
    return _global_config


def set_config(config: ConvergenceConfig) -> None:
    """Set the global configuration."""
    global _global_config
    _global_config = config


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _global_config
    _global_config = None

