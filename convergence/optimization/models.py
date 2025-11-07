"""
Data models for API optimization.

These Pydantic models define the complete structure of optimization.yaml
and all related data structures for the optimization engine.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator
from datetime import datetime
from pathlib import Path


class APIResponse(BaseModel):
    """Response from API call."""
    success: bool
    result: Any
    latency_ms: float
    error: Optional[str] = None
    estimated_cost_usd: float = 0.0


class OptimizationResult(BaseModel):
    """Result of optimization run."""
    best_config: Dict[str, Any]
    best_score: float
    all_results: List[Dict[str, Any]]
    generations_run: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AuthConfig(BaseModel):
    """Authentication configuration."""
    type: str = "none"  # bearer, api_key, basic, oauth, none
    token_env: Optional[str] = None  # Environment variable name
    header_name: Optional[str] = None  # For api_key type (e.g., "x-goog-api-key", "X-API-Key")
    username: Optional[str] = None  # For basic auth
    password_env: Optional[str] = None  # For basic auth
    
    @model_validator(mode='before')
    @classmethod
    def normalize_header_name(cls, values: Any) -> Any:
        """
        Support both 'api_key_header' (legacy) and 'header_name' field names.
        Normalize to 'header_name' internally.
        """
        if isinstance(values, dict):
            # If api_key_header is provided but header_name is not, use it
            if 'api_key_header' in values and not values.get('header_name'):
                values['header_name'] = values['api_key_header']
            
            # Ensure we have a default if type is api_key and header_name is None
            if values.get('type') == 'api_key' and not values.get('header_name'):
                values['header_name'] = 'x-api-key'
        
        return values


class RequestConfig(BaseModel):
    """HTTP request configuration."""
    method: str = "POST"
    headers: Dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = 30


class ResponseConfig(BaseModel):
    """API response parsing configuration."""
    success_field: str = "success"
    result_field: str = "result"
    error_field: str = "error"


class ModelConfig(BaseModel):
    """Model configuration in registry."""
    endpoint: str


class APIConfig(BaseModel):
    """API endpoint configuration."""
    name: str
    description: Optional[str] = None
    endpoint: Optional[str] = None  # Optional: single endpoint
    models: Optional[Dict[str, ModelConfig]] = None  # Optional: model registry
    auth: AuthConfig = Field(default_factory=AuthConfig)
    request: RequestConfig = Field(default_factory=RequestConfig)
    response: ResponseConfig = Field(default_factory=ResponseConfig)
    adapter_enabled: bool = False  # Enable API-specific adapter if available
    mock_mode: bool = False  # Skip real API calls, use mock responses
    
    @model_validator(mode='after')
    def validate_endpoint_or_models(self):
        """Ensure either endpoint or models is provided."""
        if not self.endpoint and not self.models:
            raise ValueError("Either 'endpoint' or 'models' must be provided in api config")
        if self.endpoint and self.models:
            raise ValueError("Cannot specify both 'endpoint' and 'models' - use one or the other")
        return self


class SearchSpaceParameter(BaseModel):
    """Parameter in search space."""
    type: str  # categorical, continuous, discrete
    values: Optional[List[Any]] = None
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None


class TemplateConfig(BaseModel):
    """Template configuration for prompt/request generation."""
    path: str
    variables: List[str] = Field(default_factory=list)


class SearchSpaceConfig(BaseModel):
    """Search space configuration."""
    parameters: Dict[str, SearchSpaceParameter]
    templates: Optional[Dict[str, TemplateConfig]] = None


class TestCaseAugmentationConfig(BaseModel):
    """Test case augmentation/evolution configuration."""
    enabled: bool = False
    mutation_rate: float = 0.3  # Probability of mutations
    crossover_rate: float = 0.2  # Probability of crossover vs mutation
    augmentation_factor: int = 2  # Variants per original test
    preserve_originals: bool = True  # Keep originals in augmented set


class TestCasesConfig(BaseModel):
    """Test cases configuration."""
    path: Optional[str] = None  # Path to JSON file
    inline: Optional[List[Dict[str, Any]]] = None  # Inline test cases
    augmentation: Optional[TestCaseAugmentationConfig] = None  # Auto-generate variants


class MetricConfig(BaseModel):
    """Evaluation metric configuration."""
    weight: float
    type: str  # higher_is_better, lower_is_better
    function: str = "custom"  # exact_match, similarity, custom
    threshold: Optional[float] = None
    budget_per_call: Optional[float] = None  # For cost metrics


class CustomEvaluatorConfig(BaseModel):
    """Custom evaluator configuration."""
    enabled: bool = False
    module: Optional[str] = None
    function: Optional[str] = None


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""
    test_cases: Optional[TestCasesConfig] = None
    metrics: Dict[str, MetricConfig]
    custom_evaluator: CustomEvaluatorConfig = Field(default_factory=CustomEvaluatorConfig)


class MABConfig(BaseModel):
    """Multi-Armed Bandit configuration."""
    strategy: str = "thompson_sampling"
    exploration_rate: float = 0.1
    confidence_level: float = 0.95


class EvolutionConfig(BaseModel):
    """Evolution algorithm configuration."""
    population_size: int = 20
    generations: int = 10
    mutation_rate: float = 0.2
    crossover_rate: float = 0.7
    elite_size: int = 2


class EarlyStoppingConfig(BaseModel):
    """Early stopping configuration."""
    enabled: bool = True
    patience: int = 3
    min_improvement: float = 0.01


class ExecutionConfig(BaseModel):
    """Execution configuration."""
    experiments_per_generation: int = 50
    parallel_workers: int = 5
    max_retries: int = 3
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)


class OptimizationAlgorithmConfig(BaseModel):
    """Optimization algorithm configuration."""
    algorithm: str = "mab_evolution"
    mab: MABConfig = Field(default_factory=MABConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)


class ExportConfig(BaseModel):
    """Best config export configuration."""
    enabled: bool = True
    format: str = "python"  # python, json, yaml
    output_path: str = "./best_config.py"


class OutputConfig(BaseModel):
    """Output configuration."""
    save_path: str = "./results/optimization_run"
    save_all_experiments: bool = True
    formats: List[str] = Field(default_factory=lambda: ["json", "markdown", "csv"])
    visualizations: List[str] = Field(default_factory=list)
    export_best_config: ExportConfig = Field(default_factory=ExportConfig)


class AgentRoleConfig(BaseModel):
    """Agent role configuration for society."""
    count: int = 1
    strategy: str = "random"
    memory: List[str] = Field(default_factory=list)


class CollaborationConfig(BaseModel):
    """Collaboration configuration."""
    enabled: bool = True
    trust_threshold: float = 0.7


class LearningConfig(BaseModel):
    """Learning configuration."""
    rlp_enabled: bool = True
    sao_enabled: bool = True


class StorageConfig(BaseModel):
    """Storage configuration."""
    backend: str = "multi"  # multi, sqlite, file, memory
    path: str = "./data/optimization"
    cache_enabled: bool = True


class WeaveConfig(BaseModel):
    """Weave observability configuration."""
    enabled: bool = True
    organization: Optional[str] = None  # Reads from WANDB_ENTITY or WEAVE_ORGANIZATION
    project: Optional[str] = None  # Reads from WANDB_PROJECT or WEAVE_PROJECT


class LLMConfig(BaseModel):
    """LLM configuration for agent society (RLP/SAO features)."""
    model: str = Field(
        default="gemini/gemini-2.0-flash-exp",
        description="LiteLLM model string (e.g., 'gemini/gemini-2.0-flash-exp', 'openai/gpt-4o')"
    )
    api_key_env: str = Field(
        default="GEMINI_API_KEY",
        description="Environment variable name containing API key (e.g., 'OPENAI_API_KEY', 'GEMINI_API_KEY')"
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1)


class SocietyConfig(BaseModel):
    """Optional agent society configuration (advanced)."""
    enabled: bool = False
    auto_generate_agents: bool = True
    agents: Optional[Dict[str, AgentRoleConfig]] = None
    collaboration: CollaborationConfig = Field(default_factory=CollaborationConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    weave: WeaveConfig = Field(default_factory=WeaveConfig)
    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="LLM configuration for agent society (RLP/SAO). Can reuse same API key as main API."
    )


class LegacyTrackingConfig(BaseModel):
    """Legacy system configuration - enables continuous learning across optimization runs."""
    enabled: bool = True  # Enabled by default for better user experience
    session_id: Optional[str] = None  # Auto-generated if not provided
    tracking_backend: str = "builtin"  # builtin, mlflow, aim, weave
    sqlite_path: str = "./data/legacy.db"
    export_dir: str = "./legacy"
    export_formats: List[str] = Field(default_factory=lambda: ["winners_only", "full_audit"])
    
    # Future: External tracker configs
    mlflow_config: Dict[str, Any] = Field(default_factory=dict)
    aim_config: Dict[str, Any] = Field(default_factory=dict)
    weave_config: Dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    """Configuration for a single model in the agent models registry.
    
    Uses the simplified format where endpoint is the complete URL (including deployment name and API version).
    This is the recommended format for all new configurations.
    
    Example:
        models:
          gpt-4.1:
            endpoint: "https://your-resource.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2025-01-01-preview"
            description: "GPT-4 deployment"
    """
    endpoint: str  # Full endpoint URL including deployment and API version
    description: Optional[str] = None


class RedditAuthConfig(BaseModel):
    """Reddit API authentication configuration."""
    client_id_env: str = "REDDIT_CLIENT_ID"
    client_secret_env: str = "REDDIT_CLIENT_SECRET"
    user_agent: str = "the-convergence-reddit-tester/1.0"
    username_env: Optional[str] = None
    password_env: Optional[str] = None


class DiscordAuthConfig(BaseModel):
    """Discord API authentication configuration."""
    bot_token_env: str = "DISCORD_BOT_TOKEN"


class AgentConfig(BaseModel):
    """Agent configuration for Agno-based optimizations."""
    reddit_auth: Optional[RedditAuthConfig] = None
    discord_auth: Optional[DiscordAuthConfig] = None
    models: Dict[str, ModelConfig] = Field(default_factory=dict)


class OptimizationSchema(BaseModel):
    """
    Complete optimization schema - this is the root model for optimization.yaml.
    
    This represents everything the user can configure in their YAML file.
    """
    api: APIConfig
    search_space: SearchSpaceConfig
    evaluation: EvaluationConfig
    optimization: OptimizationAlgorithmConfig = Field(default_factory=OptimizationAlgorithmConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    agent: Optional[AgentConfig] = None
    society: Optional[SocietyConfig] = None
    legacy: LegacyTrackingConfig = Field(default_factory=LegacyTrackingConfig)  # Enabled by default

