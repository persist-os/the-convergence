# YAML Configuration Reference

Complete reference for all configuration options available in Convergence optimization.yaml files.

**Last Updated:** 2025-01-26

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Sections](#core-sections)
   - [API Configuration](#api-configuration)
   - [Search Space](#search-space)
   - [Evaluation](#evaluation)
   - [Optimization](#optimization)
   - [Output](#output)
3. [Advanced Sections](#advanced-sections)
   - [Agent Configuration](#agent-configuration)
   - [Society Configuration](#society-configuration)
   - [Legacy Tracking](#legacy-tracking)
4. [Configuration Examples](#configuration-examples)

---

## Quick Start

Minimal working configuration:

```yaml
api:
  name: "my_api"
  endpoint: "https://api.example.com/v1/chat"

search_space:
  parameters:
    temperature:
      type: "continuous"
      min: 0.0
      max: 1.0

evaluation:
  test_cases:
    path: "test_cases.json"
  metrics:
    quality:
      weight: 1.0
      type: "higher_is_better"
```

---

## Core Sections

### API Configuration

Defines the API endpoint, authentication, and request format.

#### Basic Structure

```yaml
api:
  name: str                                    # Required: API identifier
  description: str                             # Optional: Human-readable description
  endpoint: str                                # Required (unless using models): API endpoint URL
  models: Dict[str, ModelConfig]              # Optional: Multi-model registry
  auth: AuthConfig                            # Required: Authentication settings
  request: RequestConfig                      # Optional: HTTP request settings
  response: ResponseConfig                    # Optional: Response parsing rules
  adapter_enabled: bool                       # Optional: Enable API-specific adapter
  mock_mode: bool                             # Optional: Skip real API calls (for testing)
```

#### Field Details

**`name`** (required)  
- Type: `str`
- Description: Unique identifier for this API
- Example: `"openai_responses"`

**`description`** (optional)  
- Type: `str`
- Description: Human-readable description of what this API does
- Example: `"OpenAI Responses API for GPT-4, O3, and other models"`

**`endpoint`** (required, unless `models` is used)  
- Type: `str`
- Description: Full URL to the API endpoint
- Example: `"https://api.openai.com/v1/chat/completions"`
- Note: Either `endpoint` OR `models` must be provided, not both

**`models`** (required if `endpoint` is not provided)  
- Type: `Dict[str, ModelConfig]`
- Description: Model registry for multi-model configurations
- Structure: `{model_name: {endpoint: str, description: Optional[str]}}`
- Example:
```yaml
models:
  gpt-4:
    endpoint: "https://api.openai.com/v1/models/gpt-4"
    description: "GPT-4 Turbo"
  o4-mini:
    endpoint: "https://api.openai.com/v1/models/o4-mini"
    description: "O1 Mini for fast responses"
```

**`auth`** (required)  
- Type: `AuthConfig`
- Structure:
```yaml
auth:
  type: str                    # "bearer", "api_key", "basic", "oauth", "none"
  token_env: str               # Environment variable name for token/API key
  header_name: str             # For api_key type: header name (default: "x-api-key")
  username: str                # For basic auth
  password_env: str            # Environment variable for password
```
- Example (Bearer):
```yaml
auth:
  type: "bearer"
  token_env: "OPENAI_API_KEY"
```
- Example (API Key):
```yaml
auth:
  type: "api_key"
  token_env: "GROQ_API_KEY"
  header_name: "x-api-key"  # Default, can be omitted
```
- Example (Basic):
```yaml
auth:
  type: "basic"
  username: "myuser"
  password_env: "MY_PASSWORD"
```

**`request`** (optional, has defaults)  
- Type: `RequestConfig`
- Defaults:
  - `method: "POST"`
  - `headers: {}`
  - `timeout_seconds: 30`
- Example:
```yaml
request:
  method: "POST"
  headers:
    Content-Type: "application/json"
    User-Agent: "Convergence-Optimizer/1.0"
  timeout_seconds: 60
```

**`response`** (optional, has defaults)  
- Type: `ResponseConfig`
- Description: How to parse API responses
- Defaults:
  - `success_field: "success"`
  - `result_field: "result"`
  - `error_field: "error"`
- Example:
```yaml
response:
  success_field: "status"      # Check if status == "completed"
  result_field: "output"       # Extract output array
  error_field: "error"         # Extract error message
```

**`adapter_enabled`** (optional)  
- Type: `bool`
- Default: `false`
- Description: Enable API-specific adapters for non-standard formats
- Examples:
  - Azure OpenAI (adds model in request body)
  - Gemini (formats responses)
  - BrowserBase (handles screenshots)
  - Reddit Agents (Agno integration)

**`mock_mode`** (optional)  
- Type: `bool`
- Default: `false`
- Description: Skip real API calls, use mock responses (for testing)
- Use case: Validate configuration without incurring API costs

---

### Search Space

Defines what parameters to optimize and their possible values.

#### Basic Structure

```yaml
search_space:
  parameters: Dict[str, SearchSpaceParameter]
  templates: Dict[str, TemplateConfig]        # Optional
```

#### Parameter Types

Three parameter types are supported:

**1. Categorical** (discrete choices)
```yaml
parameters:
  model:
    type: "categorical"
    values: ["gpt-4", "gpt-3.5-turbo", "claude-3"]
    description: "Model selection"
```

**2. Continuous** (numeric range)
```yaml
parameters:
  temperature:
    type: "continuous"
    min: 0.0              # Required
    max: 1.0              # Required
    step: 0.1             # Optional: sampling step
    description: "Sampling temperature"
```

**3. Discrete** (specific numeric values)
```yaml
parameters:
  max_tokens:
    type: "discrete"
    values: [256, 512, 1024, 2048, 4096]    # Required
    description: "Maximum tokens in response"
```

#### Templates (Optional)

For complex request formats, you can define templates:

```yaml
templates:
  system_prompt:
    path: "templates/system_prompt.txt"
    variables: ["user_intent", "context"]
  user_message:
    path: "templates/user_message.md"
    variables: ["question"]
```

---

### Evaluation

Defines how to test and score configurations.

#### Basic Structure

```yaml
evaluation:
  test_cases: TestCasesConfig                 # Required
  metrics: Dict[str, MetricConfig]           # Required
  custom_evaluator: CustomEvaluatorConfig    # Optional
```

#### Test Cases

**From File:**
```yaml
test_cases:
  path: "test_cases.json"  # Relative to config file
```

**Inline:**
```yaml
test_cases:
  inline:
    - input: "What is 2+2?"
      expected: "4"
      metadata:
        category: "math"
        difficulty: "easy"
    - input: "Explain quantum mechanics"
      expected: "..."
      metadata:
        category: "science"
        difficulty: "hard"
```

**With Augmentation:**
```yaml
test_cases:
  path: "test_cases.json"
  augmentation:
    enabled: true
    mutation_rate: 0.3      # 30% mutation probability
    crossover_rate: 0.2     # 20% crossover probability
    augmentation_factor: 2  # Generate 2 variants per original
    preserve_originals: true # Keep original test cases
```

#### Metrics

Each metric requires:

```yaml
metrics:
  response_quality:
    weight: 0.4                    # Required: Importance (0.0-1.0, sum to 1.0)
    type: "higher_is_better"       # Required: "higher_is_better" or "lower_is_better"
    function: "custom"             # Optional: "exact_match", "similarity", "custom"
    threshold: 0.8                 # Optional: Minimum acceptable value
    budget_per_call: 0.10          # Optional: Cost budget for cost metrics
```

**Function Types:**
- `"custom"`: Use custom evaluator (default)
- `"exact_match"`: Exact string comparison
- `"similarity"`: Embedding-based similarity

**Common Metrics:**
```yaml
metrics:
  # Quality
  response_quality:
    weight: 0.40
    type: "higher_is_better"
    function: "custom"
  
  # Speed
  latency_ms:
    weight: 0.25
    type: "lower_is_better"
    threshold: 5000
  
  # Cost
  cost_per_call:
    weight: 0.20
    type: "lower_is_better"
    budget_per_call: 0.10
  
  # Efficiency
  token_efficiency:
    weight: 0.15
    type: "higher_is_better"
    function: "custom"
```

#### Custom Evaluator

Define custom scoring logic:

```yaml
custom_evaluator:
  enabled: true
  module: "evaluator"        # Python module name (without .py)
  function: "score_response" # Function name in the module
```

**Evaluator Function Signature:**
```python
def score_response(response: str, expected: str, metadata: dict) -> dict:
    """
    Score a response against expected output.
    
    Args:
        response: Actual API response
        expected: Expected output
        metadata: Test case metadata
    
    Returns:
        Dictionary with metric scores:
        {
            "response_quality": 0.95,
            "latency_ms": 1234.0,
            "cost_per_call": 0.05
        }
    """
    scores = {
        "response_quality": 0.95,
        "latency_ms": 1234.0,
        "cost_per_call": 0.05
    }
    return scores
```

---

### Optimization

Defines the optimization algorithm and execution settings.

#### Basic Structure

```yaml
optimization:
  algorithm: str                              # "mab_evolution", "genetic", "mab_only"
  mab: MABConfig                             # Multi-Armed Bandit settings
  evolution: EvolutionConfig                # Evolutionary algorithm settings
  execution: ExecutionConfig                # Execution settings
```

#### Algorithm Types

- `"mab_evolution"`: MAB for exploration, evolution for refinement (recommended)
- `"genetic"`: Pure evolutionary algorithm
- `"mab_only"`: Multi-Armed Bandit only (no evolution)

#### MAB Configuration

```yaml
mab:
  strategy: "thompson_sampling"  # "thompson_sampling", "epsilon_greedy", "ucb1"
  exploration_rate: 0.2          # 0.0-1.0: Higher = more exploration
  confidence_level: 0.95         # Confidence level for Thompson Sampling (0.0-1.0)
```

#### Evolution Configuration

```yaml
evolution:
  population_size: 20            # Number of configs per generation
  generations: 10                # Number of evolutionary generations
  mutation_rate: 0.2             # 0.0-1.0: Probability of mutation
  crossover_rate: 0.7            # 0.0-1.0: Probability of crossover
  elite_size: 2                  # Number of best configs to preserve
```

#### Execution Configuration

```yaml
execution:
  experiments_per_generation: 50  # Number of experiments before evolution
  parallel_workers: 5             # Parallel API calls (respect rate limits!)
  max_retries: 3                  # Retry failed requests
  early_stopping:
    enabled: true
    patience: 3                   # Stop if no improvement for N generations
    min_improvement: 0.01         # Minimum 1% improvement to count as progress
```

**Early Stopping:**
- Stops optimization if no improvement for `patience` generations
- Improvement must be at least `min_improvement` to count
- Saves time and API costs on converged problems

---

### Output

Defines where and how to save results.

#### Basic Structure

```yaml
output:
  save_path: str                              # Required: Output directory
  save_all_experiments: bool                  # Save detailed results
  formats: List[str]                         # ["json", "markdown", "csv"]
  visualizations: List[str]                  # Visualization types
  export_best_config: ExportConfig           # Export best configuration
```

#### Field Details

**`save_path`** (required)  
- Type: `str`
- Description: Directory to save results
- Example: `"./results/my_optimization"`

**`save_all_experiments`** (optional)  
- Type: `bool`
- Default: `true`
- Description: Save detailed results for every experiment (vs. just summary)

**`formats`** (optional)  
- Type: `List[str]`
- Default: `["json", "markdown", "csv"]`
- Options: `"json"`, `"markdown"`, `"csv"`

**`visualizations`** (optional)  
- Type: `List[str]`
- Options:
  - `"score_over_time"`
  - `"parameter_importance"`
  - `"pareto_front"`
  - `"cost_vs_quality"`

**`export_best_config`** (optional)  
- Structure:
```yaml
export_best_config:
  enabled: true
  format: "python"        # "python", "json", "yaml"
  output_path: "./best_config.py"
```

---

## Advanced Sections

### Agent Configuration

For agent-based optimizations (Reddit agents, browser automation, etc.)

```yaml
agent:
  reddit_auth:
    client_id_env: "REDDIT_CLIENT_ID"
    client_secret_env: "REDDIT_CLIENT_SECRET"
    user_agent: "my-app/1.0"
    username_env: "REDDIT_USERNAME"
    password_env: "REDDIT_PASSWORD"
  
  models:
    gpt-4:
      endpoint: "https://api.openai.com/v1/chat/completions"
      description: "GPT-4 for agent reasoning"
```

---

### Society Configuration

Enables AI agent society for collaborative optimization (optional, advanced).

```yaml
society:
  enabled: false                            # Set to true to enable
  auto_generate_agents: true               # Auto-create agent roles
  
  # Agent Learning
  learning:
    rlp_enabled: true   # Reasoning-based Learning Process
    sao_enabled: true   # Self-Alignment Optimization
  
  # Agent Collaboration
  collaboration:
    enabled: true
    trust_threshold: 0.7        # 0.0-1.0: Minimum trust level for agent collaboration
  
  # LLM for Agent Society
  llm:
    model: "gemini/gemini-2.0-flash-exp"    # LiteLLM model string
    api_key_env: "GEMINI_API_KEY"           # Environment variable name
    temperature: 0.7
    max_tokens: 1000
  
  # Storage for Agent Society
  storage:
    backend: "multi"                        # "multi", "sqlite", "file", "memory"
    path: "./data/optimization"
    cache_enabled: true
  
  # Observability (Weave integration for logging/tracking)
  weave:
    enabled: true
    organization: Optional[str]            # Optional: Reads from WANDB_ENTITY or WEAVE_ORGANIZATION env var if not set
    project: Optional[str]                 # Optional: Reads from WANDB_PROJECT or WEAVE_PROJECT env var if not set
```

---

### Legacy Tracking

Continuous learning across optimization runs (enabled by default).

```yaml
legacy:
  enabled: true                            # Enabled by default
  session_id: "my_optimization"            # Optional: group related runs
  tracking_backend: "builtin"              # "builtin", "mlflow", "aim", "weave"
  sqlite_path: "./data/legacy.db"          # SQLite database path
  export_dir: "./legacy"                   # Export directory for CSV/JSON
  export_formats: ["winners_only", "full_audit"]
  
  # Future: External tracker configs
  mlflow_config:
    enabled: false
    tracking_uri: "http://localhost:5000"
    experiment_name: "my-experiment"
  
  aim_config:
    enabled: false
    repo_path: "./aim_repo"
  
  weave_config:
    enabled: false
    project_name: "my-legacy-tracking"
```

**What Legacy Tracking Does:**
- Saves winning configurations across runs
- Enables "warm start" (reuse previous winners)
- Tracks improvement over time
- Exports results to CSV/JSON for analysis

---

## Configuration Examples

### Minimal OpenAI Configuration

```yaml
api:
  name: "openai"
  endpoint: "https://api.openai.com/v1/chat/completions"
  auth:
    type: "bearer"
    token_env: "OPENAI_API_KEY"

search_space:
  parameters:
    temperature:
      type: "continuous"
      min: 0.0
      max: 1.0

evaluation:
  test_cases:
    path: "test_cases.json"
  metrics:
    quality:
      weight: 1.0
      type: "higher_is_better"
```

### Multi-Model Azure Configuration

```yaml
api:
  name: "azure_multi_model"
  description: "Azure OpenAI with multiple models"
  auth:
    type: "api_key"
    token_env: "AZURE_API_KEY"
    header_name: "api-key"
  models:
    gpt-4:
      endpoint: "https://your-resource.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2025-01-01-preview"
    o4-mini:
      endpoint: "https://your-resource.openai.azure.com/openai/deployments/o4-mini/chat/completions?api-version=2025-01-01-preview"

search_space:
  parameters:
    model:
      type: "categorical"
      values: ["gpt-4", "o4-mini"]
    temperature:
      type: "continuous"
      min: 0.0
      max: 1.0

evaluation:
  test_cases:
    path: "test_cases.json"
  metrics:
    response_quality:
      weight: 0.4
      type: "higher_is_better"
    latency_ms:
      weight: 0.3
      type: "lower_is_better"
    cost_per_call:
      weight: 0.3
      type: "lower_is_better"

output:
  save_path: "./results/azure_optimization"
```

### Advanced Configuration with Agent Society

```yaml
api:
  name: "my_api"
  endpoint: "https://api.example.com/chat"
  auth:
    type: "api_key"
    token_env: "MY_API_KEY"

search_space:
  parameters:
    temperature:
      type: "continuous"
      min: 0.0
      max: 2.0
      step: 0.1
    max_tokens:
      type: "discrete"
      values: [256, 512, 1024, 2048]

evaluation:
  test_cases:
    path: "test_cases.json"
    augmentation:
      enabled: true
      mutation_rate: 0.3
      augmentation_factor: 2
      preserve_originals: true
  metrics:
    quality:
      weight: 0.5
      type: "higher_is_better"
    speed:
      weight: 0.3
      type: "lower_is_better"
    cost:
      weight: 0.2
      type: "lower_is_better"
  custom_evaluator:
    enabled: true
    module: "evaluator"
    function: "score_response"

optimization:
  algorithm: "mab_evolution"
  mab:
    strategy: "thompson_sampling"
    exploration_rate: 0.2
  evolution:
    population_size: 10
    generations: 5
    mutation_rate: 0.3
    crossover_rate: 0.7
    elite_size: 2
  execution:
    experiments_per_generation: 100
    parallel_workers: 5
    max_retries: 3
    early_stopping:
      enabled: true
      patience: 3
      min_improvement: 0.01

output:
  save_path: "./results/my_optimization"
  save_all_experiments: true
  formats: ["json", "markdown", "csv"]
  visualizations:
    - "score_over_time"
    - "parameter_importance"
    - "cost_vs_quality"
  export_best_config:
    enabled: true
    format: "python"
    output_path: "./best_config.py"

society:
  enabled: true
  auto_generate_agents: true
  learning:
    rlp_enabled: true
    sao_enabled: true
  llm:
    model: "gemini/gemini-2.0-flash-exp"
    api_key_env: "GEMINI_API_KEY"
    temperature: 0.7
    max_tokens: 1000

legacy:
  enabled: true
  session_id: "my_optimization"
  export_formats: ["winners_only"]
```

---

## Best Practices

1. **Start Simple**: Begin with minimal configuration, add complexity as needed
2. **Set Reasonable Limits**: Use early stopping to avoid excessive API calls
3. **Parallel Workers**: Keep parallel_workers low (2-5) to respect rate limits
4. **Test Cases**: Start with 3-5 test cases, expand as you validate
5. **Metrics**: Keep total weight = 1.0 across all metrics
6. **Legacy Tracking**: Enable for continuous improvement across runs
7. **Mock Mode**: Use `mock_mode: true` to validate configuration without API costs

---

## Troubleshooting

**"Field required" errors:**
- Ensure all required fields are present
- Check spelling and indentation (YAML is sensitive)

**"Either 'endpoint' or 'models' must be provided":**
- Provide either `api.endpoint` (single model) OR `api.models` (multi-model), not both

**"Weight sum does not equal 1.0":**
- Sum all metric weights to exactly 1.0

**"Environment variable not found":**
- Set environment variable: `export MY_API_KEY="your-key"`
- Or specify in `.env` file

**Rate limit errors:**
- Reduce `parallel_workers` to 1
- Increase `max_retries` and add delays

---

## Additional Resources

- **Examples**: See `examples/` directory for complete working configurations
- **Documentation**: `GETTING_STARTED.md`, `TROUBLESHOOTING.md`
- **Support**: GitHub Issues for questions and bug reports
