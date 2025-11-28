# The Convergence

**API Optimization Framework powered by evolutionary algorithms, multi-armed bandits, and agent societies**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.7-orange.svg)](pyproject.toml)

The Convergence automatically finds optimal API configurations through intelligent experimentation. Instead of manually tuning parameters (model, temperature, tokens, etc.), it runs automated experiments, evaluates results, and evolves better configurations over multiple generations.

## 🚀 Quick Start

### Installation

```bash
pip install the-convergence
```

### 2-Minute Example

```bash
# Interactive setup wizard
convergence init

# Run optimization
convergence optimize optimization.yaml
```

**Or use the SDK:**

```python
from convergence import run_optimization
from convergence.types import ConvergenceConfig, ApiConfig, SearchSpaceConfig

config = ConvergenceConfig(
    api=ApiConfig(name="my_api", endpoint="https://api.example.com/v1/chat"),
    search_space=SearchSpaceConfig(parameters={
        "temperature": {"type": "float", "min": 0.1, "max": 1.5},
        "model": {"type": "categorical", "choices": ["gpt-4o-mini", "gpt-4o"]}
    }),
    evaluation=EvaluationConfig(required_metrics=["quality"], weights={"quality": 1.0}),
    runner=RunnerConfig(generations=10, population=20)
)

result = await run_optimization(config)
print(f"Best config: {result['best_config']}")
print(f"Best score: {result['best_score']}")
```

## 🎯 What It Does

The Convergence optimizes API parameters to maximize performance metrics:

- **Quality** - Response quality (LLM judge, similarity, exact match)
- **Latency** - Response time (milliseconds)
- **Cost** - Price per API call (USD)

**Example:** Find the best `temperature` and `model` combination that maximizes quality while minimizing cost.

## 🧬 How It Works

The Convergence combines three optimization strategies:

1. **Multi-Armed Bandits (MAB)** - Intelligent exploration vs exploitation
   - Thompson Sampling balances trying new configs vs exploiting known good ones
   - Bayesian probability guides selection

2. **Evolutionary Algorithms** - Genetic mutation and crossover
   - Mutation: Random parameter changes
   - Crossover: Combine two successful configs
   - Selection: Keep top performers (elitism)

3. **Reinforcement Learning (RL)** - Meta-learning from history
   - Learns which parameter ranges work best
   - Adjusts evolution parameters dynamically
   - Hierarchical learning across runs

**Optional:** Agent Society (RLP + SAO) for advanced reasoning and self-improvement.

## 📖 Documentation

- **[Getting Started](GETTING_STARTED.md)** - Complete setup guide
- **[SDK Usage](SDK_USAGE.md)** - Programmatic API reference
- **[Quick Start](QUICKSTART.md)** - Minimal examples
- **[YAML Configuration](YAML_CONFIGURATION_REFERENCE.md)** - Full config reference
- **[Examples](examples/)** - Working examples for OpenAI, Groq, Azure, BrowserBase, Agno agents

## 🏗️ Architecture

### Core Components

- **Optimization Engine** - Coordinates MAB, Evolution, RL
- **API Caller** - Makes HTTP requests with config parameters
- **Evaluator** - Scores responses against test cases
- **Storage** - Persists results (SQLite, File, Convex, Memory)
- **Adapters** - Provider-specific request/response transformations

### Entry Points

1. **CLI** - `convergence optimize config.yaml`
2. **SDK** - `from convergence import run_optimization`
3. **Runtime** - Per-request bandit selection for production

## 🔧 Features

### Optimization Modes

- **Batch Optimization** - Full optimization runs (CLI/SDK)
- **Runtime Selection** - Per-request config selection (production)
- **Continuous Evolution** - Arms evolve during production use

### Evaluation

- **Custom Evaluators** - Write Python functions for domain-specific scoring
- **Built-in Metrics** - Quality, latency, cost, exact match, similarity
- **Multi-Objective** - Optimize multiple metrics simultaneously

### Storage

- **Multi-Backend** - SQLite (default), File, Convex, Memory
- **Legacy System** - Tracks optimization history across runs
- **Warm-Start** - Resume from previous winners

### Provider Support

- **LLM APIs** - OpenAI, Azure OpenAI, Groq, Google Gemini
- **Web Automation** - BrowserBase
- **Agno Agents** - Discord, Gmail, Reddit agents
- **Local Functions** - Optimize internal Python functions

## 📦 Installation Options

### Basic Installation

```bash
pip install the-convergence
```

### With Agent Society (RLP + SAO)

```bash
pip install "the-convergence[agents]"
```

### With All Features

```bash
pip install "the-convergence[all]"
```

### Development Mode

```bash
git clone https://github.com/persist-os/the-convergence.git
cd the-convergence
pip install -e ".[dev]"
```

## 🎓 Example Use Cases

### 1. LLM API Optimization

Optimize ChatGPT parameters for your use case:

```yaml
api:
  name: "openai_chat"
  endpoint: "https://api.openai.com/v1/chat/completions"
  auth:
    type: "bearer"
    token_env: "OPENAI_API_KEY"

search_space:
  parameters:
    model: ["gpt-4o-mini", "gpt-4o"]
    temperature: [0.3, 0.5, 0.7, 0.9]
    max_tokens: [500, 1000, 2000]

evaluation:
  test_cases:
    path: "test_cases.json"
  metrics:
    quality: {weight: 0.6, type: "llm_judge"}
    latency_ms: {weight: 0.3}
    cost_usd: {weight: 0.1}
```

### 2. Context Enrichment Optimization

Optimize MAB parameters for context enrichment:

```python
from convergence import run_optimization

config = ConvergenceConfig(
    api=ApiConfig(name="context_enrichment", endpoint="http://backend:8000/api/enrich"),
    search_space=SearchSpaceConfig(parameters={
        "threshold": {"type": "float", "min": 0.1, "max": 0.5},
        "limit": {"type": "int", "min": 5, "max": 20}
    }),
    # ... evaluation config
)

result = await run_optimization(config)
```

### 3. Runtime Per-Request Selection

Use optimized configs in production:

```python
from convergence import configure_runtime, runtime_select, runtime_update

# Configure once
await configure_runtime("context_enrichment", config=config, storage=storage)

# Per request
selection = await runtime_select("context_enrichment", user_id="user_123")
# Use selection.params in your application

# After request
await runtime_update("context_enrichment", user_id="user_123", 
                     decision_id=selection.decision_id, reward=0.8)
```

## 🔍 How It Works (Detailed)

### Optimization Flow

1. **Initialization** - Load config, validate, initialize storage
2. **Generation Loop** (for each generation):
   - **Population Generation** - Create configs (random, mutation, crossover)
   - **MAB Selection** - Thompson Sampling selects configs to test
   - **Parallel Execution** - Test configs against test cases
   - **Evaluation** - Score responses (quality, latency, cost)
   - **Evolution** - Generate next generation (elite + mutation + crossover)
   - **RL Meta-Optimization** - Adjust evolution parameters
   - **Early Stopping** - Stop if converged or max generations reached
3. **Results Export** - Save best config, all results, reports

### Runtime Flow

1. **Selection** - Thompson Sampling selects arm (config) for request
2. **Execution** - Application uses selected config
3. **Update** - Record reward (quality, latency, cost)
4. **Evolution** - Periodically evolve arms (mutation, crossover)

## 🛠️ Configuration

### Minimal Config

```yaml
api:
  name: "my_api"
  endpoint: "https://api.example.com/v1/endpoint"
  auth:
    type: "bearer"
    token_env: "API_KEY"

search_space:
  parameters:
    param1: {type: "float", min: 0, max: 1}

evaluation:
  test_cases:
    inline:
      - {id: "test_1", input: {}, expected: {}}
  metrics:
    accuracy: {weight: 1.0}

optimization:
  algorithm: "mab_evolution"
  evolution:
    population_size: 10
    generations: 5
```

See [YAML_CONFIGURATION_REFERENCE.md](YAML_CONFIGURATION_REFERENCE.md) for complete reference.

## 📊 Results

Results are saved in `results/` directory:

- **`best_config.json`** - Best configuration found
- **`detailed_results.json`** - All configs tested with scores
- **`detailed_results.csv`** - CSV export
- **`report.md`** - Markdown report with analysis

## 🧪 Testing

```bash
# Run tests
pytest

# Test SDK import
python -c "from convergence import run_optimization; print('✅ Ready!')"
```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 License

Apache 2.0 - See [LICENSE](LICENSE) file.

## 🆘 Support

- **Documentation** - See `GETTING_STARTED.md`, `SDK_USAGE.md`, `examples/`
- **Issues** - [GitHub Issues](https://github.com/persist-os/the-convergence/issues)
- **Security** - See [SECURITY.md](SECURITY.md)

## 🗺️ Roadmap

- [ ] Automated test suite with pytest
- [ ] Performance benchmarking suite
- [ ] Plugin development tutorial
- [ ] Video documentation
- [ ] Integration examples for popular APIs

## 🙏 Acknowledgments

Built by the PersistOS team:
- Aria Han (aria@persistos.co)
- Shreyash Hamal (shrey@persistos.co)
- Myat Pyae Paing (paing@persistos.co)

---

**Happy optimizing! 🚀**

