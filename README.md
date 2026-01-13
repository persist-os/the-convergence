# The Convergence

**Self-evolving agent framework powered by reinforcement learning**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.8-orange.svg)](pyproject.toml)

Systems that improve themselves outperform systems you tune manually. The Convergence is a framework for building agents that learn optimal behavior through experience - using Thompson Sampling, evolutionary algorithms, and self-improving policy networks.

## The Problem

You're tuning parameters by hand. Temperature, model selection, context limits, sampling strategies - all configured once and left static. But optimal parameters depend on your data, your users, your use case. They change over time. Manual tuning can't keep up.

## The Solution

Let your system learn. The Convergence treats every decision as a learning opportunity:

- **Thompson Sampling** explores the configuration space intelligently
- **Evolutionary algorithms** breed better configurations from successful ones
- **Dense reward signals** update beliefs after every interaction
- **Self-improving policies** (RLP + SAO) generate their own training data

The result: systems that converge toward optimal behavior automatically.

## Quick Start

```bash
pip install the-convergence
```

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
# Watch your system evolve toward optimal
```

Or use the CLI:

```bash
convergence init      # Interactive setup
convergence optimize config.yaml
```

## How It Works

The Convergence combines three reinforcement learning strategies that work together:

### 1. Thompson Sampling (Bayesian Exploration)

Every configuration maintains a probability distribution over its expected reward. Selection samples from these distributions, naturally balancing exploration of uncertain options with exploitation of known good ones.

```
Config A: Beta(15, 5) → sample 0.73
Config B: Beta(8, 12) → sample 0.42
Config C: Beta(2, 2)  → sample 0.61  ← High uncertainty, worth exploring

Select: A (highest sample)
```

### 2. Evolutionary Algorithms (Genetic Optimization)

Successful configurations breed. The population evolves through:

- **Selection**: Top performers survive (elitism)
- **Mutation**: Random parameter changes explore nearby space
- **Crossover**: Combine traits from two successful parents

Each generation is better than the last.

### 3. Self-Improving Agents (RLP + SAO)

Based on cutting-edge research from NVIDIA and Hugging Face (Oct 2024):

**RLP (Reinforcement Learning on Policy)**: Agents think before acting. Internal reasoning is rewarded when it improves prediction accuracy - creating dense learning signals without external verifiers.

**SAO (Self-Alignment Optimization)**: Agents generate their own training data. Through persona-based prompting and self-judgment, the system creates preference pairs for continuous improvement - no human labeling required.

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                  OPTIMIZATION LOOP                      │
│                                                        │
│   ┌──────────┐   ┌───────────┐   ┌──────────────┐    │
│   │ Thompson │──▶│ Evolution │──▶│ RL Meta-     │    │
│   │ Sampling │   │  Engine   │   │ Optimizer    │    │
│   └────┬─────┘   └─────┬─────┘   └──────┬───────┘    │
│        │               │                 │            │
│        │    ┌──────────┴──────────┐     │            │
│        └───▶│   Test Population   │◀────┘            │
│             │   (parallel eval)   │                  │
│             └──────────┬──────────┘                  │
│                        │                             │
│             ┌──────────▼──────────┐                  │
│             │   Reward Signals    │                  │
│             │ (quality, latency,  │                  │
│             │  cost, custom...)   │                  │
│             └──────────┬──────────┘                  │
│                        │                             │
│   ┌────────────────────┼────────────────────┐       │
│   │                    │                    │       │
│   ▼                    ▼                    ▼       │
│ ┌─────┐          ┌──────────┐        ┌─────────┐   │
│ │ RLP │          │ Storage  │        │   SAO   │   │
│ │Think│          │ (SQLite, │        │  Self-  │   │
│ │First│          │  Convex) │        │  Train  │   │
│ └─────┘          └──────────┘        └─────────┘   │
│                                                     │
└────────────────────────────────────────────────────┘
```

## Entry Points

**1. Batch Optimization** - Full optimization runs

```python
from convergence import run_optimization
result = await run_optimization(config)
```

**2. Runtime Selection** - Per-request bandit in production

```python
from convergence import configure_runtime, runtime_select, runtime_update

await configure_runtime("my_endpoint", config=config)
selection = await runtime_select("my_endpoint", user_id="user_123")
# Use selection.params in your application
await runtime_update("my_endpoint", decision_id=selection.decision_id, reward=0.8)
```

**3. CLI** - Interactive setup and optimization

```bash
convergence init
convergence optimize config.yaml
```

## What You Can Optimize

- **LLM APIs** - OpenAI, Azure OpenAI, Groq, Google Gemini
- **Web Automation** - BrowserBase parameters
- **Agent Systems** - Discord, Gmail, Reddit agents via Agno
- **Custom Endpoints** - Any HTTP API
- **Local Functions** - Pure Python functions

## Installation

```bash
# Core framework
pip install the-convergence

# With self-improving agents (RLP + SAO)
pip install "the-convergence[agents]"

# Everything
pip install "the-convergence[all]"
```

## Configuration

```yaml
api:
  name: "my_api"
  endpoint: "https://api.example.com/v1/chat"
  auth:
    type: "bearer"
    token_env: "API_KEY"

search_space:
  parameters:
    temperature: {type: "float", min: 0.1, max: 1.5}
    model: {type: "categorical", choices: ["gpt-4o-mini", "gpt-4o"]}

evaluation:
  test_cases:
    path: "test_cases.json"
  metrics:
    quality: {weight: 0.6, type: "llm_judge"}
    latency_ms: {weight: 0.3}
    cost_usd: {weight: 0.1}

optimization:
  algorithm: "mab_evolution"
  evolution:
    population_size: 20
    generations: 10

# Enable self-improving agents
society:
  enabled: true
  learning:
    rlp_enabled: true   # Think before acting
    sao_enabled: true   # Self-generate training data
```

## Results

After optimization, find your evolved configurations:

- `results/best_config.json` - Optimal configuration
- `results/detailed_results.json` - Full evolution history
- `results/report.md` - Analysis and recommendations

## Documentation

- **[Getting Started](GETTING_STARTED.md)** - Complete setup guide
- **[SDK Usage](SDK_USAGE.md)** - Programmatic API reference
- **[YAML Configuration](YAML_CONFIGURATION_REFERENCE.md)** - Full config reference
- **[Examples](examples/)** - Working examples

## Research Foundation

The Convergence builds on:

- **Thompson Sampling** - Bayesian approach to exploration/exploitation
- **Evolutionary Strategies** - Genetic algorithms for optimization
- **RLP** - [Reinforcement Learning on Policy](https://arxiv.org/abs/2510.01265) (NVIDIA, Oct 2024)
- **SAO** - [Self-Alignment Optimization](https://arxiv.org/abs/2510.06652) (Hugging Face, Oct 2024)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache 2.0 - See [LICENSE](LICENSE) file.

## Team

Built by [PersistOS](https://persistos.co):
- Aria Han
- Shreyash Hamal
- Myat Pyae Paing

---

**Stop tuning. Start evolving.**
