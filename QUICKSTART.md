# Quick Start: Building Self-Evolving Systems

## Install (30 seconds)

```bash
pip install the-convergence

# Verify
python -c "from convergence import run_optimization; print('Ready!')"
```

## Three Patterns (Choose Your Level)

### Level 1: One-Shot Optimization

Basic optimization - run once, get best config.

```python
from convergence import run_optimization

config = {
    "api": {
        "name": "my_system",
        "endpoint": "http://localhost:8000/api",
        "mock_mode": True
    },
    "search_space": {
        "parameters": {
            "temperature": {"type": "float", "min": 0.1, "max": 1.0},
            "model": {"type": "categorical", "choices": ["gpt-4o-mini", "gpt-4o"]}
        }
    },
    "evaluation": {
        "test_cases": {"inline": [{"id": "test_1", "input": {}}]},
        "metrics": {"quality": {"weight": 1.0}}
    }
}

result = await run_optimization(config_dict=config)
print(f"Best: {result['best_config']}")
```

### Level 2: Multi-Episode Learning

Enable warm-start - each run learns from previous runs.

```python
from convergence import run_optimization

config = {
    "api": {"name": "my_system", "endpoint": "...", "mock_mode": True},
    "search_space": {"parameters": {...}},
    "evaluation": {...},
    # Enable learning across runs
    "legacy": {
        "enabled": True,
        "storage_path": "./learning_history"
    }
}

# Run 1: Explores configuration space
result1 = await run_optimization(config_dict=config)
print(f"Episode 1 best: {result1['best_config']}")

# Run 2: Starts from Run 1's winners, explores further
result2 = await run_optimization(config_dict=config)
print(f"Episode 2 best: {result2['best_config']}")
# Likely better than Episode 1!
```

### Level 3: Production Runtime (Self-Evolving)

Per-request Thompson Sampling - system evolves during production use.

```python
from convergence import configure_runtime, runtime_select, runtime_update

# Configure once at startup
await configure_runtime(
    "my_endpoint",
    config=config,
    storage=storage
)

# Per-request: Select configuration
async def handle_request(user_id: str):
    # Thompson Sampling selects optimal config for this request
    selection = await runtime_select("my_endpoint", user_id=user_id)

    # Use the selected parameters
    result = await call_api(**selection.params)

    # Feed reward back - system learns!
    await runtime_update(
        "my_endpoint",
        user_id=user_id,
        decision_id=selection.decision_id,
        reward=calculate_reward(result)
    )

    return result
```

## Self-Improving Agents (Optional)

Enable RLP (reasoning before acting) and SAO (self-generated training):

```python
config = {
    # ... base config ...
    "society": {
        "enabled": True,
        "learning": {
            "rlp_enabled": True,   # Think before selecting configs
            "sao_enabled": True    # Generate own training data
        }
    }
}
```

## Important

- **Package name**: `the-convergence` (with hyphens)
- **Import name**: `convergence` (no hyphens)
- **Learning**: Enabled by default via `legacy` section
- **Start small**: 2-3 generations for testing, 10-20 for real optimization

## Full Documentation

- Getting Started: `GETTING_STARTED.md`
- SDK Reference: `SDK_USAGE.md`
- Examples: `examples/` directory
- Configuration: `YAML_CONFIGURATION_REFERENCE.md`

## Support

Issues/Questions: https://github.com/persist-os/the-convergence/issues

---

**Stop tuning. Start evolving.**
