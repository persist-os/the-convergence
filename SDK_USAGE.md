# Convergence SDK: Building Self-Learning Systems

Programmatic interface for building systems that evolve through experience.

## Installation

```bash
# Core framework
pip install the-convergence

# With self-improving agents (RLP + SAO)
pip install "the-convergence[agents]"

# Development mode
pip install -e .
```

## Three Optimization Patterns

### Pattern 1: Batch Optimization

Run optimization episodes to find optimal configurations.

```python
from convergence import run_optimization

config = {
    "api": {
        "name": "my_api",
        "endpoint": "https://api.example.com/v1/chat",
        "auth": {"type": "bearer", "token_env": "API_KEY"}
    },
    "search_space": {
        "parameters": {
            "temperature": {"type": "float", "min": 0.1, "max": 1.5},
            "model": {"type": "categorical", "choices": ["gpt-4o-mini", "gpt-4o"]}
        }
    },
    "evaluation": {
        "test_cases": {"inline": [{"id": "test_1", "input": {"prompt": "Hello"}}]},
        "metrics": {"quality": {"weight": 0.7}, "cost": {"weight": 0.3}}
    },
    "optimization": {
        "evolution": {"generations": 5, "population_size": 10}
    }
}

result = await run_optimization(config_dict=config)
print(f"Best: {result['best_config']} (score: {result['best_score']})")
```

### Pattern 2: Multi-Episode Learning

Enable warm-start to learn across optimization runs.

```python
config = {
    # ... base config ...
    "legacy": {
        "enabled": True,
        "storage_path": "./learning_history"
    }
}

# Episode 1: Explore
result1 = await run_optimization(config_dict=config)

# Episode 2: Start from Episode 1's winners
result2 = await run_optimization(config_dict=config)
# result2 benefits from result1's learnings
```

### Pattern 3: Runtime Selection (Production)

Per-request Thompson Sampling that evolves during production use.

```python
from convergence import configure_runtime, runtime_select, runtime_update

# Configure at startup
await configure_runtime(
    "my_endpoint",
    config=RuntimeConfigSDK(
        system="my_system",
        default_arms=[
            {"arm_id": "config_a", "params": {"temperature": 0.7}},
            {"arm_id": "config_b", "params": {"temperature": 0.5}},
        ]
    ),
    storage=my_storage
)

# Per-request handling
async def handle_request(user_id: str, request):
    # Thompson Sampling selects configuration
    selection = await runtime_select("my_endpoint", user_id=user_id)

    # Use selected parameters
    response = await call_api(**selection.params)

    # Feed reward back - system learns!
    reward = calculate_quality(response)
    await runtime_update(
        "my_endpoint",
        user_id=user_id,
        decision_id=selection.decision_id,
        reward=reward
    )

    return response
```

This is the **self-evolving pattern** - your system improves with every request.

---

## Response Format

### Success Response

```python
{
    "success": True,
    "best_config": {"temperature": 0.7, "model": "gpt-4o-mini"},
    "best_score": 0.87,
    "configs_generated": 50,
    "generations_run": 5,
    "learning_session_id": "my_api_1729375200_a1b2c3d4",
    "timestamp": "2025-10-19T12:00:00"
}
```

### Error Response

```python
{
    "success": False,
    "error": "Config validation failed: missing 'api' field",
    "configs_generated": 0
}
```

---

## Configuration Reference

### Minimal Config

```python
config = {
    "api": {
        "name": "my_api",
        "endpoint": "https://api.example.com",
        "mock_mode": True  # For testing
    },
    "search_space": {
        "parameters": {
            "param1": {"type": "float", "min": 0, "max": 1}
        }
    },
    "evaluation": {
        "test_cases": {"inline": [{"id": "test_1", "input": {}}]},
        "metrics": {"accuracy": {"weight": 1.0}}
    }
}
```

### Learning Configuration

```python
config = {
    # ... base config ...

    # Cross-run learning (warm-start)
    "legacy": {
        "enabled": True,
        "storage_path": "./learning_history"
    },

    # Self-improving agents
    "society": {
        "enabled": True,
        "learning": {
            "rlp_enabled": True,  # Think before acting
            "sao_enabled": True   # Self-generate training data
        }
    }
}
```

### Full Config Examples

See `examples/` directory:
- `examples/ai/openai/` - LLM optimization
- `examples/web_browsing/browserbase/` - Web automation
- `examples/agno_agents/` - Agent optimization

---

## Runtime Storage Protocol

For production runtime selection, implement the storage protocol:

```python
from convergence.storage.runtime_protocol import RuntimeStorageProtocol

class MyRuntimeStorage(RuntimeStorageProtocol):
    async def get_arms(self, *, user_id: str, agent_type: str):
        """Get current arms (configurations) for user."""
        ...

    async def initialize_arms(self, *, user_id: str, agent_type: str, arms):
        """Initialize arms for new user."""
        ...

    async def create_decision(self, *, user_id: str, agent_type: str,
                              arm_pulled: str, strategy_params, arms_snapshot,
                              metadata=None):
        """Record a selection decision."""
        ...

    async def update_performance(self, *, user_id: str, agent_type: str,
                                 decision_id: str, reward: float,
                                 engagement=None, metadata=None):
        """Update arm with reward signal."""
        ...

    async def get_decision(self, *, user_id: str, decision_id: str):
        """Retrieve decision for analysis."""
        ...
```

See `convergence/storage/runtime_protocol.py` for complete interface.

---

## Self-Improving Agents

### RLP (Reinforcement Learning on Policy)

Agents think before selecting configurations:

```python
config = {
    "society": {
        "enabled": True,
        "learning": {
            "rlp_enabled": True
        },
        "llm": {
            "model": "gpt-4o-mini",
            "api_key_env": "OPENAI_API_KEY"
        }
    }
}
```

RLP generates internal reasoning before each generation:
- "Based on previous results, temperatures around 0.7 work best..."
- Reasoning is rewarded when it improves prediction accuracy
- Dense learning signal at every decision

### SAO (Self-Alignment Optimization)

Agents generate their own training data:

```python
config = {
    "society": {
        "enabled": True,
        "learning": {
            "sao_enabled": True
        }
    }
}
```

SAO creates preference pairs from optimization history:
- Generates prompts via persona role-play
- Creates response comparisons
- Self-judges to create preference labels
- No external labeling required

---

## Error Handling

```python
try:
    result = await run_optimization(config_dict=config)

    if not result["success"]:
        logger.error(f"Optimization failed: {result['error']}")
        return None

    return result["best_config"]

except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
    return None
```

---

## Testing

```bash
# Test SDK import
python -c "from convergence import run_optimization; print('Ready!')"

# Run with mock mode
python -c "
import asyncio
from convergence import run_optimization

config = {
    'api': {'name': 'test', 'endpoint': 'http://test', 'mock_mode': True},
    'search_space': {'parameters': {'x': {'type': 'float', 'min': 0, 'max': 1}}},
    'evaluation': {'test_cases': {'inline': [{'id': 't1', 'input': {}}]}, 'metrics': {'a': {'weight': 1}}}
}

result = asyncio.run(run_optimization(config_dict=config))
print(f'Success: {result[\"success\"]}')
"
```

---

## Notes

- **Mock Mode**: Use `mock_mode: True` for testing without real API calls
- **Generations**: 3-5 for testing, 10-20 for production
- **Population**: 10-20 works for most use cases
- **Parallel Workers**: Set to 1 for sequential evaluation (easier debugging)
- **Learning**: Enable `legacy.enabled` for cross-run improvement

---

## Support

- **Documentation**: `README.md`, `GETTING_STARTED.md`
- **Examples**: `examples/` directory
- **Issues**: https://github.com/persist-os/the-convergence/issues

---

**Stop tuning. Start evolving.**
