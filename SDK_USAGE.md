# Convergence SDK Usage Guide

Simple programmatic interface for running Convergence optimizations from Python code.

## Installation

### For Development (Editable Install)

```bash
# In the-convergence directory
pip install -e .
```

### For Production

```bash
pip install the-convergence
```

## Quick Start

### Basic Usage

```python
from convergence import run_optimization

# Define optimization config
config = {
    "api": {
        "name": "my_api",
        "endpoint": "https://api.example.com/v1/chat",
        "request": {
            "method": "POST",
            "headers": {"Content-Type": "application/json"},
            "timeout_seconds": 30
        },
        "auth": {
            "type": "bearer",
            "token_env": "API_KEY"
        }
    },
    "search_space": {
        "parameters": {
            "temperature": {"type": "float", "min": 0.1, "max": 1.5},
            "max_tokens": {"type": "int", "min": 100, "max": 2000}
        }
    },
    "evaluation": {
        "test_cases": {
            "inline": [
                {
                    "id": "test_1",
                    "input": {"prompt": "Write a short story"},
                    "expected": {"min_length": 100}
                }
            ]
        },
        "metrics": {
            "quality": {"weight": 0.7, "type": "llm_judge"},
            "cost": {"weight": 0.3, "type": "cost_normalized"}
        }
    },
    "optimization": {
        "evolution": {
            "generations": 5,
            "population_size": 10
        }
    }
}

# Run optimization
result = await run_optimization(config_dict=config)

# Access results
if result["success"]:
    print(f"Best config: {result['best_config']}")
    print(f"Best score: {result['best_score']}")
else:
    print(f"Error: {result['error']}")
```

### Using YAML Config

```python
from convergence import run_optimization

# Load from YAML file
result = await run_optimization(yaml_path="./optimization.yaml")
```

### Mock Mode (For Testing)

```python
config = {
    "api": {
        "name": "test_api",
        "endpoint": "http://localhost:8000/test",
        "mock_mode": True  # Uses mock responses for testing
    },
    # ... rest of config
}

result = await run_optimization(config_dict=config)
```

## Backend Integration Example

### Context Enrichment MAB Optimization

```python
from convergence import run_optimization

async def optimize_context_enrichment():
    """Optimize MAB parameters for context enrichment."""
    
    config = {
        "api": {
            "name": "context_enrichment",
            "endpoint": "http://backend:8000/api/enrich",
            "mock_mode": False
        },
        "search_space": {
            "parameters": {
                "threshold": {"type": "float", "min": 0.1, "max": 0.5},
                "limit": {"type": "int", "min": 5, "max": 20},
                "content_types": {
                    "type": "categorical",
                    "choices": [
                        ["crystal"],
                        ["crystal", "note"],
                        ["crystal", "note", "conversation"]
                    ]
                }
            }
        },
        "evaluation": {
            "test_cases": {
                "path": "./test_cases/enrichment_tests.json"
            },
            "metrics": {
                "relevance": {"weight": 0.6, "type": "cosine_similarity"},
                "diversity": {"weight": 0.3, "type": "diversity_score"},
                "latency": {"weight": 0.1, "type": "latency_penalty"}
            }
        },
        "optimization": {
            "evolution": {
                "generations": 10,
                "population_size": 20
            }
        }
    }
    
    result = await run_optimization(config_dict=config)
    
    if result["success"]:
        # Save best config to Convex
        await save_to_convex(
            system_name="context_enrichment",
            params=result["best_config"],
            score=result["best_score"]
        )
    
    return result
```

## Response Format

```python
{
    "success": True,
    "best_config": {
        "temperature": 0.7,
        "max_tokens": 1024
    },
    "best_score": 0.87,  # 0.0-1.0
    "configs_generated": 50,
    "configs_saved": 50,
    "optimization_run_id": "my_api_1729375200_a1b2c3d4",
    "generations_run": 5,
    "timestamp": "2025-10-19T12:00:00"
}
```

### Error Response

```python
{
    "success": False,
    "error": "Config validation failed: missing 'api' field",
    "configs_generated": 0,
    "configs_saved": 0
}
```

## Configuration Options

### Minimal Config

```python
config = {
    "api": {
        "name": "my_api",
        "endpoint": "https://api.example.com",
        "mock_mode": True
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

### Full Config (All Options)

See `examples/` directory for complete configuration examples:
- `examples/ai/groq_optimization.yaml` - LLM optimization
- `examples/web_browsing/browserbase_optimization.yaml` - Web browsing
- `examples/heycontext/context_enrichment.yaml` - Context enrichment

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

## Testing

```bash
# Test SDK import
python -c "from convergence import run_optimization; print('✅ SDK import works')"

# Run example optimization
cd examples/ai
python run_groq_optimization.py
```

## Notes

- **Mock Mode**: Use `mock_mode: True` for testing without real API calls
- **Generations**: Start with 3-5 for quick testing, use 10-20 for production
- **Population Size**: 10-20 works for most use cases
- **Metrics**: Define custom evaluators for domain-specific scoring
- **Parallel Workers**: Set to 1 for sequential evaluation (simpler debugging)

## Runtime Loop (Per-Request Bandit)

```python
from convergence import (
    configure_runtime,
    runtime_select,
    runtime_update,
    RuntimeConfigSDK,
)
from my_app.runtime_storage import MyRuntimeStorage

config = RuntimeConfigSDK(
    system="context_enrichment",
    agent_type="chat",
    default_arms=[
        {
            "arm_id": "balanced",
            "name": "Balanced",
            "params": {"threshold": 0.35, "limit": 5},
        }
    ],
)

storage = MyRuntimeStorage()
await configure_runtime("context_enrichment", config=config, storage=storage)

selection = await runtime_select(
    "context_enrichment",
    user_id="user_123",
    context={"conversation_id": "conv_456"},
)

# Use selection.params in your application logic
await runtime_update(
    "context_enrichment",
    user_id="user_123",
    decision_id=selection.decision_id or "",
    reward=0.8,
)
```

Implement `RuntimeStorageProtocol` in your backend to persist arms and decisions:

```python
from convergence.storage.runtime_protocol import RuntimeStorageProtocol

class MyRuntimeStorage(RuntimeStorageProtocol):
    async def get_arms(self, *, user_id: str, agent_type: str):
        ...

    async def initialize_arms(self, *, user_id: str, agent_type: str, arms):
        ...

    async def create_decision(self, *, user_id: str, agent_type: str, arm_pulled: str,
                              strategy_params, arms_snapshot, metadata=None):
        ...

    async def update_performance(self, *, user_id: str, agent_type: str, decision_id: str,
                                 reward: float, engagement=None, grading=None, metadata=None):
        ...

    async def get_decision(self, *, user_id: str, decision_id: str):
        ...
```

See `convergence/storage/runtime_protocol.py` for complete method signatures.

## Advanced Usage

### Custom Evaluator

```python
config = {
    # ...
    "evaluation": {
        "custom_evaluator": {
            "module": "./my_evaluator.py",
            "class": "MyCustomEvaluator"
        }
    }
}
```

### Agent Society (Advanced)

```python
config = {
    # ...
    "society": {
        "enabled": True,
        "learning": {
            "rlp_enabled": True,  # Reasoning
            "sao_enabled": True   # Self-improvement
        },
        "llm": {
            "provider": "litellm",
            "model": "gpt-4",
            "api_key_env": "OPENAI_API_KEY"
        }
    }
}
```

## Support

- **Documentation**: See `README.md` and `examples/`
- **Issues**: https://github.com/persist-os/the-convergence/issues
- **Examples**: Check `examples/` directory for working examples

