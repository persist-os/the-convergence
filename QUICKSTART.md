# Convergence SDK - Quick Start

## Install (30 seconds)

```bash
# Development mode (recommended)
cd /path/to/the-convergence
pip install -e .

# Verify
python -c "from convergence import run_optimization; print('✅ Ready!')"
```

## Use (3 lines of code)

```python
from convergence import run_optimization

result = await run_optimization(config_dict={...})

print(result["best_config"])  # Your optimized parameters
```

## Minimal Example

```python
from convergence import run_optimization

config = {
    "api": {
        "name": "my_system",
        "endpoint": "http://localhost:8000/api",
        "mock_mode": True  # Use mock for testing
    },
    "search_space": {
        "parameters": {
            "threshold": {"type": "float", "min": 0.1, "max": 0.5},
            "limit": {"type": "int", "min": 5, "max": 20}
        }
    },
    "evaluation": {
        "test_cases": {
            "inline": [{"id": "test_1", "input": {}}]
        },
        "metrics": {"accuracy": {"weight": 1.0}}
    }
}

result = await run_optimization(config_dict=config)

if result["success"]:
    print(f"✅ Best: {result['best_config']}")
    print(f"   Score: {result['best_score']:.3f}")
else:
    print(f"❌ Error: {result['error']}")
```

## Backend Integration

```python
# In your executor
from convergence import run_optimization

async def optimize_system(payload):
    config = {
        "api": {"name": payload["system_name"], ...},
        "search_space": {"parameters": payload["parameters"]},
        "evaluation": {"test_cases": {...}, "metrics": {...}}
    }
    
    result = await run_optimization(config_dict=config)
    
    if result["success"]:
        # Save to Convex
        await save_config(
            system_name=payload["system_name"],
            params=result["best_config"],
            score=result["best_score"]
        )
    
    return result
```

## Test

```bash
cd backend
python test_convergence_sdk.py
```

## Important

- **Package name**: `the-convergence` (with hyphens)
- **Import name**: `convergence` (no hyphens)
- **Mock mode**: Set `mock_mode: True` for testing
- **Generations**: Start with 2-3 for quick tests, 10-20 for real optimization

## Full Documentation

- Installation: `INSTALLATION.md`
- SDK Usage: `SDK_USAGE.md`
- Examples: `examples/` directory
- Complete docs: `README.md`

## Support

Issues/Questions: https://github.com/persist-os/the-convergence/issues

