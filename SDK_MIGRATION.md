# Convergence SDK Migration to Programmatic-Only Interface

## Overview

The Convergence SDK has been refactored to provide a fully programmatic interface without YAML configuration files.

## Key Changes

### Before (YAML-Based)

```python
from convergence.sdk import run_optimization

# Had to create YAML files
result = await run_optimization(
    config_dict={...}  # Or yaml_path="config.yaml"
)
```

### After (Programmatic-Only)

```python
from convergence.types import (
    ConvergenceConfig,
    ApiConfig,
    SearchSpaceConfig,
    RunnerConfig,
    EvaluationConfig,
)
from convergence.sdk import run_optimization

# Define everything as Python objects
config = ConvergenceConfig(
    api=ApiConfig(name="my_api", endpoint="http://localhost:8000"),
    search_space=SearchSpaceConfig(parameters={...}),
    runner=RunnerConfig(generations=10, population=20),
    evaluation=EvaluationConfig(required_metrics=["score"], weights={"score": 1.0})
)

result = await run_optimization(
    config=config,
    evaluator=my_evaluator_function,
    test_cases=[{"input": {...}, "expected": {...}}],
    logging_mode="summary"
)
```

## What Was Removed

- `config_dict` parameter from `run_optimization()`
- `yaml_path` parameter from `run_optimization()`
- `load_config_from_file()` function
- Temporary YAML file creation
- All YAML dependencies from the SDK layer

## What Was Added

### New Type System (`convergence.types`)

- `ConvergenceConfig`: Top-level configuration
- `ApiConfig`: API endpoint configuration
- `SearchSpaceConfig`: Search space parameters
- `RunnerConfig`: Runner settings (generations, population, etc.)
- `EvaluationConfig`: Evaluation metrics and requirements
- `AdaptersConfig`: Data transformation adapters
- `OptimizationRunResult`: Structured result type
- `Evaluator`: Protocol for evaluation functions

### New SDK Features

- `resolve_callable()`: Resolve callables from strings or direct functions
- `normalize_test_cases()`: Normalize test cases into consistent format
- `TestCase`: Structured test case class
- `run_optimization_sync()`: Synchronous wrapper for async function

## Migration Guide

### Step 1: Update Imports

```python
# Old
from convergence.sdk import run_optimization, load_config_from_file

# New
from convergence.types import ConvergenceConfig, ApiConfig, ...
from convergence.sdk import run_optimization
```

### Step 2: Convert YAML to Python Config

```python
# Old: config.yaml
# api:
#   name: "test"
#   endpoint: "http://localhost:8000"

# New: config.py
config = ConvergenceConfig(
    api=ApiConfig(name="test", endpoint="http://localhost:8000"),
    # ... other configs
)
```

### Step 3: Update Test Cases

```python
# Old: test_cases.json
# [{"input": {...}, "expected": {...}}]

# New: Python list or generator
test_cases = [
    {"input": {...}, "expected": {...}},
    # ...
]

# Or from a function
def generate_test_cases():
    yield {"input": {...}, "expected": {...}}
```

### Step 4: Update Evaluator

```python
# Old: custom_evaluator.py with function reference
# evaluation:
#   custom_evaluator:
#     module: "my_evaluator"
#     function: "score_response"

# New: Pass function directly
def evaluator(prediction, expected, *, context=None):
    return {"score": 0.95}

result = await run_optimization(
    config=config,
    evaluator=evaluator,  # Direct function
    test_cases=test_cases
)
```

### Step 5: Update Adapters (If Using)

```python
from convergence.types import AdaptersConfig

def input_adapter(payload, *, context=None):
    return transformed_payload

def output_adapter(response, *, context=None):
    return transformed_response

adapters = AdaptersConfig(
    input_adapter=input_adapter,
    output_adapter=output_adapter
)

result = await run_optimization(
    config=config,
    evaluator=evaluator,
    test_cases=test_cases,
    adapters=adapters
)
```

## Example

See `examples/simple_programmatic_example.py` for a complete working example.

## Benefits

1. **Type Safety**: Pydantic models validate configuration at runtime
2. **No File I/O**: Everything in memory, no temporary files
3. **IDE Support**: Full autocomplete and type hints
4. **Testability**: Easier to mock and test configurations
5. **Dynamic Config**: Generate configs programmatically
6. **Cleaner Code**: No string-based paths or YAML parsing

## Backward Compatibility

- CLI still supports YAML files (uses `ConfigLoader` directly)
- `ConfigLoader` remains available for legacy code
- Internal optimization engine unchanged
- Only the public SDK interface changed

## Questions?

- See `examples/simple_programmatic_example.py` for examples
- Check `convergence/types/` for all available types
- Read SDK documentation in `convergence/sdk.py`

