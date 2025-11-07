# Discord Agent Optimization Examples

This directory contains examples of optimizing Agno Discord agents using the Convergence framework.

## Two Ways to Run

### 1. YAML-Based (Legacy/CLI)

Uses the traditional YAML configuration approach:

```bash
# Requires config file
convergence optimize discord_agent_optimization.yaml
```

**Files:**
- `discord_agent_optimization.yaml` - Complete YAML configuration
- `discord_evaluator.py` - Custom evaluator module
- `discord_test_cases.json` - Test cases in JSON format
- `discord_agent_runner.py` - Agent runner logic

### 2. Programmatic (SDK)

Uses the new programmatic Python-only interface:

```python
import asyncio
from discord_programmatic_example import discord_optimization_example

asyncio.run(discord_optimization_example())
```

**Files:**
- `discord_programmatic_example.py` - Complete programmatic example
- `discord_evaluator.py` - Same evaluator (shared)
- `discord_test_cases.json` - Same test cases (shared)
- `discord_agent_runner.py` - Same runner (shared)

## Key Differences

### YAML Approach

```yaml
# discord_agent_optimization.yaml
api:
  name: "agno_discord_agent"
  endpoint: "https://..."
search_space:
  parameters:
    temperature:
      type: "categorical"
      values: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
evaluation:
  metrics:
    accuracy:
      weight: 0.40
  custom_evaluator:
    module: "discord_evaluator"
    function: "score_discord_agent_response"
```

### Programmatic Approach

```python
# discord_programmatic_example.py
from convergence.types import ConvergenceConfig, ApiConfig, SearchSpaceConfig

config = ConvergenceConfig(
    api=ApiConfig(name="agno_discord_agent", endpoint="https://..."),
    search_space=SearchSpaceConfig(
        parameters={
            "temperature": {
                "type": "categorical",
                "choices": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            }
        }
    ),
    evaluation=EvaluationConfig(...)
)

result = await run_optimization(
    config=config,
    evaluator=discord_evaluator_function,  # Direct callable
    test_cases=my_test_cases
)
```

## Shared Components

Both approaches use the same components:

- **Evaluator Logic** - `discord_evaluator.py`
- **Test Cases** - `discord_test_cases.json`
- **Agent Runner** - `discord_agent_runner.py`
- **Models & Registry** - Azure OpenAI deployment info

## Benefits of Programmatic Approach

1. **Type Safety** - Full IDE autocomplete and validation
2. **No File I/O** - Everything in memory, faster
3. **Dynamic Config** - Generate configs programmatically
4. **Testability** - Easier to mock and test
5. **Cleaner Code** - No string paths or YAML parsing

## Running the Examples

### Prerequisites

```bash
# Required environment variables
export DISCORD_BOT_TOKEN="your_discord_bot_token"
export AZURE_API_KEY="your_azure_openai_key"

# Optionally for agent society features
export GEMINI_API_KEY="your_gemini_key"
```

### Run YAML Example

```bash
cd /path/to/the-convergence/examples/agno_agents/discord
convergence optimize discord_agent_optimization.yaml
```

### Run Programmatic Example

```bash
cd /path/to/the-convergence/examples/agno_agents/discord
python discord_programmatic_example.py
```

Or from Python:

```python
import asyncio
from examples.agno_agents.discord.discord_programmatic_example import discord_optimization_example

asyncio.run(discord_optimization_example())
```

## Next Steps

1. **Customize Test Cases** - Add your own Discord test scenarios
2. **Tune Metrics** - Adjust weights for your use case
3. **Expand Search Space** - Add more parameters to optimize
4. **Use Results** - Apply best config to your Discord bot

## See Also

- [SDK Usage Guide](../../../SDK_USAGE.md)
- [SDK Migration Guide](../../../SDK_MIGRATION.md)
- [Convergence Documentation](../../../README.md)

