# New Agent Creation Guide

**Purpose:** Quick guide to create new agent implementations using the `BaseAgentRunner` abstraction.

**Time Required:** ~30 minutes for basic agent  
Complexity:** Minimal - only 5 methods to implement

## Overview

The agent abstraction layer allows you to create new agent implementations with minimal code. You only need to implement 5 service-specific methods; all common logic (Azure integration, response parsing, test execution, etc.) is handled by `BaseAgentRunner`.

## Architecture

```
BaseAgentRunner (441 lines - common logic)
    ↓
YourAgentRunner (~150 lines - service-specific logic)
```

**What you implement:**
- Service configuration access
- Credential validation  
- Tool initialization
- Tool selection logic
- Query building

**What you inherit:**
- Azure OpenAI model setup
- Agent creation and execution
- Response parsing
- Token tracking
- Error handling
- Test orchestration

## Step-by-Step Guide

### Step 1: Create Agent Runner File

Create `your_service/your_agent_runner.py`:

```python
"""
YourService Agent Runner with Azure OpenAI Integration (Agno 2.1.8+)

Refactored to extend BaseAgentRunner (agent abstraction):
- Inherits common agent execution logic from base class
- Implements only YourService-specific tool initialization, credential validation, and query building
- Reduces code from ~500 lines to ~150 lines (70% reduction)
"""

import os
import sys
import logging
from typing import Dict, Any, List

from agno.tools.yourservice import YourServiceTools
AGNO_AVAILABLE = True

# Import base class from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from base_agent_runner import BaseAgentRunner

logger = logging.getLogger(__name__)


class YourServiceAgentRunner(BaseAgentRunner):
    """
    YourService agent runner - inherits common logic from BaseAgentRunner.
    
    Implements only YourService-specific functionality:
    - YourService credential validation
    - YourServiceTools initialization
    - YourService-specific tool selection
    - YourService-specific query building
    
    All other logic (model creation, test execution, response parsing) inherited from base class.
    """
```

### Step 2: Define Instruction Styles

```python
INSTRUCTION_STYLES = {
    'minimal': [
        "You are a YourService assistant.",
        "Use YourService tools to complete tasks accurately."
    ],
    'detailed': [
        "You are a specialized YourService assistant with access to the API.",
        "When asked to interact with YourService:",
        "1. Choose the appropriate tool for the task",
        "2. Use correct parameters",
        "3. Extract and present relevant information from results",
        "4. Ensure data completeness",
        "5. Provide accurate, factual responses",
        "6. If multiple tools are needed, execute them in sequence"
    ],
    'structured': [
        "You are a YourService data assistant with structured data requirements.",
        "For each query, you must:",
        "- Identify the correct tool to use",
        "- Call the tool with precise parameters",
        "- Return results in a structured format with all available fields",
        "- Format output as clear, complete JSON-like data structures"
    ]
}
```

### Step 3: Implement Required Methods

#### Method 1: `_get_service_config()`

Return your service-specific configuration section.

```python
def _get_service_config(self) -> Dict[str, Any]:
    """Get YourService-specific configuration from agent_config."""
    return self.agent_config.get('yourservice_auth', {})
```

#### Method 2: `_validate_credentials()`

Validate that required credentials are available.

```python
def _validate_credentials(self) -> None:
    """Validate that YourService API credentials are available."""
    api_key = os.getenv(self.service_config.get('api_key_env', 'YOURSERVICE_API_KEY'))
    
    if not api_key:
        raise ValueError(
            "YourService API key not found. Set environment variable:\n"
            f"  export {self.service_config.get('api_key_env', 'YOURSERVICE_API_KEY')}='your_api_key'\n"
            "Get your API key from: https://yourservice.com/developers"
        )
```

#### Method 3: `_initialize_tools()`

Initialize and return your service's tools.

```python
def _initialize_tools(self) -> YourServiceTools:
    """Initialize YourServiceTools with credentials."""
    api_key = os.getenv(self.service_config.get('api_key_env', 'YOURSERVICE_API_KEY'))
    
    # Optional: Add additional config parameters
    extra_config = {}
    if self.service_config.get('api_url'):
        extra_config['api_url'] = self.service_config['api_url']
    
    # Initialize tools
    return YourServiceTools(api_key=api_key, **extra_config)
```

#### Method 4: `_select_tools()`

Select which tools to use based on strategy and test case.

```python
def _select_tools(self, tools: YourServiceTools, strategy: str, test_case: Dict[str, Any] = None) -> List:
    """Select tools based on strategy and test case restrictions."""
    # Check for tool restrictions in test case metadata
    if test_case and 'metadata' in test_case:
        tool_restriction = test_case['metadata'].get('tool_restriction')
        
        if tool_restriction == 'read_only':
            # Only read operations
            logger.info("🔧 Tool restriction: read-only operations")
            return [tools.get_data, tools.list_items]
        
        elif tool_restriction == 'all_tools_available':
            # All tools available - agent must choose
            logger.info("🔧 Tool restriction: all tools available - agent chooses")
            return [tools]
    
    # Fallback to strategy-based selection
    if strategy == 'include_all':
        return [tools]
    elif strategy == 'include_specific':
        # Return specific tools if needed
        return [tools.get_data, tools.create_item]
    else:
        return [tools]
```

#### Method 5: `_build_query()`

Build the query string from test case input.

```python
def _build_query(self, test_case: Dict[str, Any]) -> str:
    """Build query string from test case input."""
    input_data = test_case.get('input', {})
    query = input_data.get('query', '')
    
    # Append additional context fields
    other_fields = []
    for key, value in input_data.items():
        if key not in ['query', 'task']:
            other_fields.append(f"{key}: {value}")
    
    if other_fields:
        query += "\n\nContext:\n" + "\n".join(other_fields)
    
    return query
```

## Complete Example

See the following files for complete implementations:
- `discord/discord_agent_runner.py` (141 lines)
- `gmail/gmail_agent_runner.py` (249 lines)
- `reddit/reddit_agent_runner.py` (231 lines)

## Configuration File

Create `your_service/your_service_agent_optimization.yaml`:

```yaml
api:
  name: "agno_yourservice_agent"
  description: "Agno agent with YourService toolkit via Azure OpenAI"
  endpoint: "https://placeholder"

agent:
  # YourService API authentication
  yourservice_auth:
    api_key_env: "YOURSERVICE_API_KEY"
    # Add other service-specific config here
  
  # Model registry
  models:
    gpt-4:
      endpoint: "https://your-resource.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2025-01-01-preview"
      api_key_env: "AZURE_API_KEY"
      description: "GPT-4 for high-quality responses"

# Search space: Agent parameters to optimize
search_space:
  parameters:
    model:
      type: "categorical"
      values: ["gpt-4"]
    
    temperature:
      type: "categorical"
      values: [0.2, 0.6, 0.8]
    
    max_completion_tokens:
      type: "discrete"
      values: [1000, 2000]
    
    instruction_style:
      type: "categorical"
      values: ["detailed", "structured"]
    
    tool_strategy:
      type: "categorical"
      values: ["include_all", "include_specific"]
```

## Test Cases File

Create `your_service/your_service_test_cases.json`:

```json
{
  "test_cases": [
    {
      "id": "basic_operation",
      "function": "get_data",
      "input": {
        "query": "Get information about resource X",
        "resource_id": "x123"
      },
      "expected": {
        "tools_called": ["get_data"],
        "result_validation": {
          "required_fields": ["id", "name", "status"],
          "min_result_count": 1
        }
      }
    }
  ]
}
```

## Verification Checklist

- [ ] Created `your_service/your_agent_runner.py` with all 5 methods
- [ ] Inherits from `BaseAgentRunner`
- [ ] Implements `_get_service_config()`
- [ ] Implements `_validate_credentials()`
- [ ] Implements `_initialize_tools()`
- [ ] Implements `_select_tools()`
- [ ] Implements `_build_query()`
- [ ] Defined `INSTRUCTION_STYLES` dict
- [ ] Added test helper function
- [ ] Created configuration YAML file
- [ ] Created test cases JSON file
- [ ] Tested with real API calls
- [ ] Verified no linter errors

## Common Patterns

### Pattern 1: Simple API Key Authentication

```python
def _validate_credentials(self) -> None:
    api_key = os.getenv(self.service_config.get('api_key_env', 'SERVICE_API_KEY'))
    if not api_key:
        raise ValueError("API key not found")

def _initialize_tools(self) -> YourServiceTools:
    api_key = os.getenv(self.service_config.get('api_key_env', 'SERVICE_API_KEY'))
    return YourServiceTools(api_key=api_key)
```

### Pattern 2: Client ID/Secret Authentication

```python
def _validate_credentials(self) -> None:
    client_id = os.getenv(self.service_config.get('client_id_env', 'CLIENT_ID'))
    client_secret = os.getenv(self.service_config.get('client_secret_env', 'CLIENT_SECRET'))
    if not client_id or not client_secret:
        raise ValueError("Credentials not found")

def _initialize_tools(self) -> YourServiceTools:
    client_id = os.getenv(self.service_config.get('client_id_env', 'CLIENT_ID'))
    client_secret = os.getenv(self.service_config.get('client_secret_env', 'CLIENT_SECRET'))
    return YourServiceTools(client_id=client_id, client_secret=client_secret)
```

### Pattern 3: Token File Authentication

```python
def _initialize_tools(self) -> YourServiceTools:
    from pathlib import Path
    
    token_path = self.service_config.get('token_path', 'token.json')
    if not Path(token_path).is_absolute():
        service_dir = Path(__file__).parent
        token_path = str(service_dir / token_path)
    
    return YourServiceTools(token_path=token_path)
```

## Tips & Best Practices

1. **Keep it simple:** Only implement service-specific logic
2. **Follow patterns:** Look at existing agents for patterns
3. **Test early:** Use the test helper function to verify credentials
4. **Handle errors:** Provide clear error messages with setup instructions
5. **Log appropriately:** Use logger.info() for important events
6. **Extract configuration:** Use env vars and config files, not hardcoded values
7. **Support restrictions:** Allow test cases to restrict tool usage
8. **Add context:** Build queries with all available input fields

## Getting Help

- Check `PHASE_2_COMPLETION_SUMMARY.md` for architecture details
- See `base_agent_runner.py` for common method implementations
- Reference `discord_agent_runner.py` for simplest example
- Refer to `gmail_agent_runner.py` for complex auth example
- Look at `reddit_agent_runner.py` for tool selection patterns

## Expected Results

After completing this guide, you should have:
- A new agent runner (~150 lines)
- Working configuration file
- Test cases for validation
- Tool integration with your service
- Full Azure OpenAI integration through BaseAgentRunner
- Consistent behavior with other agents

**Time Investment:** 30 minutes to 2 hours (depending on service complexity)

