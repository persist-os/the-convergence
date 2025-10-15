# API Adapters Architecture

## Overview

The Convergence uses an **adapter pattern** to support multiple API providers without hardcoding provider-specific logic in the core optimization engine.

**Design Principle**: OpenAI format is the baseline. Adapters are only needed for providers that deviate from this standard.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Optimization Runner                        │
│  (Provider-agnostic optimization logic)                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ├─ No Adapter (OpenAI-compatible)
                     │  └─> OpenAI, Groq, Anthropic, etc.
                     │
                     └─ With Adapter (Provider-specific)
                        ├─> AzureOpenAIAdapter
                        ├─> GeminiAdapter
                        └─> BrowserBaseAdapter
```

## When to Use Adapters

### ✅ Use an Adapter When:
- **Request format differs**: Provider requires nested structures (e.g., Gemini)
- **Response format differs**: Provider returns data in non-standard format
- **Special requirements**: Provider needs preprocessing (e.g., BrowserBase project ID)

### ❌ Don't Need an Adapter When:
- Provider follows OpenAI's request/response format
- Simple parameter differences (handled by YAML config)
- Only authentication differs (handled by auth config)

## Available Adapters

### 1. OpenAIAdapter (Baseline)
**Purpose**: Explicit representation of default behavior  
**When to use**: Not required - this IS the default behavior  
**Format**:
```json
{
  "messages": [...],
  "model": "gpt-4",
  "temperature": 0.7
}
```

### 2. AzureOpenAIAdapter
**Purpose**: Azure-specific endpoints and responses  
**When to use**: When using Azure OpenAI Service  
**Differences from OpenAI**:
- Different endpoint structure (includes deployment name)
- Uses `api-key` header instead of Bearer token
- Otherwise identical to OpenAI

**YAML Configuration**:
```yaml
api:
  name: "azure_o4_mini"  # "azure" triggers adapter
  auth:
    type: "api_key"
    header_name: "api-key"
    token_env: "AZURE_API_KEY"
```

### 3. GeminiAdapter
**Purpose**: Handle Gemini's nested request/response structure  
**When to use**: When using Google Gemini API  
**Differences from OpenAI**:
- Nested request: `contents[].parts[].text` vs flat `messages`
- Configuration in `generationConfig` vs root level
- Response extraction: `candidates[0].content.parts[0].text`

**Request Transformation**:
```python
# Input (optimization params)
{
  "temperature": 0.7,
  "topK": 40,
  "topP": 0.9,
  "maxOutputTokens": 1024
}

# Output (Gemini format)
{
  "contents": [{
    "parts": [{"text": "prompt"}]
  }],
  "generationConfig": {
    "temperature": 0.7,
    "topK": 40,
    "topP": 0.9,
    "maxOutputTokens": 1024
  }
}
```

**YAML Configuration**:
```yaml
api:
  name: "gemini_task_decomposition"  # "gemini" triggers adapter
  auth:
    type: "api_key"
    header_name: "x-goog-api-key"
    token_env: "OPENAI_API_KEY"
```

### 4. BrowserBaseAdapter
**Purpose**: Browser automation session management  
**When to use**: When optimizing BrowserBase sessions  
**Special Requirements**:
- Requires `BROWSERBASE_PROJECT_ID` environment variable
- Transforms viewport/timeout params into browserSettings

**YAML Configuration**:
```yaml
api:
  name: "browserbase"  # "browserbase" triggers adapter
```

## Creating a New Adapter

### Step 1: Create Adapter File

Create `convergence/optimization/adapters/your_provider.py`:

```python
"""Your Provider API adapter."""
from typing import Dict, Any
from . import APIAdapter
from ..models import APIResponse


class YourProviderAdapter(APIAdapter):
    """Adapter for Your Provider's API."""
    
    def transform_request(
        self,
        optimization_params: Dict[str, Any],
        test_case: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Transform to provider's request format."""
        # Your transformation logic here
        request = {}
        # ... build provider-specific structure ...
        return request
    
    def transform_response(
        self,
        api_response: APIResponse,
        optimization_params: Dict[str, Any]
    ) -> APIResponse:
        """Extract data from provider's response."""
        if not api_response.success:
            return api_response
        
        # Extract and normalize response data
        result = api_response.result
        # ... extract relevant data ...
        
        return APIResponse(
            success=True,
            result=extracted_data,
            latency_ms=api_response.latency_ms,
            estimated_cost_usd=api_response.estimated_cost_usd
        )
```

### Step 2: Register in Runner

Edit `convergence/optimization/runner.py`:

```python
# Add import
from .adapters.your_provider import YourProviderAdapter

# Update _detect_adapter method
def _detect_adapter(self, api_name: str):
    # ...
    
    # Your Provider
    if "yourprovider" in api_name_lower:
        return YourProviderAdapter() if YourProviderAdapter else None
    
    # ...
```

### Step 3: Use in Configuration

```yaml
api:
  name: "yourprovider_optimization"  # Naming triggers adapter
  endpoint: "https://api.yourprovider.com/v1/endpoint"
  # ... rest of config ...
```

## Adapter Detection Logic

Adapters are detected based on the `api.name` field in your YAML configuration:

| API Name Contains | Adapter Used | Example |
|-------------------|-------------|---------|
| `azure` | AzureOpenAIAdapter | `azure_o4_mini` |
| `gemini` | GeminiAdapter | `gemini_task_decomposition` |
| `browserbase` | BrowserBaseAdapter | `browserbase_session` |
| `openai` | None (default) | `openai_responses` |
| anything else | None (default) | `groq`, `anthropic`, etc. |

## Best Practices

### 1. **Keep Adapters Focused**
- Only handle request/response transformation
- Don't add business logic or optimization logic
- Keep it simple and testable

### 2. **Follow OpenAI Format When Possible**
- If your provider is close to OpenAI format, don't create an adapter
- Use YAML configuration to handle small differences
- Adapters are for significant format differences

### 3. **Handle Errors Gracefully**
- Check for required environment variables in `__init__`
- Return original response if transformation fails
- Log warnings, don't crash the optimization

### 4. **Document Differences**
- Clearly document what makes your provider different
- Include example request/response formats
- Show YAML configuration examples

## Migration Guide

If you have hardcoded provider logic in your code, migrate it to an adapter:

**Before** (❌ Hardcoded):
```python
# In runner.py
if api_name == "gemini":
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {...}
    }
```

**After** (✅ Adapter):
```python
# In gemini.py adapter
def transform_request(self, optimization_params, test_case):
    return {
        "contents": [{"parts": [{"text": test_case["input"]["prompt"]}]}],
        "generationConfig": optimization_params
    }
```

## Testing Adapters

Test your adapter with a simple script:

```python
from convergence.optimization.adapters.your_provider import YourProviderAdapter
from convergence.optimization.models import APIResponse

adapter = YourProviderAdapter()

# Test request transformation
params = {"temperature": 0.7}
test_case = {"input": {"prompt": "Hello"}}
request = adapter.transform_request(params, test_case)
print(f"Request: {request}")

# Test response transformation  
mock_response = APIResponse(
    success=True,
    result={"your": "response"},
    latency_ms=100.0
)
transformed = adapter.transform_response(mock_response, params)
print(f"Transformed: {transformed.result}")
```

## FAQ

**Q: My provider is almost like OpenAI but with one small difference. Do I need an adapter?**  
A: No! Use YAML configuration to handle small differences. Adapters are for significant structural differences.

**Q: Can I have multiple adapters for the same provider?**  
A: Yes, but use different names in detection logic (e.g., `azure_chat` vs `azure_embeddings`).

**Q: What if my adapter initialization fails?**  
A: The system will log a warning and fall back to default OpenAI behavior.

**Q: How do I test my adapter without running full optimization?**  
A: Create a simple Python script that instantiates your adapter and calls its methods with mock data.

## Support

For questions or issues with adapters:
1. Check existing adapters for examples
2. Review this documentation
3. Open an issue with your specific use case

