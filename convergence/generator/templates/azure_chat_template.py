"""
Azure Chat API Template

Based on proven patterns from examples/ai/azure/azure_multi_model_optimization.yaml
Handles Azure's multi-endpoint architecture with model registry support.
"""
from typing import Dict, List, Any
import yaml
import json


class AzureChatTemplate:
    """Template for Azure OpenAI chat APIs with multi-endpoint support."""
    
    def generate_config(self, models_with_endpoints: Dict[str, str], api_key_env: str, description: str) -> Dict[str, Any]:
        """Generate Azure OpenAI configuration with model registry.
        
        Args:
            models_with_endpoints: Dict mapping model names to full endpoint URLs
                e.g., {"gpt-4": "https://resource.openai.azure.com/...", "o4-mini": "https://..."}
            api_key_env: Environment variable name for Azure API key
            description: Description of the API functionality
        """
        model_names = list(models_with_endpoints.keys())
        
        return {
            'api': {
                'name': 'azure_multi_model',
                'description': f'Azure OpenAI with multiple models - {description}',
                
                # Single API key for all models in the same resource
                'auth': {
                    'type': 'api_key',
                    'token_env': api_key_env,
                    'header_name': 'api-key'
                },
                
                # Model registry - each model has its own endpoint
                'models': {
                    model: {'endpoint': endpoint}
                    for model, endpoint in models_with_endpoints.items()
                },
                
                'request': {
                    'method': 'POST',
                    'headers': {'Content-Type': 'application/json'},
                    'timeout_seconds': 120
                }
            },
            'search_space': {
                'parameters': {
                    # Model selection - references model registry keys
                    'model': {
                        'type': 'categorical',
                        'values': model_names,
                        'description': 'Model key from api.models registry'
                    },
                    
                    # Temperature - affects response creativity
                    'temperature': {
                        'type': 'categorical',
                        'values': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        'description': 'Sampling temperature'
                    },
                    
                    # Max completion tokens
                    'max_completion_tokens': {
                        'type': 'discrete',
                        'values': [500, 1000, 2000, 4000],
                        'description': 'Maximum tokens in response'
                    }
                }
            },
            'evaluation': {
                'test_cases': {'path': 'test_cases.json'},
                'custom_evaluator': {
                    'enabled': True,
                    'module': 'evaluator',
                    'function': 'score_azure_response'
                },
                'metrics': {
                    'response_quality': {'weight': 0.4, 'type': 'higher_is_better', 'function': 'custom'},
                    'response_length': {'weight': 0.2, 'type': 'higher_is_better', 'function': 'custom'},
                    'latency_sec': {'weight': 0.2, 'type': 'lower_is_better', 'function': 'custom'},
                    'cost_per_task': {'weight': 0.2, 'type': 'lower_is_better', 'function': 'custom'}
                }
            },
            'optimization': {
                'algorithm': 'mab_evolution',
                'mab': {'strategy': 'thompson_sampling', 'exploration_rate': 0.2},
                'evolution': {
                    'generations': 3,
                    'population_size': 3,
                    'mutation_rate': 0.25,
                    'crossover_rate': 0.7,
                    'elite_size': 1
                },
                'execution': {
                    'experiments_per_generation': 2,
                    'parallel_workers': 1,
                    'max_retries': 3,
                    'early_stopping': {
                        'enabled': True,
                        'patience': 2,
                        'min_improvement': 0.0005
                    }
                }
            },
            'output': {
                'save_path': './results/azure_multi_model_optimization',
                'save_all_experiments': True,
                'formats': ['json', 'markdown', 'csv'],
                'visualizations': ['score_over_time', 'parameter_importance'],
                'export_best_config': {
                    'enabled': True,
                    'format': 'python',
                    'output_path': './best_config.py'
                }
            },
            'legacy': {
                'enabled': True,
                'sqlite_path': './data/legacy.db',
                'export_dir': './legacy_exports'
            }
        }
    
    def generate_test_cases(self, description: str) -> List[Dict]:
        """Generate Azure-specific test cases."""
        # Use existing augmentation system from LLMChatTemplate
        try:
            from .llm_chat_template import LLMChatTemplate
            llm_template = LLMChatTemplate()
            return llm_template.generate_test_cases(description)
        except ImportError:
            # Fallback test cases
            return [
                {
                    "id": "azure_reasoning_test",
                    "description": "Azure reasoning capability test",
                    "input": {"messages": [{"role": "user", "content": "Explain the difference between Azure OpenAI and OpenAI API"}]},
                    "expected": {"contains": ["Azure", "OpenAI"], "min_length": 50, "min_quality_score": 0.8},
                    "metadata": {"category": "reasoning", "difficulty": "medium", "weight": 1.5}
                },
                {
                    "id": "azure_creative_test",
                    "description": "Azure creative writing test",
                    "input": {"messages": [{"role": "user", "content": "Write a short story about AI and cloud computing"}]},
                    "expected": {"min_length": 100, "min_quality_score": 0.7},
                    "metadata": {"category": "creative", "difficulty": "medium", "weight": 1.2}
                }
            ]
    
    def generate_evaluator(self) -> str:
        """Generate Azure-specific evaluator."""
        return '''"""
Azure OpenAI API Evaluator

Generated by Convergence Azure Chat Template

This evaluator scores Azure OpenAI responses based on proven patterns from Azure examples.
"""
import json
from typing import Dict, Any, Optional


def score_azure_response(
    result: Any,
    expected: Dict[str, Any],
    params: Dict[str, Any],
    metric: Optional[str] = None
) -> float:
    """
    Score Azure OpenAI response based on proven patterns.
    
    Args:
        result: Azure OpenAI API response
        expected: Expected criteria
        params: Configuration parameters
        metric: Specific metric to return
    
    Returns:
        Score between 0.0 and 1.0
    """
    # Extract text from Azure OpenAI response
    text = _extract_azure_text(result)
    if not text:
        return 0.0
    
    # Calculate scores
    scores = {}
    scores['response_quality'] = _score_response_quality(text, expected)
    scores['response_length'] = _score_response_length(text, expected)
    scores['latency_sec'] = _score_latency(result, expected)
    scores['cost_per_task'] = _score_cost(result, expected)
    
    # Return specific metric if requested
    if metric:
        return scores.get(metric, 0.0)
    
    # Weighted overall score
    overall_score = (
        scores['response_quality'] * 0.4 +
        scores['response_length'] * 0.2 +
        scores['latency_sec'] * 0.2 +
        scores['cost_per_task'] * 0.2
    )
    
    return min(1.0, max(0.0, overall_score))


def _extract_azure_text(result):
    """Extract text from Azure OpenAI response."""
    if isinstance(result, dict):
        if 'choices' in result and len(result['choices']) > 0:
            choice = result['choices'][0]
            if 'message' in choice and 'content' in choice['message']:
                return choice['message']['content']
        # Fallback to direct text fields
        for field in ['text', 'content', 'output', 'response']:
            if field in result:
                return str(result[field])
    elif isinstance(result, str):
        return result
    return ""


def _score_response_quality(text, expected):
    """Score based on response quality."""
    if not text or len(text.strip()) < 3:
        return 0.0
    
    score = 0.5  # Base score
    
    # Check for required keywords
    if 'contains' in expected:
        required_keywords = expected['contains']
        if required_keywords:
            text_lower = text.lower()
            found_keywords = sum(1 for keyword in required_keywords if keyword.lower() in text_lower)
            score += (found_keywords / len(required_keywords)) * 0.3
    
    # Length appropriateness
    if 10 <= len(text) <= 1000:
        score += 0.2
    
    return min(1.0, score)


def _score_response_length(text, expected):
    """Score based on response length."""
    length = len(text)
    min_length = expected.get('min_length', 0)
    
    if length < min_length:
        return 0.0
    
    # Gradual scoring based on length
    if length >= min_length * 2:
        return 1.0
    elif length >= min_length * 1.5:
        return 0.8
    elif length >= min_length:
        return 0.6
    else:
        return 0.3


def _score_latency(result, expected):
    """Score based on response latency."""
    if isinstance(result, dict) and 'latency_ms' in result:
        latency_ms = result['latency_ms']
        max_latency = expected.get('max_latency_ms', 5000)
        
        if latency_ms <= max_latency:
            return 1.0
        else:
            penalty = min(0.8, (latency_ms - max_latency) / max_latency)
            return max(0.0, 1.0 - penalty)
    
    return 0.5  # Default score if no latency data


def _score_cost(result, expected):
    """Score based on cost efficiency."""
    if isinstance(result, dict) and 'cost_usd' in result:
        cost = result['cost_usd']
        max_cost = expected.get('max_cost_usd', 0.01)
        
        if cost <= max_cost:
            return 1.0
        else:
            penalty = min(0.8, (cost - max_cost) / max_cost)
            return max(0.0, 1.0 - penalty)
    
    return 0.5  # Default score if no cost data
'''
    
    def generate_yaml_content(self, config: Dict[str, Any]) -> str:
        """Generate YAML content from config."""
        model_list = ', '.join(config['api']['models'].keys())
        
        yaml_content = f"""# Azure OpenAI Multi-Model Optimization Configuration
# Generated by Convergence Azure Chat Template
# 
# API: {config['api']['name']}
# Description: {config['api']['description']}
#
# Required Environment Variables:
#   {config['api']['auth']['token_env']} - Azure OpenAI API key
#
# Setup:
#   1. Deploy models to Azure AI Foundry
#   2. export {config['api']['auth']['token_env']}="your_azure_key"
#   3. Configure models in api.models registry below
#   4. Select active model(s) in search_space.parameters.model.values
#   5. convergence optimize optimization.yaml
#
# Model Registry:
#   Models configured: {model_list}
#   Each model has its own endpoint URL

"""
        yaml_content += yaml.dump(config, default_flow_style=False, sort_keys=False)
        return yaml_content
    
    def generate_json_content(self, test_cases: List[Dict]) -> str:
        """Generate JSON content from test cases."""
        return json.dumps({"test_cases": test_cases}, indent=2)
    
    def generate_readme_content(self, config: Dict[str, Any]) -> str:
        """Generate README content."""
        model_list = ', '.join(config['api']['models'].keys())
        
        return f"""# Azure OpenAI Multi-Model Optimization

This configuration optimizes Azure OpenAI API calls using **{config['api']['description']}**.

## Setup

### 1. Set Azure OpenAI API Key

```bash
export {config['api']['auth']['token_env']}='your-azure-openai-api-key'
```

**Important:** This is your Azure OpenAI API key from the Azure portal, not an OpenAI key.

### 2. Configure Model Registry

The configuration includes these models: {model_list}

Each model in the `api.models` registry has its own endpoint URL. Update these URLs to match your Azure deployments:

```yaml
models:
  gpt-4:
    endpoint: "https://your-resource.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2025-01-01-preview"
  o4-mini:
    endpoint: "https://your-resource.openai.azure.com/openai/deployments/o4-mini/chat/completions?api-version=2025-01-01-preview"
```

### 3. Select Active Models

In `search_space.parameters.model.values`, choose which models to test:

```yaml
model:
  type: "categorical"
  values:
    - "gpt-4"        # Primary model
    - "o4-mini"       # Fast reasoning model
    # - "gpt-4o-mini"  # Cost-effective model
```

### 4. Run Optimization

```bash
convergence optimize optimization.yaml
```

## What's Being Optimized

- **model**: {', '.join(config['search_space']['parameters']['model']['values'])}
- **temperature**: {', '.join(map(str, config['search_space']['parameters']['temperature']['values']))}
- **max_completion_tokens**: {', '.join(map(str, config['search_space']['parameters']['max_completion_tokens']['values']))}

## Test Cases

The configuration includes test cases for:
- Azure reasoning capabilities
- Creative writing tasks
- Multi-model comparison

## Metrics

- **Response Quality** ({config['evaluation']['metrics']['response_quality']['weight']*100:.0f}%): Content quality and completeness
- **Response Length** ({config['evaluation']['metrics']['response_length']['weight']*100:.0f}%): Appropriate response length
- **Latency** ({config['evaluation']['metrics']['latency_sec']['weight']*100:.0f}%): Response time performance
- **Cost** ({config['evaluation']['metrics']['cost_per_task']['weight']*100:.0f}%): Cost efficiency per task

## Results

Results will be saved to `{config['output']['save_path']}/`

- `best_config.py`: Best configuration found
- `report.md`: Detailed optimization report
- `detailed_results.json`: All experiment results
"""


# Export for easy import
__all__ = ['AzureChatTemplate']
