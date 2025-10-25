"""
Agno Agent API Template

Based on proven patterns from examples/agno_agents/reddit/
"""
from typing import Dict, List, Any
import yaml
import json


class AgnoAgentTemplate:
    """Template for Agno agent APIs."""
    
    def generate_config(self, endpoint: str, api_key_env: str, description: str, provider_name: str = None, models: List[str] = None) -> Dict[str, Any]:
        """Generate Agno agent API configuration.
        
        Args:
            endpoint: API endpoint URL (for LLM provider, e.g., Azure OpenAI)
            api_key_env: Environment variable name for LLM provider API key
            description: Description of the API functionality
            provider_name: LLM provider name (e.g., 'azure', 'openai')
            models: List of model deployment names
        """
        # Default to Azure OpenAI structure if not specified
        if not provider_name:
            provider_name = 'azure'
        if not models:
            models = ['gpt-4-1']
        
        # Use provider-specific endpoint
        if not endpoint or endpoint == 'https://api.example.com/v1/agent':
            endpoint = 'https://YOUR_RESOURCE.openai.azure.com' if provider_name == 'azure' else 'https://api.openai.com/v1'
        
        return {
            'api': {
                'name': 'custom_agno_agent',
                'description': f'Agno agent with custom tools via {provider_name}',
                'endpoint': 'https://placeholder-see-agent-models-registry',  # Managed by model registry
                'adapter_enabled': True,  # Enable agent adapter
                'request': {
                    'method': 'POST',
                    'headers': {'Content-Type': 'application/json'},
                    'timeout_seconds': 120
                },
                'auth': {
                    'type': 'api_key',
                    'token_env': api_key_env,
                    'header_name': 'api-key' if provider_name == 'azure' else 'Authorization'
                }
            },
            'agent': {
                # Tool-specific authentication (customize based on your tools)
                'tool_auth': {
                    'example_tool_key_env': 'TOOL_API_KEY',
                    'user_agent': 'the-convergence-agent-tester/1.0'
                },
                # Model Registry: Define all available LLM models
                'models': self._generate_model_registry(models, endpoint, api_key_env, provider_name)
            },
            'search_space': {
                'parameters': {
                    'model': {
                        'type': 'categorical',
                        'values': models,  # Reference model registry keys
                        'description': 'Model key from agent.models registry'
                    },
                    'temperature': {
                        'type': 'categorical',
                        'values': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        'description': 'Temperature: 0.0=deterministic, 1.0=creative'
                    },
                    'max_completion_tokens': {
                        'type': 'discrete',
                        'values': [500, 1000, 2000, 4000],
                        'description': 'Max tokens for agent response'
                    },
                    'instruction_style': {
                        'type': 'categorical',
                        'values': ['minimal', 'detailed', 'structured'],
                        'description': 'Agent instruction/prompting style'
                    },
                    'tool_strategy': {
                        'type': 'categorical',
                        'values': ['include_specific', 'include_all'],
                        'description': 'Which tools to include'
                    }
                }
            },
            'evaluation': {
                'test_cases': {'path': 'test_cases.json'},
                'metrics': {
                    'task_success': {'weight': 0.35, 'type': 'higher_is_better', 'function': 'custom'},
                    'tool_efficiency': {'weight': 0.25, 'type': 'higher_is_better', 'function': 'custom'},
                    'reasoning_quality': {'weight': 0.20, 'type': 'higher_is_better', 'function': 'custom'},
                    'response_quality': {'weight': 0.20, 'type': 'higher_is_better', 'function': 'custom'}
                }
            },
            'optimization': {
                'algorithm': 'mab_evolution',
                'mab': {'strategy': 'thompson_sampling', 'exploration_rate': 0.3},
                'evolution': {
                    'population_size': 3,
                    'generations': 2,
                    'mutation_rate': 0.3,
                    'crossover_rate': 0.5,
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
                'save_path': './results/custom_agno_agent_optimization',
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
    
    def _generate_model_registry(self, models: List[str], endpoint: str, api_key_env: str, provider_name: str) -> Dict[str, Any]:
        """Generate model registry for agent configuration.
        
        Args:
            models: List of model deployment names
            endpoint: LLM provider endpoint
            api_key_env: Environment variable for LLM API key (not used per model)
            provider_name: Provider name (e.g., 'azure', 'openai')
        
        Returns:
            Dictionary mapping model names to their configurations
        """
        model_registry = {}
        
        for model in models:
            if provider_name == 'azure':
                # Generate full endpoint URL for Azure
                full_endpoint = f"{endpoint}/openai/deployments/{model}/chat/completions?api-version=2025-01-01-preview"
                model_registry[model] = {
                    'endpoint': full_endpoint,
                    'description': f'{model} deployment'
                }
            else:
                # OpenAI or other providers
                model_registry[model] = {
                    'endpoint': endpoint,
                    'description': f'{model} model'
                }
        
        return model_registry
    
    def generate_test_cases(self, description: str) -> List[Dict]:
        """Generate agent test cases based on Reddit example."""
        base_tests = [
            {
                "id": "simple_research_task",
                "description": "Basic agent research task",
                "input": {"task": "Search for recent AI news and summarize the top 3 influential developments"},
                "expected": {
                    "task_completed": True,
                    "tools_used": ["search", "summarization"],
                    "min_quality_score": 0.8,
                    "min_response_length": 100
                },
                "metadata": {"category": "research", "difficulty": "medium", "weight": 1.5}
            },
            {
                "id": "multi_step_analysis",
                "description": "Complex multi-step analysis",
                "input": {"task": "Analyze sentiment of recent discussions about AI safety and provide recommendations"},
                "expected": {
                    "task_completed": True,
                    "tools_used": ["search", "sentiment_analysis"],
                    "min_quality_score": 0.75,
                    "min_response_length": 150
                },
                "metadata": {"category": "analysis", "difficulty": "hard", "weight": 2.0}
            },
            {
                "id": "data_extraction",
                "description": "Data extraction and processing",
                "input": {"task": "Extract key metrics from the latest AI research papers and create a summary"},
                "expected": {
                    "task_completed": True,
                    "tools_used": ["search", "data_extraction"],
                    "min_quality_score": 0.7,
                    "min_response_length": 120
                },
                "metadata": {"category": "data_processing", "difficulty": "medium", "weight": 1.5}
            }
        ]
        
        # Use existing augmentation system
        try:
            from convergence.optimization.test_case_evolution import TestCaseEvolutionEngine
            engine = TestCaseEvolutionEngine(
                mutation_rate=0.3,
                crossover_rate=0.2,
                augmentation_factor=1,
                preserve_originals=True
            )
            return engine.augment_test_cases(base_tests)
        except ImportError:
            # Fallback if augmentation not available
            return base_tests
    
    def generate_evaluator(self) -> str:
        """Generate evaluator based on Reddit agent example."""
        return '''"""
Custom Agno Agent API Evaluator
Generated by Convergence Custom Template Generator

This evaluator scores agent responses based on proven patterns from Agno Reddit agent example.
"""
import json
from typing import Dict, Any, Optional


def score_custom_agent_response(result, expected, params, metric=None):
    """Score agent response based on proven Agno patterns."""
    # Parse agent response (from reddit_evaluator.py)
    agent_data = _parse_agent_response(result)
    
    # Route to appropriate evaluator (from example)
    if metric == "task_success":
        return _score_task_success(agent_data, expected, params)
    elif metric == "tool_efficiency":
        return _score_tool_efficiency(agent_data, expected, params)
    elif metric == "reasoning_quality":
        return _score_reasoning_quality(agent_data, expected, params)
    elif metric == "response_quality":
        return _score_response_quality(agent_data, expected, params)
    
    # Aggregate score (from reddit_evaluator.py weights)
    return (
        _score_task_success(agent_data, expected, params) * 0.35 +
        _score_tool_efficiency(agent_data, expected, params) * 0.25 +
        _score_reasoning_quality(agent_data, expected, params) * 0.20 +
        _score_response_quality(agent_data, expected, params) * 0.20
    )


def _parse_agent_response(result):
    """Parse agent response (from reddit_evaluator.py)."""
    if isinstance(result, str):
        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            return {
                'tool_calls': [],
                'final_response': result,
                'tool_results': [],
                'tokens_used': {},
                'latency_seconds': 0.0
            }
    elif isinstance(result, dict):
        data = result
    else:
        data = {'raw_result': str(result)}
    
    # Extract structured data (from example)
    parsed = {
        'tool_calls': [],
        'final_response': '',
        'tool_results': [],
        'tokens_used': {},
        'latency_seconds': 0.0
    }
    
    # Extract from Agno agent runner format (from example)
    if 'final_response' in data:
        parsed['final_response'] = data.get('final_response', '')
        parsed['tool_calls'] = data.get('tool_calls', [])
        parsed['tool_results'] = data.get('tool_results', [])
        parsed['tokens_used'] = data.get('tokens_used', {})
        parsed['latency_seconds'] = data.get('latency_seconds', 0.0)
    
    # Extract from Azure OpenAI format (fallback)
    elif 'choices' in data and len(data['choices']) > 0:
        choice = data['choices'][0]
        message = choice.get('message', {})
        parsed['final_response'] = message.get('content', '')
        if 'tool_calls' in message:
            parsed['tool_calls'] = message['tool_calls']
    
    # Extract usage
    if 'usage' in data:
        parsed['tokens_used'] = data['usage']
    
    return parsed


def _score_task_success(agent_data, expected, params):
    """Score based on task completion (from example)."""
    if expected.get('task_completed', False):
        return 1.0 if agent_data['final_response'] else 0.0
    return 0.5


def _score_tool_efficiency(agent_data, expected, params):
    """Score based on appropriate tool usage (from example)."""
    tools_used = len(agent_data['tool_calls'])
    expected_tools = expected.get('tools_used', [])
    if not expected_tools:
        return 1.0 if tools_used > 0 else 0.5
    return min(1.0, tools_used / len(expected_tools))


def _score_reasoning_quality(agent_data, expected, params):
    """Score based on reasoning coherence (from example)."""
    response = agent_data['final_response']
    if not response:
        return 0.0
    # Basic reasoning quality (can be enhanced)
    return 0.7 if len(response) > 50 else 0.5


def _score_response_quality(agent_data, expected, params):
    """Score based on response quality (from example)."""
    response = agent_data['final_response']
    if not response:
        return 0.0
    return 0.8 if len(response) > 20 else 0.5
'''
    
    def generate_yaml_content(self, config: Dict[str, Any]) -> str:
        """Generate YAML content from config."""
        # Get model info for helpful comments
        models = list(config.get('agent', {}).get('models', {}).keys())
        model_list = ', '.join(models) if models else 'configured models'
        
        yaml_content = f"""# Custom Agno Agent API Optimization Configuration
# Generated by Convergence Custom Template Generator
# 
# API: {config['api']['name']}
# Description: {config['api']['description']}
#
# Required Environment Variables:
#   {config['api']['auth']['token_env']} - LLM provider API key (e.g., Azure OpenAI, OpenAI)
#   TOOL_API_KEY - Tool-specific API key (if your agent uses authenticated tools)
#
# Setup:
#   1. Set LLM provider credentials:
#      export {config['api']['auth']['token_env']}='your-llm-provider-key'
#   2. Update agent.models registry below with your LLM deployments
#   3. Configure agent.tool_auth with your tool-specific credentials
#   4. Select models to test in search_space.parameters.model.values
#
# Model Registry:
#   Models configured: {model_list}
#   Update azure_deployment, azure_endpoint, and api_version for each model

"""
        yaml_content += yaml.dump(config, default_flow_style=False, sort_keys=False)
        return yaml_content
    
    def generate_json_content(self, test_cases: List[Dict]) -> str:
        """Generate JSON content from test cases."""
        return json.dumps({"test_cases": test_cases}, indent=2)
    
    def generate_readme_content(self, config: Dict[str, Any], provider_name: str = None) -> str:
        """Generate README content.
        
        Args:
            config: Configuration dictionary
            provider_name: Provider name (unused for agent templates, for API consistency)
        """
        # Get model info from agent config
        models = list(config.get('agent', {}).get('models', {}).keys())
        model_list = ', '.join(models) if models else 'configured models'
        
        return f"""# Custom Agno Agent Optimization

This configuration optimizes Agno agent calls using **{config['api']['description']}**.

## Setup

### 1. Set LLM Provider API Key

Agno agents use an underlying LLM provider (Azure OpenAI, OpenAI, etc.):

```bash
export {config['api']['auth']['token_env']}='your-llm-provider-api-key'
```

**Important:** This is your LLM provider API key (e.g., Azure OpenAI key), NOT an "Agno API key".

### 2. Configure Tool-Specific Credentials (if needed)

If your agent uses tools that require authentication, set those credentials:

```bash
# Example: If using Reddit tools
export TOOL_API_KEY='your-tool-api-key'
```

Update the `agent.tool_auth` section in `optimization.yaml` with your specific tool requirements.

### 3. Update Model Registry

Edit the `agent.models` section in `optimization.yaml` to configure your LLM deployments.
The template includes: {model_list}

For Azure OpenAI, update the `endpoint` field for each model:
```yaml
models:
  gpt-4:
    endpoint: "https://your-resource.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2025-01-01-preview"
  o4-mini:
    endpoint: "https://your-resource.openai.azure.com/openai/deployments/o4-mini/chat/completions?api-version=2025-01-01-preview"
```

For OpenAI, update the `endpoint` field (same endpoint for all models):
```yaml
models:
  gpt-4:
    endpoint: "https://api.openai.com/v1/chat/completions"
  gpt-3.5-turbo:
    endpoint: "https://api.openai.com/v1/chat/completions"
```

### 4. Run Optimization

```bash
convergence optimize optimization.yaml
```

## What's Being Optimized

- **model**: {', '.join(config['search_space']['parameters']['model']['values'])}
- **temperature**: {', '.join(map(str, config['search_space']['parameters']['temperature']['values']))}
- **max_completion_tokens**: {', '.join(map(str, config['search_space']['parameters']['max_completion_tokens']['values']))}
- **instruction_style**: {', '.join(config['search_space']['parameters']['instruction_style']['values'])}
- **tool_strategy**: {', '.join(config['search_space']['parameters']['tool_strategy']['values'])}

## Test Cases

The configuration includes test cases for:
- Research tasks with tool usage
- Multi-step analysis workflows
- Data extraction and processing

## Metrics

- **Task Success** ({config['evaluation']['metrics']['task_success']['weight']*100:.0f}%): Whether agent completed the goal
- **Tool Efficiency** ({config['evaluation']['metrics']['tool_efficiency']['weight']*100:.0f}%): Appropriate tool usage
- **Reasoning Quality** ({config['evaluation']['metrics']['reasoning_quality']['weight']*100:.0f}%): Coherent reasoning
- **Response Quality** ({config['evaluation']['metrics']['response_quality']['weight']*100:.0f}%): Final answer quality

## Results

Results will be saved to `{config['output']['save_path']}/`

- `best_config.py`: Best configuration found
- `report.md`: Detailed optimization report
- `detailed_results.json`: All experiment results
"""


# Export for easy import
__all__ = ['AgnoAgentTemplate']
