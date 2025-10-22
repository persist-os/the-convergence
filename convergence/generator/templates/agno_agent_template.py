"""
Agno Agent API Template

Based on proven patterns from examples/agno_agents/reddit/
"""
from typing import Dict, List, Any
import yaml
import json


class AgnoAgentTemplate:
    """Template for Agno agent APIs."""
    
    def generate_config(self, endpoint: str, api_key_env: str, description: str) -> Dict[str, Any]:
        """Generate Agno agent API configuration."""
        return {
            'api': {
                'name': 'custom_agno_agent',
                'endpoint': endpoint or 'https://api.example.com/v1/agent',
                'adapter_enabled': True,  # Enable agent adapter
                'auth': {'type': 'bearer', 'token_env': api_key_env}
            },
            'search_space': {
                'parameters': {
                    'instruction_style': {'type': 'categorical', 'values': ['minimal', 'detailed', 'structured']},
                    'tool_selection_strategy': {'type': 'categorical', 'values': ['all', 'minimal', 'adaptive']},
                    'reasoning_temperature': {'type': 'continuous', 'min': 0.1, 'max': 1.0, 'step': 0.1},
                    'max_reasoning_tokens': {'type': 'discrete', 'values': [256, 512, 1024, 2048]}
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
        yaml_content = f"""# Custom Agno Agent API Optimization Configuration
# Generated by Convergence Custom Template Generator
# 
# API: {config['api']['name']}
# Endpoint: {config['api']['endpoint']}
# Generated: 2025-01-23 10:30:00
#
# Required Environment Variables:
#   {config['api']['auth']['token_env']} - Your API key
#
# Set before running:
#   export {config['api']['auth']['token_env']}='your-actual-key-here'

"""
        yaml_content += yaml.dump(config, default_flow_style=False, sort_keys=False)
        return yaml_content
    
    def generate_json_content(self, test_cases: List[Dict]) -> str:
        """Generate JSON content from test cases."""
        return json.dumps({"test_cases": test_cases}, indent=2)
    
    def generate_readme_content(self, config: Dict[str, Any]) -> str:
        """Generate README content."""
        return f"""# Custom Agno Agent Optimization

This configuration optimizes API calls to **{config['api']['name']}** at `{config['api']['endpoint']}`.

## Setup

1. **Set your API key environment variable:**
   ```bash
   export {config['api']['auth']['token_env']}='your-actual-api-key-value'
   ```
   
   **Important:** Replace `your-actual-api-key-value` with your real API key, not the variable name.

2. **Update the endpoint (if needed):**
   Edit `optimization.yaml` and change the `endpoint` field to your actual API endpoint.

3. **Run optimization:**
   ```bash
   convergence optimize optimization.yaml
   ```

## What's Being Optimized

- **instruction_style**: {', '.join(config['search_space']['parameters']['instruction_style']['values'])}
- **tool_selection_strategy**: {', '.join(config['search_space']['parameters']['tool_selection_strategy']['values'])}
- **reasoning_temperature**: {config['search_space']['parameters']['reasoning_temperature']['min']} to {config['search_space']['parameters']['reasoning_temperature']['max']} (step: {config['search_space']['parameters']['reasoning_temperature']['step']})
- **max_reasoning_tokens**: {', '.join(map(str, config['search_space']['parameters']['max_reasoning_tokens']['values']))}

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
