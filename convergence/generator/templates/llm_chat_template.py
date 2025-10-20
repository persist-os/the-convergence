"""
LLM Chat API Template

Based on proven patterns from examples/ai/openai/, examples/ai/groq/, examples/ai/azure/
"""
from typing import Dict, List, Any
import yaml
import json


class LLMChatTemplate:
    """Template for OpenAI-compatible chat APIs."""
    
    def generate_config(self, endpoint: str, api_key_env: str, description: str) -> Dict[str, Any]:
        """Generate LLM chat API configuration."""
        return {
            'api': {
                'name': 'custom_llm_chat',
                'endpoint': endpoint or 'https://api.example.com/v1/chat/completions',
                'auth': {'type': 'bearer', 'token_env': api_key_env},
                'request': {'method': 'POST', 'headers': {'Content-Type': 'application/json'}},
                'response': {'result_field': 'choices[0].message.content'}
            },
            'search_space': {
                'parameters': {
                    'model': {'type': 'categorical', 'values': ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo']},
                    'temperature': {'type': 'continuous', 'min': 0.1, 'max': 1.0, 'step': 0.1},
                    'max_tokens': {'type': 'discrete', 'values': [100, 256, 512, 1024]}
                }
            },
            'evaluation': {
                'test_cases': {'path': 'test_cases.json'},
                'metrics': {
                    'response_quality': {'weight': 0.40, 'type': 'higher_is_better', 'function': 'custom'},
                    'latency_ms': {'weight': 0.30, 'type': 'lower_is_better', 'threshold': 2000},
                    'cost_per_call': {'weight': 0.30, 'type': 'lower_is_better', 'budget_per_call': 0.01}
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
                'save_path': './results/custom_llm_chat_optimization',
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
        """Generate test cases based on description."""
        # Generate test cases based on description keywords
        description_lower = description.lower()
        
        # Base test cases that work for most APIs
        base_tests = []
        
        # Add relevant test cases based on description
        if any(word in description_lower for word in ['chat', 'chatbot', 'conversation', ' support']):
            base_tests.extend([
                {
                    "id": "chat_response",
                    "description": "Basic chat response",
                    "input": {"messages": [{"role": "user", "content": "Hello, how are you?"}]},
                    "expected": {"min_length": 10, "min_quality_score": 0.7},
                    "metadata": {"category": "chat", "difficulty": "easy", "weight": 1.0}
                },
                {
                    "id": "help_request",
                    "description": "Help request handling",
                    "input": {"messages": [{"role": "user", "content": "Can you help me with my problem?"}]},
                    "expected": {"contains": ["help"], "min_length": 15, "min_quality_score": 0.75},
                    "metadata": {"category": "support", "difficulty": "easy", "weight": 1.0}
                }
            ])
        
        if any(word in description_lower for word in ['creative', 'writing', 'story', 'content']):
            base_tests.extend([
                {
                    "id": "creative_writing",
                    "description": "Creative text generation",
                    "input": {"messages": [{"role": "user", "content": "Write a short story about a robot"}]},
                    "expected": {"contains": ["robot"], "min_length": 20, "min_quality_score": 0.75},
                    "metadata": {"category": "creative", "difficulty": "easy", "weight": 1.0}
                }
            ])
        
        if any(word in description_lower for word in ['qa', 'question', 'answer', 'knowledge']):
            base_tests.extend([
                {
                    "id": "factual_qa",
                    "description": "Question answering", 
                    "input": {"messages": [{"role": "user", "content": "What is the capital of Japan?"}]},
                    "expected": {"contains": ["Tokyo"], "min_length": 5, "min_quality_score": 0.95},
                    "metadata": {"category": "qa", "difficulty": "easy", "weight": 1.0}
                }
            ])
        
        if any(word in description_lower for word in ['math', 'calculation', 'reasoning', 'problem']):
            base_tests.extend([
                {
                    "id": "math_reasoning",
                    "description": "Math problem solving",
                    "input": {"messages": [{"role": "user", "content": "A car travels at 80 km/h for 3 hours. How far does it travel?"}]},
                    "expected": {"contains": ["240"], "min_length": 10, "min_quality_score": 0.9},
                    "metadata": {"category": "reasoning", "difficulty": "medium", "weight": 1.5}
                }
            ])
        
        # If no specific test cases were added, use generic ones
        if not base_tests:
            base_tests = [
                {
                    "id": "general_response",
                    "description": "General API response",
                    "input": {"messages": [{"role": "user", "content": "Test message"}]},
                    "expected": {"min_length": 5, "min_quality_score": 0.6},
                    "metadata": {"category": "general", "difficulty": "easy", "weight": 1.0}
                },
                {
                    "id": "detailed_response",
                    "description": "Detailed API response",
                    "input": {"messages": [{"role": "user", "content": "Please provide a detailed explanation"}]},
                    "expected": {"min_length": 20, "min_quality_score": 0.7},
                    "metadata": {"category": "general", "difficulty": "medium", "weight": 1.2}
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
        """Generate evaluator based on OpenAI example."""
        return '''"""
Custom LLM API Evaluator
Generated by Convergence Custom Template Generator

This evaluator scores LLM responses based on proven patterns from OpenAI examples.
"""
import re
from typing import Dict, Any, Optional


def score_custom_llm_response(result, expected, params, metric=None):
    """Score LLM response based on proven patterns."""
    # Extract text from response
    text = _extract_text(result)
    if not text:
        return 0.0
    
    # Calculate scores using existing patterns
    scores = {}
    scores['completeness'] = _score_completeness(text, expected)
    scores['quality'] = _score_quality(text)
    scores['length'] = _score_length(text, expected)
    
    # Combined quality score (like examples)
    quality_score = (
        scores['completeness'] * 0.6 +
        scores['quality'] * 0.25 + 
        scores['length'] * 0.15
    )
    
    # Latency and cost scores
    scores['latency'] = _score_latency(result, expected)
    scores['cost'] = _score_cost(result, expected)
    
    # Return specific metric if requested
    if metric:
        metric_lower = metric.lower()
        if metric_lower in scores:
            return scores[metric_lower]
        elif metric_lower == 'response_quality':
            return quality_score
        return 0.0
    
    # Weighted overall score
    overall_score = (
        quality_score * 0.40 +
        scores['latency'] * 0.30 +
        scores['cost'] * 0.30
    )
    
    return min(1.0, max(0.0, overall_score))


def _extract_text(result):
    """Extract text from OpenAI-compatible response."""
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


def _score_completeness(text, expected):
    """Score based on required keywords (from examples)."""
    if 'contains' not in expected:
        return 1.0
    
    required_keywords = expected['contains']
    if not required_keywords:
        return 1.0
    
    text_lower = text.lower()
    found_keywords = 0
    
    for keyword in required_keywords:
        if keyword.lower() in text_lower:
            found_keywords += 1
    
    return found_keywords / len(required_keywords)


def _score_quality(text):
    """Basic text quality scoring (from examples)."""
    if not text or len(text.strip()) < 3:
        return 0.0
    
    score = 0.5  # Base score
    
    # Length appropriateness
    if 10 <= len(text) <= 1000:
        score += 0.2
    
    # Sentence structure
    sentences = text.split('.')
    if len(sentences) > 1:
        score += 0.1
    
    # Word variety (basic)
    words = text.split()
    if len(set(words)) / len(words) > 0.5:
        score += 0.1
    
    # No obvious errors
    if not re.search(r'\\b(error|fail|wrong|incorrect)\\b', text.lower()):
        score += 0.1
    
    return min(1.0, score)


def _score_length(text, expected):
    """Score based on length requirements (from examples)."""
    length = len(text)
    
    min_length = expected.get('min_length', 0)
    max_length = expected.get('max_length', float('inf'))
    
    if length < min_length:
        return 0.0
    
    if length > max_length:
        # Gradual penalty for being too long
        excess = length - max_length
        penalty = min(0.5, excess / max_length)
        return max(0.0, 1.0 - penalty)
    
    return 1.0


def _score_latency(result, expected):
    """Score based on response latency (from examples)."""
    if isinstance(result, dict) and 'latency_ms' in result:
        latency = result['latency_ms']
        max_latency = expected.get('max_latency_ms', 2000)
        
        if latency <= max_latency:
            return 1.0
        else:
            # Gradual penalty for slow responses
            penalty = min(0.8, (latency - max_latency) / max_latency)
            return max(0.0, 1.0 - penalty)
    
    return 0.5  # Default score if no latency data


def _score_cost(result, expected):
    """Score based on cost efficiency (from examples)."""
    if isinstance(result, dict) and 'cost_usd' in result:
        cost = result['cost_usd']
        max_cost = expected.get('max_cost_usd', 0.01)
        
        if cost <= max_cost:
            return 1.0
        else:
            # Gradual penalty for expensive calls
            penalty = min(0.8, (cost - max_cost) / max_cost)
            return max(0.0, 1.0 - penalty)
    
    return 0.5  # Default score if no cost data
'''
    
    def generate_yaml_content(self, config: Dict[str, Any]) -> str:
        """Generate YAML content from config."""
        yaml_content = f"""# Custom LLM Chat API Optimization Configuration
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
        return f"""# Custom LLM Chat Optimization

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

- **model**: {', '.join(config['search_space']['parameters']['model']['values'])}
- **temperature**: {config['search_space']['parameters']['temperature']['min']} to {config['search_space']['parameters']['temperature']['max']} (step: {config['search_space']['parameters']['temperature']['step']})
- **max_tokens**: {', '.join(map(str, config['search_space']['parameters']['max_tokens']['values']))}

## Test Cases

The configuration includes test cases for:
- Creative writing
- Question answering  
- Math reasoning

## Metrics

- **Response Quality** ({config['evaluation']['metrics']['response_quality']['weight']*100:.0f}%): Completeness, quality, length
- **Latency** ({config['evaluation']['metrics']['latency_ms']['weight']*100:.0f}%): Response time performance
- **Cost** ({config['evaluation']['metrics']['cost_per_call']['weight']*100:.0f}%): Token efficiency

## Results

Results will be saved to `{config['output']['save_path']}/`

- `best_config.py`: Best configuration found
- `report.md`: Detailed optimization report
- `detailed_results.json`: All experiment results
"""


# Export for easy import
__all__ = ['LLMChatTemplate']
