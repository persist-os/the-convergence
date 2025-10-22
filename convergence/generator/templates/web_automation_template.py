"""
Web Automation API Template

Based on proven patterns from examples/web_browsing/browserbase/
"""
from typing import Dict, List, Any
import yaml
import json


class WebAutomationTemplate:
    """Template for web automation APIs like BrowserBase."""
    
    def generate_config(self, endpoint: str, api_key_env: str, description: str) -> Dict[str, Any]:
        """Generate web automation API configuration."""
        return {
            'api': {
                'name': 'custom_web_automation',
                'endpoint': endpoint or 'https://api.example.com/v1/browser',
                'adapter_enabled': True,  # Enable browser adapter
                'auth': {'type': 'api_key', 'token_env': api_key_env, 'header_name': 'x-bb-api-key'}
            },
            'search_space': {
                'parameters': {
                    'browser_type': {'type': 'categorical', 'values': ['chromium', 'firefox', 'webkit']},
                    'headless': {'type': 'categorical', 'values': [True, False]},
                    'viewport_width': {'type': 'discrete', 'values': [1280, 1920, 2560]},
                    'timeout_ms': {'type': 'discrete', 'values': [5000, 10000, 30000]}
                }
            },
            'evaluation': {
                'test_cases': {'path': 'test_cases.json'},
                'metrics': {
                    'page_load_success': {'weight': 0.40, 'type': 'higher_is_better', 'function': 'custom'},
                    'element_extraction': {'weight': 0.30, 'type': 'higher_is_better', 'function': 'custom'},
                    'load_time_ms': {'weight': 0.20, 'type': 'lower_is_better', 'threshold': 5000},
                    'resource_efficiency': {'weight': 0.10, 'type': 'higher_is_better', 'function': 'custom'}
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
                'save_path': './results/custom_web_automation_optimization',
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
        """Generate web automation test cases based on BrowserBase example."""
        base_tests = [
            {
                "id": "page_navigation",
                "description": "Basic page navigation and loading",
                "input": {"url": "https://example.com", "wait_for": "body"},
                "expected": {
                    "page_loaded": True,
                    "status_code": 200,
                    "max_load_time_ms": 5000,
                    "min_quality_score": 0.8
                },
                "metadata": {"category": "navigation", "difficulty": "easy", "weight": 1.0}
            },
            {
                "id": "element_extraction",
                "description": "Extract text content from page elements",
                "input": {"url": "https://news.ycombinator.com", "selector": ".storylink"},
                "expected": {
                    "elements_found": True,
                    "min_elements": 1,
                    "max_load_time_ms": 8000,
                    "min_quality_score": 0.7
                },
                "metadata": {"category": "extraction", "difficulty": "medium", "weight": 1.5}
            },
            {
                "id": "form_interaction",
                "description": "Fill and submit forms",
                "input": {"url": "https://httpbin.org/forms/post", "form_data": {"custname": "Test User"}},
                "expected": {
                    "form_submitted": True,
                    "max_load_time_ms": 10000,
                    "min_quality_score": 0.75
                },
                "metadata": {"category": "interaction", "difficulty": "hard", "weight": 2.0}
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
        """Generate evaluator based on BrowserBase example."""
        return '''"""
Custom Web Automation API Evaluator
Generated by Convergence Custom Template Generator

This evaluator scores web automation responses based on proven patterns from BrowserBase examples.
"""
import json
from typing import Dict, Any, Optional


def score_custom_web_automation_response(result, expected, params, metric=None):
    """Score web automation response based on proven patterns."""
    # Parse browser response
    browser_data = _parse_browser_response(result)
    
    # Route to appropriate evaluator
    if metric == "page_load_success":
        return _score_page_load_success(browser_data, expected, params)
    elif metric == "element_extraction":
        return _score_element_extraction(browser_data, expected, params)
    elif metric == "load_time_ms":
        return _score_load_time(browser_data, expected, params)
    elif metric == "resource_efficiency":
        return _score_resource_efficiency(browser_data, expected, params)
    
    # Aggregate score
    return (
        _score_page_load_success(browser_data, expected, params) * 0.40 +
        _score_element_extraction(browser_data, expected, params) * 0.30 +
        _score_load_time(browser_data, expected, params) * 0.20 +
        _score_resource_efficiency(browser_data, expected, params) * 0.10
    )


def _parse_browser_response(result):
    """Parse browser automation response."""
    if isinstance(result, dict):
        return result
    elif isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {'raw_response': result}
    else:
        return {'raw_response': str(result)}


def _score_page_load_success(browser_data, expected, params):
    """Score based on successful page load."""
    if expected.get('page_loaded', False):
        if browser_data.get('status_code') == expected.get('status_code', 200):
            return 1.0
        elif browser_data.get('status_code'):
            return 0.5  # Partial credit for any response
        else:
            return 0.0
    return 0.5


def _score_element_extraction(browser_data, expected, params):
    """Score based on element extraction success."""
    if expected.get('elements_found', False):
        elements = browser_data.get('elements', [])
        min_elements = expected.get('min_elements', 1)
        if len(elements) >= min_elements:
            return 1.0
        elif len(elements) > 0:
            return len(elements) / min_elements
        else:
            return 0.0
    return 0.5


def _score_load_time(browser_data, expected, params):
    """Score based on load time performance."""
    load_time = browser_data.get('load_time_ms', 0)
    max_load_time = expected.get('max_load_time_ms', 5000)
    
    if load_time <= max_load_time:
        return 1.0
    else:
        # Gradual penalty for slow loads
        penalty = min(0.8, (load_time - max_load_time) / max_load_time)
        return max(0.0, 1.0 - penalty)


def _score_resource_efficiency(browser_data, expected, params):
    """Score based on resource usage efficiency."""
    # Basic resource efficiency scoring
    memory_usage = browser_data.get('memory_usage_mb', 0)
    cpu_usage = browser_data.get('cpu_usage_percent', 0)
    
    score = 1.0
    
    # Penalty for high memory usage
    if memory_usage > 500:  # 500MB threshold
        score -= 0.3
    
    # Penalty for high CPU usage
    if cpu_usage > 80:  # 80% threshold
        score -= 0.3
    
    return max(0.0, score)
'''
    
    def generate_yaml_content(self, config: Dict[str, Any]) -> str:
        """Generate YAML content from config."""
        yaml_content = f"""# Custom Web Automation API Optimization Configuration
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
        return f"""# Custom Web Automation Optimization

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

- **browser_type**: {', '.join(config['search_space']['parameters']['browser_type']['values'])}
- **headless**: {', '.join(map(str, config['search_space']['parameters']['headless']['values']))}
- **viewport_width**: {', '.join(map(str, config['search_space']['parameters']['viewport_width']['values']))}
- **timeout_ms**: {', '.join(map(str, config['search_space']['parameters']['timeout_ms']['values']))}

## Test Cases

The configuration includes test cases for:
- Page navigation and loading
- Element extraction from web pages
- Form interaction and submission

## Metrics

- **Page Load Success** ({config['evaluation']['metrics']['page_load_success']['weight']*100:.0f}%): Successful page loading
- **Element Extraction** ({config['evaluation']['metrics']['element_extraction']['weight']*100:.0f}%): Content extraction success
- **Load Time** ({config['evaluation']['metrics']['load_time_ms']['weight']*100:.0f}%): Performance optimization
- **Resource Efficiency** ({config['evaluation']['metrics']['resource_efficiency']['weight']*100:.0f}%): Resource usage optimization

## Results

Results will be saved to `{config['output']['save_path']}/`

- `best_config.py`: Best configuration found
- `report.md`: Detailed optimization report
- `detailed_results.json`: All experiment results
"""


# Export for easy import
__all__ = ['WebAutomationTemplate']
