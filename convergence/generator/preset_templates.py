"""
Preset templates for quick setup.

Converts any example folder into a ready-to-use template.
"""
from pathlib import Path
from typing import Dict, Any, List
import json
import shutil
import yaml


def list_available_templates() -> List[Dict[str, Any]]:
    """
    List all available preset templates.
    
    Returns:
        List of template metadata dicts
    """
    return [
        {
            "id": "openai",
            "name": "OpenAI/ChatGPT (Recommended)",
            "description": "Optimize ChatGPT API calls - model & temperature tuning",
            "features": ["8 test cases", "Built-in evaluator", "Agent society ready"],
            "test_count": 8
        },
        {
            "id": "browserbase",
            "name": "BrowserBase Web Automation",
            "description": "Optimize browser automation - viewport, timeout, wait strategy",
            "features": ["5 test cases", "Custom evaluator", "RLP + SAO enabled"],
            "test_count": 5
        },
        {
            "id": "groq",
            "name": "Groq (Ultra-Fast LLMs)",
            "description": "Optimize Groq's fast inference - models, temperature, tokens",
            "features": ["Test cases included", "Speed-focused", "Legacy warm-start"],
            "test_count": "Multiple"
        },
        {
            "id": "azure",
            "name": "Azure OpenAI",
            "description": "Optimize Azure-hosted OpenAI - enterprise deployment",
            "features": ["Reasoning tests", "Custom evaluator", "Enterprise config"],
            "test_count": "Multiple"
        },
        {
            "id": "reddit",
            "name": "Reddit Agent (Agno + Azure)",
            "description": "Optimize Reddit data retrieval agents with Azure OpenAI",
            "features": ["4 test cases", "Custom evaluator", "MAB optimization", "Azure model registry"],
            "test_count": 4
        }
    ]


async def create_preset_config(
    preset_name: str, 
    project_dir: Path, 
    output_dir: Path,
    society_config: Dict[str, Any] = None,
    config_overrides: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create configuration by copying a working example.
    
    Args:
        preset_name: Template ID (openai, browserbase, groq, gemini, azure)
        project_dir: Project directory
        output_dir: Where to write config files
        society_config: Optional agent society configuration to inject
        config_overrides: Optional configuration overrides from interactive setup
    
    Returns:
        Result dict with paths and metadata
    """
    # Template mapping to actual example paths
    template_map = {
            "openai": {
        "config": "ai/openai/openai_optimization.yaml",
            "tests": "ai/openai/openai_responses_tests.json",
            "evaluator": "ai/openai/openai_responses.py",
            "name": "OpenAI"
        },
        "browserbase": {
            "config": "web_browsing/browserbase/browserbase_optimization.yaml",
            "tests": "web_browsing/browserbase/browserbase_tests.json",
            "evaluator": "web_browsing/browserbase/browserbase_evaluator.py",
            "name": "BrowserBase"
        },
        "groq": {
            "config": "ai/groq/groq_optimization.yaml",
            "tests": "ai/groq/groq_responses_tests.json",
            "evaluator": "ai/groq/groq_responses.py",
            "name": "Groq"
        },
        "azure": {
            "config": "ai/azure/azure_multi_model_optimization.yaml",
            "tests": "ai/azure/azure_test_cases.json",
            "evaluator": "ai/azure/azure_multi_model_evaluator.py",
            "name": "Azure OpenAI"
        },
        "reddit": {
            "config": "agno_agents/reddit/reddit_agent_optimization.yaml",
            "tests": "agno_agents/reddit/reddit_test_cases.json",
            "evaluator": "agno_agents/reddit/reddit_evaluator.py",
            "runner": "agno_agents/reddit/reddit_agent_runner.py",
            "name": "Reddit Agent"
        }
    }
    
    if preset_name not in template_map:
        raise ValueError(f"Unknown template: {preset_name}. Available: {list(template_map.keys())}")
    
    template = template_map[preset_name]
    
    # Find the convergence package root
    import convergence
    convergence_root = Path(convergence.__file__).parent.parent
    
    # Source files (the working examples)
    source_config = convergence_root / "examples" / template["config"]
    source_tests = convergence_root / "examples" / template["tests"]
    source_evaluator = convergence_root / "examples" / template["evaluator"]
    
    # Optional runner file (for Reddit template)
    source_runner = None
    if "runner" in template:
        source_runner = convergence_root / "examples" / template["runner"]
    
    # Destination files
    config_path = output_dir / "optimization.yaml"
    tests_path = output_dir / "test_cases.json"
    evaluator_path = output_dir / "evaluator.py"
    runner_path = output_dir / "reddit_agent_runner.py" if source_runner else None
    
    # Verify source files exist
    if not source_config.exists():
        raise FileNotFoundError(f"Source config not found: {source_config}")
    if not source_tests.exists():
        raise FileNotFoundError(f"Source tests not found: {source_tests}")
    
    # Copy and modify config
    config_content = source_config.read_text()
    
    # Update test cases path
    config_content = _update_config_paths(config_content, preset_name)
    
    # Apply user configuration overrides
    if config_overrides:
        config_content = _apply_config_overrides(config_content, config_overrides)
    
    # Inject society config if provided
    if society_config and society_config.get("enabled"):
        config_content = _inject_society_config(config_content, society_config)
    
    config_path.write_text(config_content)
    
    # Copy test cases
    shutil.copy2(source_tests, tests_path)
    test_cases = json.loads(source_tests.read_text())
    test_count = len(test_cases.get('test_cases', []) if isinstance(test_cases, dict) else test_cases)
    
    # Copy evaluator if it exists
    evaluator_copied = False
    if source_evaluator.exists():
        shutil.copy2(source_evaluator, evaluator_path)
        evaluator_copied = True
    
    # Copy runner if it exists (for Reddit template)
    runner_copied = False
    if source_runner and source_runner.exists():
        shutil.copy2(source_runner, runner_path)
        runner_copied = True
    
    # Create data directory structure
    (output_dir / "data").mkdir(exist_ok=True)
    (output_dir / "results").mkdir(exist_ok=True)
    
    # Import console for output
    from rich.console import Console
    console = Console()
    
    console.print("")
    console.print(f"✅ Copied {template['name']} config")
    console.print(f"✅ Copied {test_count} test cases")
    if evaluator_copied:
        console.print(f"✅ Copied custom evaluator")
    if runner_copied:
        console.print(f"✅ Copied agent runner")
    console.print("")
    console.print("🎉 [bold green]Template ready![/bold green]")
    console.print("")
    console.print("Next steps:")
    console.print(f"  1. Review optimization.yaml")
    console.print(f"  2. Set required API keys (see config comments)")
    console.print(f"  3. Run: convergence optimize optimization.yaml")
    console.print("")
    
    return {
        'spec_path': f'preset:{preset_name}',
        'config_path': str(config_path),
        'tests_path': str(tests_path),
        'evaluator_path': str(evaluator_path) if source_evaluator.exists() else None,
        'runner_path': str(runner_path) if runner_copied else None,
        'test_cases': test_cases.get('test_cases', []) if isinstance(test_cases, dict) else test_cases,
        'config': {},
        'elapsed': 0.1
    }


def _update_config_paths(config_content: str, preset_name: str) -> str:
    """Update all paths in config to be relative to current directory."""
    
    # Generic path updates that apply to all templates
    replacements = [
        # Test cases paths
        (r'path: ".*test.*\.json"', 'path: "test_cases.json"'),
        # Results paths
        (r'save_path: "\.\/results\/.*"', 'save_path: "./results"'),
        # Best config paths
        (r'output_path: "\.\/best_.*\.py"', 'output_path: "./best_config.py"'),
        # Data/legacy paths (keep relative)
        (r'sqlite_path: "\.\/data/', 'sqlite_path: "./data/'),
        (r'export_dir: "\./', 'export_dir: "./'),
    ]
    
    # Template-specific updates
    if preset_name == "openai":
        replacements.extend([
            ('path: "examples/ai/openai/openai_responses_tests.json"', 'path: "test_cases.json"'),
            ('path: "openai_responses_tests.json"', 'path: "test_cases.json"'),
            ('module: "openai_responses"', 'module: "evaluator"'),
        ])
    elif preset_name == "browserbase":
        replacements.extend([
            ('path: "browserbase_tests.json"', 'path: "test_cases.json"'),
            ('module: "browserbase_evaluator"', 'module: "evaluator"'),
        ])
    elif preset_name == "groq":
        replacements.extend([
            ('path: "groq_responses_tests.json"', 'path: "test_cases.json"'),
            ('module: "groq_responses"', 'module: "evaluator"'),
        ])
    elif preset_name == "gemini":
        replacements.extend([
            ('path: "gemini_task_decomposition_tests.json"', 'path: "test_cases.json"'),
            ('module: "gemini_evaluator"', 'module: "evaluator"'),
        ])
    elif preset_name == "azure":
        replacements.extend([
            ('path: "azure_test_cases.json"', 'path: "test_cases.json"'),
            ('path: "reasoning_tests.json"', 'path: "test_cases.json"'),
            ('module: "azure_multi_model_evaluator"', 'module: "evaluator"'),
        ])
    elif preset_name == "reddit":
        replacements.extend([
            ('path: "reddit_test_cases.json"', 'path: "test_cases.json"'),
            ('module: "reddit_evaluator"', 'module: "evaluator"'),
        ])
    
    # Apply all replacements
    import re
    for pattern, replacement in replacements:
        config_content = re.sub(pattern, replacement, config_content)
    
    return config_content


def _apply_config_overrides(config_content: str, overrides: Dict[str, Any]) -> str:
    """
    Apply user configuration overrides to the YAML config.
    
    Overrides can include:
    - api_key_env: Environment variable for API authentication
    - optimization: Dict with execution settings (parallel_workers, generations, etc.)
    - output_path: Where to save results
    - metric_weights: Dict with metric weights (response_quality, latency_ms, etc.)
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        config_data = yaml.safe_load(config_content)
    except Exception as e:
        logger.warning(f"Could not parse YAML to apply overrides: {e}")
        return config_content
    
    # Apply API key override
    if 'api_key_env' in overrides:
        if 'api' not in config_data:
            config_data['api'] = {}
        if 'auth' not in config_data['api']:
            config_data['api']['auth'] = {}
        config_data['api']['auth']['token_env'] = overrides['api_key_env']
    
    # Apply optimization settings
    if 'optimization' in overrides:
        if 'optimization' not in config_data:
            config_data['optimization'] = {}
        if 'execution' not in config_data['optimization']:
            config_data['optimization']['execution'] = {}
        
        opt_overrides = overrides['optimization']
        if 'parallel_workers' in opt_overrides:
            config_data['optimization']['execution']['parallel_workers'] = opt_overrides['parallel_workers']
        if 'experiments_per_generation' in opt_overrides:
            config_data['optimization']['execution']['experiments_per_generation'] = opt_overrides['experiments_per_generation']
        
        # Evolution settings
        if 'evolution' not in config_data['optimization']:
            config_data['optimization']['evolution'] = {}
        if 'population_size' in opt_overrides:
            config_data['optimization']['evolution']['population_size'] = opt_overrides['population_size']
        if 'generations' in opt_overrides:
            config_data['optimization']['evolution']['generations'] = opt_overrides['generations']
    
    # Apply output path
    if 'output_path' in overrides:
        if 'output' not in config_data:
            config_data['output'] = {}
        config_data['output']['save_path'] = overrides['output_path']
    
    # Apply metric weights
    if 'metric_weights' in overrides:
        if 'evaluation' not in config_data:
            config_data['evaluation'] = {}
        if 'metrics' not in config_data['evaluation']:
            config_data['evaluation']['metrics'] = {}
        
        for metric_name, weight in overrides['metric_weights'].items():
            if metric_name in config_data['evaluation']['metrics']:
                config_data['evaluation']['metrics'][metric_name]['weight'] = weight
    
    # Convert back to YAML
    try:
        return yaml.dump(config_data, default_flow_style=False, sort_keys=False)
    except Exception as e:
        logger.warning(f"Could not serialize YAML after applying overrides: {e}")
        return config_content


def _inject_society_config(config_content: str, society_config: Dict[str, Any]) -> str:
    """
    Inject agent society configuration into YAML config.
    
    If society section exists, updates it. Otherwise, adds it.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        config_data = yaml.safe_load(config_content)
    except Exception as e:
        logger.warning(f"Could not parse YAML to inject society config: {e}")
        return config_content
    
    # Update or create society section
    if 'society' not in config_data:
        config_data['society'] = {}
    
    config_data['society']['enabled'] = True
    
    # Add LLM config
    if 'llm' not in config_data['society']:
        config_data['society']['llm'] = {}
    
    config_data['society']['llm']['model'] = society_config.get('model', 'gemini/gemini-2.0-flash-exp')
    config_data['society']['llm']['api_key_env'] = society_config.get('api_key_env', 'GEMINI_API_KEY')
    
    # Keep other society settings if they exist
    if 'auto_generate_agents' not in config_data['society']:
        config_data['society']['auto_generate_agents'] = True
    
    # Convert back to YAML
    try:
        return yaml.dump(config_data, default_flow_style=False, sort_keys=False)
    except Exception as e:
        logger.warning(f"Could not serialize YAML after injecting society config: {e}")
        return config_content
