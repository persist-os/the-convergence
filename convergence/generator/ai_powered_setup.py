"""
AI-Powered Setup for The Convergence

Provides natural language interface for generating optimization configurations.
"""
import json
import yaml
from pathlib import Path
from typing import Dict, Any
from rich.console import Console
from rich.prompt import Prompt

from .natural_language_processor import NaturalLanguageProcessor


async def run_ai_powered_setup(project_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """
    Run AI-powered natural language setup.
    
    Args:
        project_dir: Project directory
        output_dir: Output directory for generated files
        
    Returns:
        Dict with paths to generated files
    """
    console = Console()
    
    console.print("")
    console.print("╔" + "═" * 58 + "╗")
    console.print("║" + " " * 10 + "🤖 AI-POWERED SETUP" + " " * 25 + "║")
    console.print("║" + " " * 10 + "Natural Language Interface" + " " * 20 + "║")
    console.print("╚" + "═" * 58 + "╝")
    console.print("")
    
    # Explain what The Convergence does
    console.print("[bold cyan]What is The Convergence?[/bold cyan]")
    console.print("")
    console.print("The Convergence is an API optimization framework that finds the best")
    console.print("configuration for your API calls through intelligent testing and")
    console.print("evolutionary algorithms. It optimizes parameters like:")
    console.print("")
    console.print("• [bold]Model selection[/bold] - Which AI model works best for your use case")
    console.print("• [bold]Temperature tuning[/bold] - How creative vs consistent responses should be")
    console.print("• [bold]Token limits[/bold] - Optimal response length for your needs")
    console.print("• [bold]Cost optimization[/bold] - Balance quality with API costs")
    console.print("• [bold]Speed optimization[/bold] - Find the fastest configurations")
    console.print("")
    console.print("The system tests hundreds of configurations and learns which")
    console.print("combinations work best for your specific API and use case.")
    console.print("")
    console.print("─" * 60)
    console.print("")
    
    # Get user input with more detailed questions
    console.print("[bold cyan]Tell us about your API optimization needs[/bold cyan]")
    console.print("")
    console.print("[dim]Please provide 4-5 sentences describing:[/dim]")
    console.print("[dim]• What type of API you're using (OpenAI, Groq, etc.)[/dim]")
    console.print("[dim]• What you want to optimize for (speed, quality, cost, etc.)[/dim]")
    console.print("[dim]• Your specific use case (creative writing, Q&A, etc.)[/dim]")
    console.print("[dim]• Any specific requirements or constraints[/dim]")
    console.print("")
    console.print("[dim]Example:[/dim]")
    console.print("[dim]\"I'm using OpenAI's GPT-4 API for creative writing tasks. I want to optimize for quality over speed, but keep costs reasonable. I need responses that are creative and engaging, around 200-500 words. I'm willing to pay more for better quality but want to avoid unnecessary token usage.\"[/dim]")
    console.print("")
    
    user_input = Prompt.ask("Describe your optimization needs")
    
    if not user_input.strip():
        console.print("[yellow]No input provided, falling back to guided setup...[/yellow]")
        from .interactive_setup import run_guided_setup
        return await run_guided_setup(project_dir, output_dir)
    
    console.print("")
    console.print("─" * 60)
    console.print("")
    console.print("[bold cyan]🤖 Processing Your Request...[/bold cyan]")
    console.print("")
    
    try:
        # Initialize processor
        processor = NaturalLanguageProcessor()
        
        # Process user intent
        result = await processor.process_user_intent(user_input)
        
        console.print("[green]✅ Successfully processed your request![/green]")
        console.print("")
        
        # Save generated files
        saved_files = await _save_generated_files(
            result, output_dir, console
        )
        
        console.print("")
        console.print("─" * 60)
        console.print("")
        console.print("[bold green]🎉 AI-Powered Setup Complete![/bold green]")
        console.print("")
        console.print("Generated files:")
        console.print(f"  📄 {saved_files['config_path']}")
        console.print(f"  📄 {saved_files['test_cases_path']}")
        console.print(f"  📄 {saved_files['evaluator_path']}")
        console.print("")
        console.print("Next steps:")
        console.print(f"  1. Set your API key: export {saved_files['config']['api']['auth']['token_env']}='your-key'")
        console.print("  2. Run optimization: convergence optimize optimization.yaml")
        console.print("")
        
        return saved_files
        
    except Exception as e:
        console.print(f"[red]❌ Error in AI-powered setup: {e}[/red]")
        console.print(f"[yellow]Error type: {type(e).__name__}[/yellow]")
        import traceback
        console.print(f"[yellow]Traceback: {traceback.format_exc()}[/yellow]")
        console.print("[yellow]Falling back to guided setup...[/yellow]")
        from .interactive_setup import run_guided_setup
        return await run_guided_setup(project_dir, output_dir)


async def _save_generated_files(
    result: Dict[str, Any], 
    output_dir: Path, 
    console: Console
) -> Dict[str, Any]:
    """Save the generated files to disk."""
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate file contents
    config = result['config']  # This is now a raw YAML string
    test_cases = result['test_cases']  # This is now a raw JSON string
    evaluator_code = result['evaluator_code']
    
    # Use raw strings directly (they already contain comments)
    yaml_content = config
    json_content = test_cases
    
    # Generate README content (parse config for README generation)
    try:
        parsed_config = yaml.safe_load(config) if isinstance(config, str) else config
        parsed_test_cases = json.loads(test_cases) if isinstance(test_cases, str) else test_cases
        readme_content = _generate_readme_content(parsed_config, result['extracted_info'])
    except Exception as e:
        console.print(f"[red]❌ Error parsing generated content: {e}[/red]")
        console.print(f"[yellow]Config type: {type(config)}, Test cases type: {type(test_cases)}[/yellow]")
        if isinstance(config, str):
            console.print(f"[yellow]Config preview: {config[:200]}...[/yellow]")
        if isinstance(test_cases, str):
            console.print(f"[yellow]Test cases preview: {test_cases[:200]}...[/yellow]")
        raise e
    
    # Save files
    config_path = output_dir / "optimization.yaml"
    test_cases_path = output_dir / "test_cases.json"
    evaluator_path = output_dir / "evaluator.py"
    readme_path = output_dir / "README.md"
    
    config_path.write_text(yaml_content)
    test_cases_path.write_text(json_content)
    evaluator_path.write_text(evaluator_code)
    readme_path.write_text(readme_content)
    
    return {
        'spec_path': str(config_path),
        'config_path': str(config_path),
        'test_cases_path': str(test_cases_path),
        'evaluator_path': str(evaluator_path),
        'readme_path': str(readme_path),
        'test_cases': parsed_test_cases,  # Return parsed test cases for compatibility
        'config': parsed_config,  # Return parsed config for compatibility
        'elapsed': 0.0
    }


def _generate_readme_content(config: Dict[str, Any], extracted_info: Dict[str, Any]) -> str:
    """Generate README content for the generated configuration."""
    
    api_name = config['api']['name']
    endpoint = config['api']['endpoint']
    api_key_env = config['api']['auth']['token_env']
    provider = extracted_info.get('provider', 'api')
    use_case = extracted_info.get('use_case', 'API optimization')
    
    # Provider-specific instructions
    provider_instructions = {
        "openai": "Get your key from: https://platform.openai.com/api-keys",
        "groq": "Get your key from: https://console.groq.com/keys", 
        "azure": "Get your key from: https://portal.azure.com",
        "anthropic": "Get your key from: https://console.anthropic.com/keys",
        "custom": "Set up your custom API key environment variable"
    }
    
    api_instructions = provider_instructions.get(provider, provider_instructions["custom"])
    
    # Get search space parameters
    search_params = config.get('search_space', {}).get('parameters', {})
    param_descriptions = []
    
    for param_name, param_config in search_params.items():
        if param_config.get('type') == 'continuous':
            min_val = param_config.get('min', 0)
            max_val = param_config.get('max', 1)
            step = param_config.get('step', 0.1)
            param_descriptions.append(f"- **{param_name}**: {min_val} to {max_val} (step: {step})")
        elif param_config.get('type') == 'discrete':
            values = param_config.get('values', [])
            param_descriptions.append(f"- **{param_name}**: {', '.join(map(str, values))}")
        elif param_config.get('type') == 'categorical':
            values = param_config.get('values', [])
            param_descriptions.append(f"- **{param_name}**: {', '.join(map(str, values))}")
    
    # Get metrics
    metrics = config.get('evaluation', {}).get('metrics', {})
    metric_descriptions = []
    
    for metric_name, metric_config in metrics.items():
        weight = metric_config.get('weight', 0)
        metric_type = metric_config.get('type', 'higher_is_better')
        metric_descriptions.append(f"- **{metric_name.replace('_', ' ').title()}** ({weight*100:.0f}%): {metric_type}")
    
    return f"""# {provider.title()} API Optimization

This configuration optimizes API calls to **{api_name}** at `{endpoint}`.

## What This Optimizes

{use_case}

## Setup

1. **Set your API key environment variable:**
   ```bash
   export {api_key_env}='your-actual-api-key-value'
   ```
   
   **Important:** Replace `your-actual-api-key-value` with your real API key, not the variable name.
   {api_instructions}

2. **Update the endpoint (if needed):**
   Edit `optimization.yaml` and change the `endpoint` field to your actual API endpoint.

3. **Run optimization:**
   ```bash
   convergence optimize optimization.yaml
   ```

## What's Being Optimized

{chr(10).join(param_descriptions)}

## Metrics

{chr(10).join(metric_descriptions)}

## Test Cases

The configuration includes test cases for:
- Various scenarios based on your use case
- Quality, latency, and cost evaluation
- Realistic expected outputs

## Results

Results will be saved to `./results/{api_name}/`

- `best_config.py`: Best configuration found
- `report.md`: Detailed optimization report
- `detailed_results.json`: All experiment results

## Generated by AI-Powered Setup

This configuration was generated using The Convergence's AI-powered setup,
which analyzed your natural language description and created an optimized
configuration tailored to your specific needs.
"""
