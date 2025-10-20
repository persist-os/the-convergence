"""
Interactive setup for Convergence when no OpenAPI spec is found.

Provides:
1. Fetch from URL
2. Use preset template (OpenAI)
"""
from pathlib import Path
from typing import Dict, Any
from rich.console import Console
from rich.prompt import Prompt, Confirm


async def run_interactive_setup(project_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """
    Main entry point for interactive setup.
    
    Returns result dict with config/test paths.
    """
    import os
    console = Console()
    
    console.print("")
    console.print("╔" + "═" * 58 + "╗")
    console.print("║" + " " * 15 + "🎯 THE CONVERGENCE" + " " * 24 + "║")
    console.print("║" + " " * 15 + "Interactive Setup" + " " * 26 + "║")
    console.print("╚" + "═" * 58 + "╝")
    console.print("")
    console.print("[dim]This wizard will help you create an optimized API configuration.[/dim]")
    console.print("[dim]Answer each question or press Enter to use recommended defaults.[/dim]")
    console.print("")
    console.print("─" * 60)
    console.print("")
    
    # Step 1: Choose setup approach
    console.print("[bold cyan]Choose your starting point:[/bold cyan]")
    console.print("")
    console.print("1. Guided Setup - Answer a few questions, we'll fill the rest")
    console.print("2. Use Preset Template - Start from working examples")
    console.print("3. Custom Template - Build from proven patterns")
    console.print("")
    
    choice = Prompt.ask(
        "Select approach",
        choices=["1", "2", "3"],
        default="1"
    )
    
    if choice == "1":
        return await run_guided_setup(project_dir, output_dir)
    elif choice == "2":
        return await run_preset_template_setup(project_dir, output_dir)
    elif choice == "3":
        return await run_custom_template_setup(project_dir, output_dir)


async def run_preset_template_setup(project_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """Run preset template setup (existing functionality)."""
    import os
    console = Console()
    
    console.print("[bold cyan]STEP 1: Choose Your API Template[/bold cyan]")
    console.print("")
    console.print("[dim]Select the API you want to optimize. Each template includes:[/dim]")
    console.print("[dim]• Pre-configured test cases[/dim]")
    console.print("[dim]• Working optimization settings[/dim]")
    console.print("[dim]• Custom evaluation functions[/dim]")
    console.print("")
    
    from .preset_templates import list_available_templates
    templates = list_available_templates()
    
    for i, template in enumerate(templates, 1):
        console.print(f"  [cyan]{i}.[/cyan] {template['name']}")
        console.print(f"      {template['description']}")
        if template.get('features'):
            console.print(f"      [dim]Features: {', '.join(template['features'])}[/dim]")
        console.print("")
    
    choice = Prompt.ask(
        "Select template",
        choices=[str(i) for i in range(1, len(templates) + 1)],
        default="1"
    )
    
    choice_int = int(choice)
    selected_template = templates[choice_int - 1]
    
    console.print("")
    console.print(f"📋 [bold]Using {selected_template['name']}[/bold]")
    console.print("")
    console.print("This will copy:")
    console.print("  ✅ Working optimization config")
    console.print(f"  ✅ {selected_template.get('test_count', 'Multiple')} test cases")
    console.print("  ✅ Custom evaluator (if needed)")
    console.print("")
    
    # Step 2: Configure key settings
    console.print("")
    console.print("─" * 60)
    console.print("")
    console.print("[bold cyan]STEP 2: Configure Optimization[/bold cyan]")
    console.print("")
    console.print("[dim]Customize how the optimization runs. These settings control:[/dim]")
    console.print("[dim]• How many API calls to make[/dim]")
    console.print("[dim]• What to optimize for (quality vs speed vs cost)[/dim]")
    console.print("[dim]• Where to save results[/dim]")
    console.print("")
    
    config_overrides = await _gather_config_preferences(console, selected_template)
    
    # Step 3: Agent Society (Experimental)
    console.print("")
    console.print("─" * 60)
    console.print("")
    console.print("[bold yellow]STEP 3: Agent Society (EXPERIMENTAL - Optional)[/bold yellow]")
    console.print("")
    console.print("[yellow]⚠️  This feature is experimental and under active development.[/yellow]")
    console.print("[yellow]   Most users should skip this for initial setup.[/yellow]")
    console.print("")
    console.print("[dim]What is Agent Society?[/dim]")
    console.print("[dim]AI agents that learn from each run to find better configurations.[/dim]")
    console.print("")
    console.print("[dim]Includes:[/dim]")
    console.print("  • [dim]RLP: Reasoning-based Learning Process[/dim]")
    console.print("  • [dim]SAO: Self-Alignment Optimization[/dim]")
    console.print("  • [dim]Auto-generated agents tailored to your API[/dim]")
    console.print("")
    console.print("[dim]Note: Multi-agent collaboration coming in future version[/dim]")
    console.print("")
    
    enable_society = Prompt.ask(
        "Enable agent society? (experimental)",
        choices=["y", "n"],
        default="n"
    ) == "y"
    
    society_config = {}
    if enable_society:
        console.print("")
        console.print("🔑 [bold]Agent Society LLM Configuration[/bold]")
        console.print("")
        console.print("The agent society needs an LLM to coordinate agents.")
        console.print("Supports any LiteLLM model (OpenAI, Gemini, Claude, etc.)")
        console.print("")
        console.print("Examples:")
        console.print("  • openai/gpt-4o-mini (OPENAI_API_KEY)")
        console.print("  • gemini/gemini-2.0-flash-exp (GEMINI_API_KEY)")
        console.print("  • anthropic/claude-3-haiku (ANTHROPIC_API_KEY)")
        console.print("")
        
        model = Prompt.ask(
            "LLM model",
            default="gemini/gemini-2.0-flash-exp"
        )
        
        # Infer API key env var from model
        if model.startswith("openai/"):
            default_key_env = "OPENAI_API_KEY"
        elif model.startswith("gemini/"):
            default_key_env = "GEMINI_API_KEY"
        elif model.startswith("anthropic/"):
            default_key_env = "ANTHROPIC_API_KEY"
        else:
            default_key_env = "API_KEY"
        
        console.print(f"[bold yellow]⚠️  Enter the ENVIRONMENT VARIABLE NAME (e.g., {default_key_env})[/bold yellow]")
        console.print("[dim]NOT the actual API key value![/dim]")
        console.print("")
        
        key_env = Prompt.ask(
            "Environment variable name",
            default=default_key_env
        )
        
        # Validate that they didn't paste an actual API key
        if _looks_like_api_key(key_env):
            console.print("")
            console.print(f"[bold red]❌ ERROR: You entered what looks like an actual API key![/bold red]")
            console.print(f"[yellow]Please enter the ENVIRONMENT VARIABLE NAME, not the key itself.[/yellow]")
            console.print(f"[dim]Example: {default_key_env}[/dim]")
            console.print("")
            key_env = Prompt.ask(
                "Environment variable name",
                default=default_key_env
            )
        
        # Check if key is set
        if not os.getenv(key_env):
            console.print("")
            console.print(f"[yellow]⚠️  {key_env} not set[/yellow]")
            console.print("")
            provide_key = Prompt.ask(
                "Provide API key now?",
                choices=["y", "n"],
                default="n"
            ) == "y"
            
            if provide_key:
                from rich.prompt import Prompt as SecurePrompt
                api_key = SecurePrompt.ask(
                    "Enter API key",
                    password=True
                )
                os.environ[key_env] = api_key
                console.print(f"✅ {key_env} set for this session")
                console.print("")
                console.print(f"💡 To persist: export {key_env}='{api_key}'")
        else:
            console.print(f"✅ {key_env} already set")
        
        console.print("")
        
        society_config = {
            "enabled": True,
            "model": model,
            "api_key_env": key_env
        }
    
    # Step 4: Create the template
    from .preset_templates import create_preset_config
    return await create_preset_config(
        selected_template['id'], 
        project_dir, 
        output_dir,
        society_config,
        config_overrides
    )


async def run_guided_setup(project_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """Run guided setup with step-by-step questions."""
    console = Console()
    
    console.print("\n🎯 [bold cyan]Guided Setup - Customize Your Configuration[/bold cyan]")
    console.print("")
    
    # Choose API type
    console.print("What type of API are you optimizing?")
    console.print("1. LLM Chat API (OpenAI-style responses)")
    console.print("2. Agent API (LLM with tools/functions)")
    console.print("3. Web Automation (browser control)")
    console.print("4. Generic REST API")
    console.print("")
    
    api_choice = Prompt.ask("Select API type", choices=["1", "2", "3", "4"], default="1")
    
    # Get basic info (simplified)
    console.print("")
    console.print("[bold yellow]⚠️  Enter the ENVIRONMENT VARIABLE NAME (default is API_KEY)[/bold yellow]")
    console.print("[dim]NOT the actual API key value! You'll need to export it in your terminal.[/dim]")
    console.print("")
    api_key_env = Prompt.ask("API key env var name", default="API_KEY")
    console.print("")
    console.print("[dim]Optional: Describe what your API does to help generate relevant test cases.[/dim]")
    description = Prompt.ask("What does your API do?", default="API optimization")
    
    # Map API choice to template type
    template_mapping = {
        "1": "llm_chat",
        "2": "agno_agent", 
        "3": "web_automation",
        "4": "llm_chat"  # Fallback to LLM chat for generic REST
    }
    
    template_type = template_mapping[api_choice]
    
    # Use the custom template generator with the selected type
    from .custom_template_generator import CustomTemplateGenerator
    generator = CustomTemplateGenerator()
    
    # Generate template using the selected type
    template = generator.templates[template_type]
    
    # For guided setup, use a generic endpoint since we don't always have one
    endpoint = "https://api.example.com/v1/chat" if template_type == "llm_chat" else "https://api.example.com/v1/agent"
    
    config = template.generate_config(endpoint, api_key_env, description)
    test_cases = template.generate_test_cases(description)
    evaluator_code = template.generate_evaluator()
    
    # Show what we're generating
    console.print("")
    console.print(f"📋 [bold]Generating {template_type.replace('_', ' ').title()} Template[/bold]")
    console.print("")
    console.print("Files to be created:")
    console.print("  • optimization.yaml - Main configuration")
    console.print("  • test_cases.json - Test cases for your API")
    console.print("  • evaluator.py - Custom scoring logic")
    console.print("  • README.md - Setup instructions")
    console.print("")
    console.print(f"API key env var: {api_key_env}")
    console.print(f"Description: {description}")
    console.print("")
    
    if Confirm.ask("Create these files?"):
        return await generator._save_template_files(config, test_cases, evaluator_code, output_dir, template_type)
    else:
        console.print("Setup cancelled.")
        return {
            'spec_path': 'cancelled',
            'config_path': None,
            'tests_path': None,
            'test_cases': [],
            'config': {},
            'elapsed': 0.0
        }


async def run_custom_template_setup(project_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """Run custom template setup using proven patterns."""
    from .custom_template_generator import CustomTemplateGenerator
    
    generator = CustomTemplateGenerator()
    return await generator.generate_custom_template(project_dir, output_dir)


async def _gather_config_preferences(console: Console, template: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gather user preferences for key configuration options.
    
    Returns dict with overrides to apply to the template.
    """
    import os
    overrides = {}
    
    # 1. API Key
    console.print("┌─ [bold cyan]API Authentication[/bold cyan] " + "─" * 38 + "┐")
    console.print("│")
    
    # Suggest API key env var based on template
    if template['id'] == 'openai':
        default_key_env = "OPENAI_API_KEY"
        console.print("│ [dim]Your API calls will be authenticated using an environment[/dim]")
        console.print("│ [dim]variable. Get your key from: platform.openai.com/api-keys[/dim]")
    elif template['id'] == 'browserbase':
        default_key_env = "BROWSERBASE_API_KEY"
        console.print("│ [dim]BrowserBase requires an API key.[/dim]")
        console.print("│ [dim]Get yours from: browserbase.com[/dim]")
    elif template['id'] == 'groq':
        default_key_env = "GROQ_API_KEY"
        console.print("│ [dim]Groq provides fast inference.[/dim]")
        console.print("│ [dim]Get your key from: console.groq.com[/dim]")
    elif template['id'] == 'azure':
        default_key_env = "AZURE_OPENAI_API_KEY"
        console.print("│ [dim]Azure-hosted OpenAI models.[/dim]")
        console.print("│ [dim]Get your key from: portal.azure.com[/dim]")
    else:
        default_key_env = "API_KEY"
    
    console.print("│")
    console.print("│ [bold yellow]⚠️  Enter the ENVIRONMENT VARIABLE NAME[/bold yellow]")
    console.print(f"│ [bold yellow]    (e.g., {default_key_env}) - NOT the actual key![/bold yellow]")
    console.print("│")
    console.print("└" + "─" * 59 + "┘")
    console.print("")
    
    key_env = Prompt.ask(
        "Environment variable name",
        default=default_key_env
    )
    
    # Validate that they didn't paste an actual API key
    if _looks_like_api_key(key_env):
        console.print("")
        console.print(f"[bold red]❌ ERROR: You entered what looks like an actual API key![/bold red]")
        console.print(f"[yellow]   Please enter the ENVIRONMENT VARIABLE NAME, not the key value.[/yellow]")
        console.print(f"[dim]   Example: {default_key_env}[/dim]")
        console.print("")
        key_env = Prompt.ask(
            "Environment variable name",
            default=default_key_env
        )
    
    # Check if set
    if not os.getenv(key_env):
        console.print(f"[yellow]⚠️  {key_env} is not currently set in your environment[/yellow]")
        console.print(f"[dim]   Before running optimization, set it with:[/dim]")
        console.print(f"[cyan]   export {key_env}='your-actual-key-here'[/cyan]")
    else:
        console.print(f"[green]✅ {key_env} is already set and ready to use[/green]")
    
    overrides['api_key_env'] = key_env
    console.print("")
    console.print("")
    
    # 2. Optimization intensity
    console.print("┌─ [bold cyan]Optimization Intensity[/bold cyan] " + "─" * 35 + "┐")
    console.print("│")
    console.print("│ [dim]How thorough should the search be?[/dim]")
    console.print("│")
    console.print("│ [cyan]quick[/cyan]:     ~12 API calls │  1-2 min   │  Fast demo ⭐")
    console.print("│ [cyan]balanced[/cyan]:  ~18 API calls │  2-3 min   │  Good balance")
    console.print("│ [cyan]thorough[/cyan]:  ~48 API calls │  5-8 min   │  Best quality")
    console.print("│")
    console.print("│ [dim]More calls = better configurations found[/dim]")
    console.print("│ [dim]You can run multiple times to improve further[/dim]")
    console.print("│")
    console.print("└" + "─" * 59 + "┘")
    console.print("")
    
    intensity = Prompt.ask(
        "Intensity level",
        choices=["quick", "balanced", "thorough"],
        default="balanced"
    )
    
    # Map intensity to config values
    intensity_map = {
        "quick": {
            "experiments_per_generation": 2,
            "population_size": 2,
            "generations": 2,
            "parallel_workers": 1
        },
        "balanced": {
            "experiments_per_generation": 2,
            "population_size": 3,
            "generations": 2,
            "parallel_workers": 1
        },
        "thorough": {
            "experiments_per_generation": 3,
            "population_size": 4,
            "generations": 3,
            "parallel_workers": 2
        }
    }
    
    overrides['optimization'] = intensity_map[intensity]
    console.print(f"   [green]→ ~{_estimate_api_calls(intensity_map[intensity])} API calls per run[/green]")
    console.print("")
    
    # 3. Parallel workers
    console.print("3️⃣  [bold cyan]Parallel Processing[/bold cyan]")
    console.print("")
    console.print("   Run multiple API calls in parallel?")
    console.print("   [dim]More parallel = faster but may hit rate limits[/dim]")
    console.print("")
    
    parallel = Prompt.ask(
        "   Parallel workers",
        default=str(overrides['optimization']['parallel_workers'])
    )
    
    overrides['optimization']['parallel_workers'] = int(parallel)
    console.print("")
    
    # 4. Output path
    console.print("4️⃣  [bold cyan]Results Output[/bold cyan]")
    console.print("")
    
    default_output = f"./results/{template['id']}_optimization"
    output_path = Prompt.ask(
        "   Save results to",
        default=default_output
    )
    
    overrides['output_path'] = output_path
    console.print("")
    
    # 5. Metrics priority (optional - only if they want to customize)
    console.print("5️⃣  [bold cyan]Optimization Goal (Optional)[/bold cyan]")
    console.print("")
    console.print("   What matters most?")
    console.print("   [dim]balanced: equal weight | quality: best results | speed: fast responses | cost: cheapest[/dim]")
    console.print("")
    
    goal = Prompt.ask(
        "   Priority",
        choices=["balanced", "quality", "speed", "cost"],
        default="balanced"
    )
    
    # Map to metric weights
    goal_map = {
        "balanced": {
            "response_quality": 0.40,
            "latency_ms": 0.25,
            "cost_per_call": 0.20,
            "token_efficiency": 0.15
        },
        "quality": {
            "response_quality": 0.60,
            "latency_ms": 0.15,
            "cost_per_call": 0.10,
            "token_efficiency": 0.15
        },
        "speed": {
            "response_quality": 0.30,
            "latency_ms": 0.50,
            "cost_per_call": 0.10,
            "token_efficiency": 0.10
        },
        "cost": {
            "response_quality": 0.30,
            "latency_ms": 0.10,
            "cost_per_call": 0.45,
            "token_efficiency": 0.15
        }
    }
    
    overrides['metric_weights'] = goal_map[goal]
    console.print("")
    
    # 6. Legacy System (continuous learning)
    console.print("6️⃣  [bold cyan]Legacy System (Continuous Learning)[/bold cyan]")
    console.print("")
    console.print("   📚 The Legacy System enables continuous learning across optimization runs.")
    console.print("   [dim]• Saves winning configurations for future runs[/dim]")
    console.print("   [dim]• Starts new optimizations from proven winners[/dim]")
    console.print("   [dim]• Gets better over time with each run[/dim]")
    console.print("   [dim]• Stores data locally in SQLite database[/dim]")
    console.print("")
    console.print("   [green]✅ Recommended: Keep enabled for best results[/green]")
    console.print("")
    
    legacy_enabled = Prompt.ask(
        "   Enable Legacy System?",
        choices=["y", "n"],
        default="y"
    ) == "y"
    
    overrides['legacy_enabled'] = legacy_enabled
    if legacy_enabled:
        console.print("   [green]→ Legacy System enabled - your optimizations will improve over time![/green]")
    else:
        console.print("   [yellow]→ Legacy System disabled - each run starts fresh[/yellow]")
    console.print("")
    
    return overrides


def _estimate_api_calls(optimization_config: Dict[str, Any]) -> int:
    """Estimate total API calls for a given optimization config."""
    experiments = optimization_config.get('experiments_per_generation', 3)
    population = optimization_config.get('population_size', 4)
    generations = optimization_config.get('generations', 3)
    
    # Assume 4 test cases on average
    test_cases = 4
    
    # MAB phase + Evolution phases
    mab_calls = experiments * test_cases
    evolution_calls = population * generations * test_cases
    
    return mab_calls + evolution_calls


def _looks_like_api_key(value: str) -> bool:
    """
    Check if a string looks like an actual API key rather than an environment variable name.
    
    Common API key patterns:
    - OpenAI: sk-...
    - Anthropic: sk-ant-...
    - Google/Gemini: AIza...
    - BrowserBase: bb_live_... or bb_test_...
    - Contains long alphanumeric strings
    - Very long (> 30 chars usually means it's a key)
    """
    if not value:
        return False
    
    # Check for common API key prefixes
    key_prefixes = ['sk-', 'sk_', 'AIza', 'bb_live_', 'bb_test_', 'api_', 'key_']
    for prefix in key_prefixes:
        if value.startswith(prefix):
            return True
    
    # Check if it's very long (likely a key)
    if len(value) > 30:
        return True
    
    # Check if it contains typical key characters (lots of numbers + letters mixed)
    import re
    # If it has many alternating letters and numbers, probably a key
    alternations = len(re.findall(r'[a-zA-Z][0-9]|[0-9][a-zA-Z]', value))
    if alternations > 5:
        return True
    
    return False
