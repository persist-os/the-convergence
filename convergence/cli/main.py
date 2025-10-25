"""
Command-line interface for The Convergence API Optimization Framework.
"""

from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import asyncio
import json
import csv
import os
import logging
import datetime

# Configure logging - suppress noisy libraries and set baseline
# Set root logger to INFO to prevent DEBUG logs from any library
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Suppress specific noisy libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("aiosqlite").setLevel(logging.WARNING)
logging.getLogger("sqlite3").setLevel(logging.WARNING)
logging.getLogger("weave").setLevel(logging.WARNING)
logging.getLogger("wandb").setLevel(logging.WARNING)
logging.getLogger("convergence.storage").setLevel(logging.INFO)
logging.getLogger("convergence.storage.sqlite_backend").setLevel(logging.INFO)
logging.getLogger("convergence.storage.file_backend").setLevel(logging.INFO)

# Load environment variables from .env file FIRST
try:
    from dotenv import load_dotenv
    # Try loading from multiple locations
    env_loaded = False
    env_paths = [
        Path.cwd() / ".env",  # Current directory
        Path.home() / ".env",  # Home directory
        Path(__file__).parent.parent.parent / ".env",  # Project root
    ]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path, override=True)
            env_loaded = True
            break
    if not env_loaded:
        # Try without explicit path (searches up directory tree)
        load_dotenv(override=True)
except ImportError:
    # python-dotenv not installed, will use system env vars only
    pass

# Initialize console and logger first (used throughout module)
console = Console()
logger = logging.getLogger(__name__)

# Initialize Weave if organization is set
try:
    import weave
    weave_org = os.getenv("WEAVE_ORGANIZATION") or os.getenv("WANDB_ENTITY")
    weave_project = os.getenv("WEAVE_PROJECT") or os.getenv("WANDB_PROJECT", "convergence")
    
    if weave_org:
        weave.init(f"{weave_org}/{weave_project}")
        console.print(f"✅ Weave initialized: {weave_org}/{weave_project}", style="green")
except Exception as e:
    # Weave not available or failed to init, continue without it
    pass

app = typer.Typer(
    name="convergence",
    help="The Convergence: API Optimization Framework powered by agent society"
)

# Import optimization components
try:
    from convergence.optimization.config_loader import ConfigLoader
    from convergence.optimization.runner import OptimizationRunner
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

# Import generator for init command
try:
    from convergence.generator import initialize_project
    GENERATOR_AVAILABLE = True
except ImportError:
    GENERATOR_AVAILABLE = False

# Import Weave for observability
try:
    import weave
    WEAVE_AVAILABLE = True
except ImportError:
    WEAVE_AVAILABLE = False
    weave = None


def initialize_weave(config) -> bool:
    """
    Initialize Weave observability at CLI level.
    
    Returns:
        True if Weave was initialized successfully, False otherwise
    """
    if not WEAVE_AVAILABLE:
        return False
    
    # Check if Weave is enabled in config
    weave_enabled = False
    weave_org = None
    weave_project = None
    
    if config.society and config.society.enabled:
        weave_config = config.society.weave
        weave_enabled = weave_config.enabled
        weave_org = weave_config.organization
        weave_project = weave_config.project
    
    if not weave_enabled:
        return False
    
    # Read from environment if not specified
    weave_org = weave_org or os.getenv("WEAVE_ORGANIZATION") or os.getenv("WANDB_ENTITY")
    weave_project = weave_project or os.getenv("WEAVE_PROJECT") or os.getenv("WANDB_PROJECT", "convergence-optimization")
    
    if not weave_org:
        console.print("[yellow]⚠️  Weave enabled but no organization specified.[/yellow]")
        console.print("[yellow]   Set WANDB_ENTITY or WEAVE_ORGANIZATION environment variable.[/yellow]")
        console.print("[yellow]   Continuing without Weave tracing...[/yellow]\n")
        return False
    
    try:
        # Initialize Weave - THIS IS THE CRITICAL PART
        weave_project_path = f"{weave_org}/{weave_project}"
        weave.init(weave_project_path)
        
        console.print(f"\n[green]✅ Weave initialized: {weave_project_path}[/green]")
        console.print("[cyan]   ─────────────────────────────────────────────────────[/cyan]")
        console.print("[cyan]   📊 LLM tracing & observability powered by Weave[/cyan]")
        console.print("[cyan]   All API calls and evaluations will be tracked[/cyan]")
        console.print("[cyan]   Real-time tracking of experiments & metrics[/cyan]\n")
        return True
        
    except Exception as e:
        console.print(f"[yellow]⚠️  Weave initialization failed: {e}[/yellow]")
        console.print(f"[yellow]   Make sure '{weave_org}' is a valid W&B entity/organization[/yellow]")
        console.print("[yellow]   Continuing without tracing...[/yellow]\n")
        return False


@app.command()
def init(
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory (default: current directory)"
    ),
):
    """
    Initialize Convergence project with interactive setup.
    
    Creates:
    - optimization.yaml (config)
    - test_cases.json (test cases)
    - evaluator.py (if needed)
    
    Example:
        convergence init                    # Interactive setup
        convergence init --output ./config  # Custom output dir
    """
    if not GENERATOR_AVAILABLE:
        console.print("[red]Error:[/red] Generator not available")
        console.print("Install dependencies: pip install -e .")
        raise typer.Exit(1)
    
    try:
        # Run async initialization
        result = asyncio.run(initialize_project(
            project_dir=Path.cwd(),
            output_dir=output_dir or Path.cwd()
        ))
        
        # Success summary (only if config was created)
        if result.get('config_path'):
            console.print("")
            console.print("─" * 60)
            console.print("")
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Setup cancelled[/yellow]")
        raise typer.Exit(0)
    
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        logger.exception("Initialization failed")
        raise typer.Exit(1)


def _check_and_prompt_api_keys(config) -> None:
    """
    Check for required API keys and prompt user if missing.
    Optionally auto-export env variables if user provides them.
    """
    missing_keys = []
    
    # Check main API auth
    if config.api.auth and config.api.auth.token_env:
        env_var = config.api.auth.token_env
        if not os.getenv(env_var):
            missing_keys.append(("Main API", env_var, config.api.name))
    
    # Check agent society LLM (if enabled)
    if config.society and config.society.enabled:
        if hasattr(config.society, 'llm') and config.society.llm:
            env_var = config.society.llm.api_key_env
            if not os.getenv(env_var):
                missing_keys.append(("Agent Society LLM", env_var, config.society.llm.model))
    
    if not missing_keys:
        return  # All keys present
    
    # Display missing keys
    console.print("\n[yellow]⚠️  Missing API Keys[/yellow]")
    console.print("The following environment variables need to be set:\n")
    
    for purpose, env_var, detail in missing_keys:
        console.print(f"  • [bold]{env_var}[/bold]")
        console.print(f"    Purpose: {purpose} ({detail})")
    
    console.print("\n[cyan]Options:[/cyan]")
    console.print("  1. Export manually: [dim]export OPENAI_API_KEY=\"sk-...\"[/dim]")
    console.print("  2. Use .env file:   [dim]echo \"OPENAI_API_KEY=sk-...\" > .env[/dim]")
    console.print("  3. Enter now to auto-export (session only)\n")
    
    # Ask if user wants to provide keys now
    provide_now = typer.confirm("Would you like to provide the API keys now?", default=False)
    
    if provide_now:
        for purpose, env_var, detail in missing_keys:
            console.print(f"\n[bold]{env_var}[/bold] ({purpose}):")
            api_key = typer.prompt(f"  Enter value", hide_input=True)
            
            if api_key and api_key.strip():
                # Set in current process
                os.environ[env_var] = api_key.strip()
                console.print(f"  [green]✓[/green] Set {env_var} for this session")
            else:
                console.print(f"  [yellow]⊗[/yellow] Skipped {env_var}")
        
        console.print("\n[green]✓[/green] Environment variables configured for this session")
        console.print("[dim]Tip: Add to .env file to persist across sessions[/dim]\n")
    else:
        console.print("\n[yellow]Please set the required environment variables and try again.[/yellow]")
        console.print("\nExample:")
        console.print(f"  [dim]export {missing_keys[0][1]}=\"your-key-here\"[/dim]")
        console.print(f"  [dim]convergence optimize {config.api.name}_optimization.yaml[/dim]\n")
        raise typer.Exit(1)


@app.command()
def optimize(
    config_file: Path = typer.Argument(..., help="Path to optimization config (YAML/JSON)"),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for results"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed progress"
    ),
):
    """
    Optimize API configuration using MAB + Evolution + RL.
    
    Example:
        convergence optimize optimization.yaml
        convergence optimize browserbase.yaml --output ./results/
    
    Under the hood: An agent society (with RLP, SAO, memory, MAB) works
    together to find optimal configurations. You just see the results.
    """
    if not OPTIMIZATION_AVAILABLE:
        console.print("[red]Error:[/red] Optimization components not available")
        console.print("Install with: pip install the-convergence")
        raise typer.Exit(1)
    
    try:
        # Validate config file exists
        if not config_file.exists():
            console.print(f"[red]Error:[/red] Config file not found: {config_file}")
            raise typer.Exit(1)
        
        console.print(Panel(
            f"[bold cyan]Optimizing API configuration:[/bold cyan]\n{config_file}",
            title="🚀 The Convergence",
            border_style="cyan"
        ))
        
        # Load config
        console.print("[cyan]Loading configuration...[/cyan]")
        config = ConfigLoader.load(config_file)
        
        # Check and prompt for required API keys
        _check_and_prompt_api_keys(config)
        
        # Initialize Weave FIRST - before any optimization starts
        initialize_weave(config)
        
        console.print(f"[green]✓[/green] Loaded config: [bold]{config.api.name}[/bold]")
        if config.api.endpoint:
            console.print(f"  - Endpoint: {config.api.endpoint}")
        elif config.api.models:
            console.print(f"  - Models: {', '.join(config.api.models.keys())}")
        console.print(f"  - Parameters: {len(config.search_space.parameters)}")
        console.print(f"  - Metrics: {len(config.evaluation.metrics)}")
        console.print(f"  - Generations: {config.optimization.evolution.generations}")
        
        # Show legacy system status (always visible since it's enabled by default)
        if config.legacy.enabled:
            console.print(f"  - 📚 Legacy System: [green]Enabled[/green] (continuous learning)")
        else:
            console.print(f"  - 📚 Legacy System: [yellow]Disabled[/yellow]")
        
        # Show society config if enabled (advanced users)
        if config.society and config.society.enabled and verbose:
            console.print("\n[dim]🤖 Agent Society Active:[/dim]")
            console.print(f"[dim]  - RLP (Reasoning): {config.society.learning.rlp_enabled}[/dim]")
            console.print(f"[dim]  - SAO (Self-Alignment): {config.society.learning.sao_enabled}[/dim]")
            console.print(f"[dim]  - Collaboration: {config.society.collaboration.enabled}[/dim]")
        
        # Create runner (pass config file path for local evaluator loading)
        runner = OptimizationRunner(config, config_file_path=config_file)
        
        # Run optimization
        console.print("\n[cyan]Starting optimization...[/cyan]")
        console.print("[dim]Agent society is optimizing in the background...[/dim]\n")
        
        # Run async optimization
        result = asyncio.run(runner.run())
        
        # Determine output directory
        if output_dir is None:
            output_dir = Path(config.output.save_path)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results in requested formats
        formats = config.output.formats
        
        console.print(f"\n[cyan]Saving results to {output_dir}...[/cyan]")
        
        # Save JSON
        if "json" in formats:
            json_path = output_dir / "best_config.json"
            
            # Add helpful metadata for non-technical users
            best_config_with_metadata = {
                "_README": {
                    "description": "Best AI configuration found in this optimization",
                    "how_to_use": "Copy the 'config' section below to your Groq API calls",
                    "score_achieved": result.best_score,
                    "score_scale": "0.0 (worst) to 1.0 (perfect)",
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                    "optimization_run": "Latest",
                    "test_cases_evaluated": len(result.all_results[0].get("test_results", [])) if result.all_results else 0
                },
                "config": result.best_config
            }
            
            with open(json_path, 'w') as f:
                json.dump(best_config_with_metadata, f, indent=2)
            console.print(f"[green]✓[/green] Saved: {json_path}")
            
            # Save detailed results with responses
            detailed_json_path = output_dir / "detailed_results.json"
            with open(detailed_json_path, 'w') as f:
                json.dump(result.all_results, f, indent=2, default=str)
            console.print(f"[green]✓[/green] Saved: {detailed_json_path}")
        
        # Save Markdown report
        if "markdown" in formats:
            md_path = output_dir / "report.md"
            _generate_markdown_report(result, config, md_path)
            console.print(f"[green]✓[/green] Saved: {md_path}")
        
        # Save CSV
        if "csv" in formats:
            csv_path = output_dir / "experiments.csv"
            _generate_csv_report(result, csv_path)
            console.print(f"[green]✓[/green] Saved: {csv_path}")
            
            # Save detailed per-test-case CSV
            detailed_csv_path = output_dir / "detailed_results.csv"
            _generate_detailed_csv_report(result, detailed_csv_path)
            console.print(f"[green]✓[/green] Saved: {detailed_csv_path}")
        
        # Create README for non-technical users
        readme_path = output_dir / "README.md"
        _generate_results_readme(result, config, readme_path)
        console.print(f"[green]✓[/green] Saved: {readme_path}")
        
        # Format best config parameters
        config_str = ", ".join([f"{k}={v}" for k, v in result.best_config.items()])
        
        # Calculate average cost if available
        total_cost = 0.0
        cost_count = 0
        for exp in result.all_results:
            if 'cost_usd' in exp.get('result', {}):
                total_cost += exp['result']['cost_usd']
                cost_count += 1
        avg_cost = total_cost / cost_count if cost_count > 0 else None
        
        # Build output
        output_lines = [
            f"[bold green]✅ Optimization Complete![/bold green]",
            f"   Best config: [cyan]{config_str}[/cyan]",
            f"   Score: [bold]{result.best_score:.2f}[/bold]"
        ]
        
        if avg_cost:
            output_lines.append(f"   Cost: [yellow]${avg_cost:.4f}/call[/yellow]")
        
        output_lines.append(f"   ")
        output_lines.append(f"   Results saved to: [cyan]{output_dir}[/cyan]")
        output_lines.append(f"   [dim]Generations: {result.generations_run} | Experiments: {len(result.all_results)}[/dim]")
        
        console.print("\n".join(output_lines))
    
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


def _generate_markdown_report(result, config, output_path: Path):
    """Generate Markdown report."""
    report = f"""# API Optimization Report

## 📖 Quick Guide for Non-Technical Users

### What Is This Report?

This report shows the results of testing different AI settings (configurations) to find the optimal combination of **quality**, **speed**, and **cost**.

### Key Terms Explained

| Term | What It Means | Example |
|------|---------------|---------|
| **Configuration** | A specific set of AI settings | `{{"model": "llama-3.1-8b", "temperature": 1.2}}` |
| **Score** | How well the AI performed (0.0 = failed, 1.0 = perfect) | `0.9307` means 93% excellent |
| **Generation** | One round of testing and improvement | Generation 0, 1, 2... |
| **Test Case** | A specific task we tested | creative_story, factual_qa |
| **Latency** | How fast the AI responded (in milliseconds) | `400ms` = 0.4 seconds |
| **Cost** | How much this API call cost in US dollars | `$0.004` = less than half a cent |

### Score Interpretation

- **0.90 - 1.00**: Excellent ⭐⭐⭐ (Use this!)
- **0.80 - 0.90**: Good ⭐⭐ (Acceptable)
- **0.70 - 0.80**: Fair ⭐ (Could be better)
- **Below 0.70**: Poor ❌ (Keep optimizing)

### How to Use This Report

1. Read "Best Configuration" section for the winning settings
2. Check the "Overall Score" - higher is better
3. Review "Sample Responses" to see actual AI outputs
4. Copy the best config to your code and start using it!

---

## Configuration
- **API**: {config.api.name}
- **Endpoint**: {config.api.endpoint or 'model-registry'}
- **Parameters Optimized**: {len(config.search_space.parameters)}

## Results
- **Best Score**: {result.best_score:.4f}
- **Generations Run**: {result.generations_run}
- **Total Experiments**: {len(result.all_results)}
- **Timestamp**: {result.timestamp}

## Best Configuration
```json
{json.dumps(result.best_config, indent=2)}
```

## Detailed Results by Configuration

"""
    
    # Add detailed breakdown for each configuration
    for idx, entry in enumerate(result.all_results, 1):
        config_params = entry.get('config', {})
        overall_score = entry.get('score', 0)
        
        report += f"### Configuration {idx}\n\n"
        report += f"**Parameters**: `{json.dumps(config_params)}`\n\n"
        report += f"**Overall Score**: {overall_score:.4f}\n\n"
        
        # Add per-test-case results if available
        test_results = entry.get('test_results', [])
        if test_results:
            report += "#### Test Case Results\n\n"
            report += "| Test Case | Score | Latency (ms) | Cost ($) | Status |\n"
            report += "|-----------|-------|--------------|----------|--------|\n"
            
            for test_result in test_results:
                test_id = test_result.get('test_case_id', 'unknown')
                score = test_result.get('score', 0)
                latency = test_result.get('latency_ms', 0)
                cost = test_result.get('cost_usd', 0)
                success = "✅" if test_result.get('success', False) else "❌"
                
                report += f"| {test_id} | {score:.4f} | {latency:.1f} | ${cost:.6f} | {success} |\n"
            
            report += "\n"
            
            # Add sample responses
            report += "##### Sample Responses\n\n"
            for test_result in test_results[:3]:  # Show first 3
                test_id = test_result.get('test_case_id', 'unknown')
                response_text = test_result.get('response_text', '')
                
                if response_text:
                    # Truncate long responses
                    display_text = response_text[:200] + "..." if len(response_text) > 200 else response_text
                    report += f"**{test_id}**: {display_text}\n\n"
            
            report += "\n"
        
        report += "---\n\n"
    
    report += "\n## Summary\n\n"
    report += "*Optimized by The Convergence agent society (MAB + Evolution + RL)*\n"
    
    with open(output_path, 'w') as f:
        f.write(report)


def _generate_csv_report(result, output_path: Path):
    """Generate CSV report."""
    with open(output_path, 'w', newline='') as f:
        if not result.all_results:
            return
        
        # Get all field names
        fieldnames = ["generation", "score"]
        if result.all_results:
            first_config = result.all_results[0].get("config", {})
            fieldnames.extend(first_config.keys())
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for entry in result.all_results:
            row = {
                "generation": entry.get("generation", 0),
                "score": entry.get("score", 0),
            }
            row.update(entry.get("config", {}))
            writer.writerow(row)


def _generate_detailed_csv_report(result, output_path: Path):
    """Generate detailed per-test-case CSV report."""
    with open(output_path, 'w', newline='') as f:
        if not result.all_results:
            return
        
        # Get config parameter names
        config_params = []
        if result.all_results:
            first_config = result.all_results[0].get("config", {})
            config_params = list(first_config.keys())
        
        # CSV headers
        fieldnames = ["generation", "test_case_id"] + config_params + [
            "score", "latency_ms", "cost_usd", "success"
        ]
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write per-test-case results
        for entry in result.all_results:
            generation = entry.get("generation", 0)
            config = entry.get("config", {})
            test_results = entry.get("test_results", [])
            
            for test_result in test_results:
                row = {
                    "generation": generation,
                    "test_case_id": test_result.get("test_case_id", "unknown"),
                    "score": test_result.get("score", 0),
                    "latency_ms": test_result.get("latency_ms", 0),
                    "cost_usd": test_result.get("cost_usd", 0),
                    "success": test_result.get("success", False)
                }
                # Add config params
                row.update(config)
                writer.writerow(row)


def _generate_results_readme(result, config, output_path: Path):
    """Generate README for results folder to help non-technical users."""
    # Calculate score rating
    score = result.best_score
    if score >= 0.9:
        rating = "⭐⭐⭐ Excellent"
    elif score >= 0.8:
        rating = "⭐⭐ Good"
    elif score >= 0.7:
        rating = "⭐ Fair"
    else:
        rating = "❌ Needs Improvement"
    
    readme_content = f"""# {config.api.name.title()} Optimization Results

**Generated**: {datetime.datetime.utcnow().strftime('%B %d, %Y at %I:%M %p UTC')}  
**Optimization Run**: Latest  
**Best Score**: {result.best_score:.4f} ({rating})

---

## 📁 What's in This Folder?

This folder contains the results from your **most recent** {config.api.name} API optimization run.

| File | What It Is | Who Should Read It |
|------|------------|-------------------|
| `README.md` | This file - explains everything | Everyone (start here!) |
| `best_config.json` | The winning AI settings | **Everyone - use these!** |
| `report.md` | Human-readable detailed summary | Everyone |
| `experiments.csv` | Spreadsheet of all configurations tested | Data analysts |
| `detailed_results.json` | Complete raw data (1000+ lines) | Developers only |
| `detailed_results.csv` | Per-test-case breakdown | Data analysts |

---

## 🚀 Quick Start (5 minutes)

### Step 1: Read the Report (2 min)

Open `report.md` to see:
- Which settings won
- What the AI actually said
- How fast and cheap each option was

### Step 2: Use the Best Config (3 min)

Open `best_config.json` and copy the settings to your code.

**Example**:
```json
{json.dumps(result.best_config, indent=2)}
```

---

## 🗄️ Understanding the Data Storage

This system stores data in **TWO places**:

### 1. This Folder (Latest Run Only)
- **Location**: `{config.output.save_path}/`
- **Updates**: Overwritten on each optimization run
- **Purpose**: Quick access to latest results

### 2. Legacy Database (All History)
- **Location**: `{config.legacy.sqlite_path if config.legacy and config.legacy.enabled else 'N/A (legacy disabled)'}`
- **Updates**: Accumulates forever (never overwritten)
- **Purpose**: Long-term learning and warm-start

**Why both?** 
- Folder = Easy to read (humans)
- Database = Easy to query (programs + future runs)

---

## 🔄 What is "Warm-Start"?

**Simple Explanation**:
When you run optimization again, the system:
1. Loads winning configs from previous runs (from database)
2. Tests them again + explores new options
3. Often finds even better settings faster!

**Analogy**: Like starting a game from your last checkpoint instead of level 1.

---

## 📊 Your Results Summary

- **Configurations Tested**: {len(result.all_results)}
- **Best Score**: {result.best_score:.4f}
- **Rating**: {rating}

### Best Configuration:
```json
{json.dumps(result.best_config, indent=2)}
```

---

## 🤔 Common Questions

### Q: Which file should I use?
**A**: Start with `report.md` (human-readable), then use `best_config.json` (settings to copy).

### Q: Why do scores in warm-start not match final scores?
**A**: Warm-start shows each config's **best performance on one specific test**, while final scores show **average across all tests**. Think of it like a student's best subject grade vs. their GPA.

### Q: Can I delete this folder?
**A**: Yes! It's regenerated on each run. The historical data is safe in the database.

### Q: How do I see previous runs?
**A**: Query the database:
```bash
sqlite3 {config.legacy.sqlite_path if config.legacy and config.legacy.enabled else './data/legacy.db'} "SELECT * FROM runs ORDER BY timestamp DESC LIMIT 10;"
```

### Q: What if I want to start fresh?
**A**: Delete the database to reset:
```bash
rm -f {config.legacy.sqlite_path if config.legacy and config.legacy.enabled else './data/legacy.db'}
```

---

## 🆘 Need Help?

- **Don't understand a term?** See glossary in `report.md`
- **Want more details?** Check `detailed_results.json`
- **Want historical data?** Query the database
- **Want to optimize more?** Run the command again!

---

**Generated by The Convergence** - API Optimization Framework  
**Session**: Generation {result.generations_run}
"""
    
    with open(output_path, 'w') as f:
        f.write(readme_content)


@app.command()
def info():
    """Show information about The Convergence framework."""
    console.print(Panel(
        f"[bold cyan]The Convergence[/bold cyan]\n"
        f"API Optimization Framework\n\n"
        f"[bold]What it does:[/bold]\n"
        f"Finds optimal API configurations through evolutionary optimization.\n\n"
        f"[bold]How it works:[/bold]\n"
        f"• Multi-Armed Bandits (MAB) - Smart exploration\n"
        f"• Evolutionary Algorithms - Breed winning configs\n"
        f"• RL Meta-Optimization - Learn from history\n"
        f"• Agent Society - Parallel optimization (RLP + SAO + Memory)\n\n"
        f"[bold]Usage:[/bold]\n"
        f"  convergence optimize config.yaml\n\n"
        f"[bold]Docs:[/bold] github.com/persist-os/the-convergence",
        title="📚 About",
        border_style="cyan"
    ))


@app.command()
def version():
    """Show version information."""
    from convergence import __version__
    console.print(f"[bold cyan]The Convergence[/bold cyan] v{__version__}")
    console.print("API Optimization Framework")


if __name__ == "__main__":
    app()
