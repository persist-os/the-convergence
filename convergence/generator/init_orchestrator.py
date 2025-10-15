"""
Orchestrate convergence init (interactive setup).

Minimal code: delegates to interactive setup module.
"""
from pathlib import Path
from typing import Optional, Dict, Any


async def initialize_project(
    project_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Initialize Convergence project with interactive setup.
    
    This is the main entry point for `convergence init`.
    
    Args:
        project_dir: Project directory (default: current dir)
        output_dir: Where to write configs (default: project_dir)
    
    Returns:
        Dict with paths to generated files
    
    Example:
        result = await initialize_project(
            project_dir=Path.cwd(),
            output_dir=Path(".")
        )
        print(f"Created: {result['config_path']}")
    """
    # Setup paths
    project_dir = project_dir or Path.cwd()
    output_dir = output_dir or project_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run interactive setup
    from .interactive_setup import run_interactive_setup
    return await run_interactive_setup(project_dir, output_dir)
