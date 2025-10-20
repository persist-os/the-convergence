"""
Custom Template Generator

Main generator class that orchestrates custom template creation using proven patterns.
"""
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from rich.console import Console
from rich.prompt import Prompt, Confirm

from .template_selector import TemplateSelector
from .templates.llm_chat_template import LLMChatTemplate
from .templates.agno_agent_template import AgnoAgentTemplate
from .templates.web_automation_template import WebAutomationTemplate


class CustomTemplateGenerator:
    """Generate custom templates using proven patterns."""
    
    def __init__(self):
        self.console = Console()
        self.template_selector = TemplateSelector()
        self.templates = {
            'llm_chat': LLMChatTemplate(),
            'agno_agent': AgnoAgentTemplate(),
            'web_automation': WebAutomationTemplate()
        }
    
    async def generate_custom_template(
        self,
        project_dir: Path,
        output_dir: Path
    ) -> Dict[str, Any]:
        """Generate custom template using proven patterns."""
        
        self.console.print("\n🎯 [bold cyan]Custom Template Builder[/bold cyan]")
        self.console.print("Choose from proven template patterns:")
        self.console.print("")
        
        templates = self.template_selector.get_available_templates()
        for i, template in enumerate(templates, 1):
            self.console.print(f"  {i}. {template['name']}")
            self.console.print(f"     {template['description']}")
            self.console.print(f"     Examples: {', '.join(template['examples'])}")
            self.console.print("")
        
        choice = Prompt.ask("Select template type", choices=[str(i) for i in range(1, len(templates) + 1)])
        template_type = templates[int(choice) - 1]['id']
        
        # Get basic info
        self.console.print("")
        endpoint = Prompt.ask("API endpoint")
        api_key_env = Prompt.ask("API key env var", default="API_KEY")
        description = Prompt.ask("What does your API do? (helps generate test cases)")
        
        # Generate using selected template
        template = self.templates[template_type]
        
        config = template.generate_config(endpoint, api_key_env, description)
        test_cases = template.generate_test_cases(description)
        evaluator_code = template.generate_evaluator()
        
        # Preview and save
        self._preview_generated_template(config, test_cases, template_type)
        if Confirm.ask("Looks good?"):
            return await self._save_template_files(config, test_cases, evaluator_code, output_dir, template_type)
        else:
            return await self._guided_customization(config, test_cases, evaluator_code, output_dir, template_type)
    
    def _preview_generated_template(self, config: Dict[str, Any], test_cases: List[Dict], template_type: str):
        """Preview the generated template."""
        self.console.print("")
        self.console.print("📋 [bold]Generated Template Preview[/bold]")
        self.console.print("")
        
        # Show config summary
        self.console.print(f"API: {config['api']['name']}")
        self.console.print(f"Endpoint: {config['api']['endpoint']}")
        self.console.print(f"Auth: {config['api']['auth']['type']} ({config['api']['auth']['token_env']})")
        self.console.print("")
        
        # Show parameters being optimized
        self.console.print("Parameters being optimized:")
        for param, details in config['search_space']['parameters'].items():
            if details['type'] == 'categorical':
                self.console.print(f"  • {param}: {', '.join(map(str, details['values']))}")
            elif details['type'] == 'continuous':
                self.console.print(f"  • {param}: {details['min']} to {details['max']} (step: {details['step']})")
            elif details['type'] == 'discrete':
                self.console.print(f"  • {param}: {', '.join(map(str, details['values']))}")
        self.console.print("")
        
        # Show test cases
        self.console.print(f"Test cases: {len(test_cases)} generated")
        for i, test_case in enumerate(test_cases[:3], 1):  # Show first 3
            self.console.print(f"  {i}. {test_case['description']}")
        if len(test_cases) > 3:
            self.console.print(f"  ... and {len(test_cases) - 3} more")
        self.console.print("")
        
        # Show metrics
        self.console.print("Evaluation metrics:")
        for metric, details in config['evaluation']['metrics'].items():
            self.console.print(f"  • {metric}: {details['weight']*100:.0f}% weight")
        self.console.print("")
        
        # Show output location
        self.console.print(f"Results will be saved to: {config['output']['save_path']}")
        self.console.print("")
    
    async def _save_template_files(
        self, 
        config: Dict[str, Any], 
        test_cases: List[Dict], 
        evaluator_code: str, 
        output_dir: Path,
        template_type: str
    ) -> Dict[str, Any]:
        """Save the generated template files."""
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate file contents
        template = self.templates[template_type]
        yaml_content = template.generate_yaml_content(config)
        json_content = template.generate_json_content(test_cases)
        readme_content = template.generate_readme_content(config)
        
        # Save files
        config_path = output_dir / "optimization.yaml"
        tests_path = output_dir / "test_cases.json"
        evaluator_path = output_dir / "evaluator.py"
        readme_path = output_dir / "README.md"
        
        config_path.write_text(yaml_content)
        tests_path.write_text(json_content)
        evaluator_path.write_text(evaluator_code)
        readme_path.write_text(readme_content)
        
        self.console.print("✅ [bold green]Template files generated successfully![/bold green]")
        self.console.print("")
        self.console.print("Files created:")
        self.console.print(f"  📄 {config_path}")
        self.console.print(f"  📄 {tests_path}")
        self.console.print(f"  📄 {evaluator_path}")
        self.console.print(f"  📄 {readme_path}")
        self.console.print("")
        self.console.print("Next steps:")
        self.console.print(f"  1. Set your API key: export {config['api']['auth']['token_env']}='your-key'")
        self.console.print("  2. Run optimization: convergence optimize optimization.yaml")
        self.console.print("")
        
        return {
            'spec_path': str(config_path),
            'config_path': str(config_path),
            'tests_path': str(tests_path),
            'test_cases': test_cases,
            'config': config,
            'elapsed': 0.0
        }
    
    async def _guided_customization(
        self, 
        config: Dict[str, Any], 
        test_cases: List[Dict], 
        evaluator_code: str, 
        output_dir: Path,
        template_type: str
    ) -> Dict[str, Any]:
        """Allow guided customization of the template."""
        
        self.console.print("")
        self.console.print("🔧 [bold yellow]Guided Customization[/bold yellow]")
        self.console.print("")
        self.console.print("⚠️  Guided customization implementation coming soon!")
        self.console.print("For now, the template will be saved as-is.")
        self.console.print("")
        
        if Confirm.ask("Save template anyway?"):
            return await self._save_template_files(config, test_cases, evaluator_code, output_dir, template_type)
        else:
            self.console.print("Template generation cancelled.")
            return {
                'spec_path': 'cancelled',
                'config_path': None,
                'tests_path': None,
                'test_cases': [],
                'config': {},
                'elapsed': 0.0
            }
    
    def get_template_info(self, template_type: str) -> Dict[str, str]:
        """Get information about a specific template type."""
        return self.template_selector.get_template_info(template_type)
    
    def list_available_templates(self) -> List[Dict[str, str]]:
        """Get list of available templates."""
        return self.template_selector.get_available_templates()


# Export for easy import
__all__ = ['CustomTemplateGenerator']
