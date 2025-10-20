"""
Template selector for custom template generation.

Provides template selection logic based on proven patterns from working examples.
"""
from typing import Dict, List, Any
from pathlib import Path


class TemplateSelector:
    """Select appropriate template based on user choice."""
    
    def __init__(self):
        self.templates = {
            'llm_chat': 'LLMChatTemplate',
            'agno_agent': 'AgnoAgentTemplate', 
            'web_automation': 'WebAutomationTemplate'
        }
    
    def select_template(self, template_type: str) -> str:
        """Get template class name by type."""
        if template_type not in self.templates:
            raise ValueError(f"Unknown template type: {template_type}")
        return self.templates[template_type]
    
    def get_available_templates(self) -> List[Dict[str, str]]:
        """Get list of available templates with descriptions."""
        return [
            {
                'id': 'llm_chat',
                'name': 'LLM Chat API',
                'description': 'OpenAI-style chat completion APIs (OpenAI, Anthropic, Groq, Azure)',
                'examples': ['OpenAI', 'Groq', 'Azure OpenAI', 'Anthropic']
            },
            {
                'id': 'agno_agent', 
                'name': 'Agent API',
                'description': 'AI agents with tools and reasoning (Agno, LangChain, custom agents)',
                'examples': ['Agno Reddit Agent', 'LangChain Agents', 'Custom AI Agents']
            },
            {
                'id': 'web_automation',
                'name': 'Web Automation',
                'description': 'Browser automation and web scraping APIs',
                'examples': ['BrowserBase', 'Playwright Cloud', 'Selenium Grid']
            }
        ]
    
    def get_template_info(self, template_type: str) -> Dict[str, str]:
        """Get detailed information about a specific template type."""
        templates = self.get_available_templates()
        for template in templates:
            if template['id'] == template_type:
                return template
        raise ValueError(f"Unknown template type: {template_type}")


# Export for easy import
__all__ = ['TemplateSelector']
