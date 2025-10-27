"""
Agno Discord Agent Runner with Azure OpenAI Integration (Agno 2.1.8+)

Wraps Agno agents with Discord toolkit for optimization testing.
Handles agent creation, Azure model configuration, and execution.

"""

import os
import sys
import logging
from typing import Dict, Any, List

from agno.tools.discord import DiscordTools
AGNO_AVAILABLE = True

# Import base class from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from base_agent_runner import BaseAgentRunner

logger = logging.getLogger(__name__)


class DiscordAgentRunner(BaseAgentRunner):
    """
    Discord agent runner - inherits common logic from BaseAgentRunner.
    
    Implements only Discord-specific functionality:
    - Discord credential validation
    - DiscordTools initialization
    - Discord-specific tool selection
    - Discord-specific query building
    
    All other logic (model creation, test execution, response parsing) inherited from base class.
    """
    
    INSTRUCTION_STYLES = {
        'minimal': [
            "You are a Discord assistant.",
            "Use Discord tools to interact with channels and servers."
        ],
        'detailed': [
            "You are a specialized Discord assistant with access to Discord servers and channels.",
            "When asked to interact with Discord:",
            "1. Choose the appropriate Discord tool for the task (send_message, get_channel_messages, list_channels, etc.)",
            "2. Use correct parameters - channel IDs, message content, limits",
            "3. Extract and present relevant information from results",
            "4. Ensure data completeness - include all important fields like channel info, message content, timestamps",
            "5. Provide accurate, factual responses based on actual Discord data",
            "6. If multiple tools are needed, execute them in logical sequence"
        ],
        'structured': [
            "You are a Discord data assistant with structured data requirements.",
            "For each query, you must:",
            "- Identify the correct Discord tool to use",
            "- Call the tool with precise parameters",
            "- Return results in a structured format with all available fields",
            "- Include: channel names, message content, timestamps, user info",
            "- For searches: return multiple relevant results",
            "- For info requests: return complete data objects",
            "- Format output as clear, complete JSON-like data structures"
        ]
    }
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Discord agent runner with configuration."""
        if not AGNO_AVAILABLE:
            raise ImportError(
                "Agno package not installed. Install with: pip install agno"
            )
        super().__init__(config)
    
    def _get_service_config(self) -> Dict[str, Any]:
        """Get Discord-specific configuration from agent_config."""
        return self.agent_config.get('discord_auth', {})
    
    def _validate_credentials(self) -> None:
        """Validate that Discord bot token is available."""
        bot_token = os.getenv(self._get_service_config().get('bot_token_env', 'DISCORD_BOT_TOKEN'))
        
        if not bot_token:
            raise ValueError(
                "Discord bot token not found. Set environment variable:\n"
                f"  export {self._get_service_config().get('bot_token_env', 'DISCORD_BOT_TOKEN')}='your_bot_token'\n"
                "Get your bot token from: https://discord.com/developers/applications"
            )
    
    def _initialize_tools(self) -> DiscordTools:
        """Initialize DiscordTools with bot token."""
        bot_token_env = self.service_config.get('bot_token_env', 'DISCORD_BOT_TOKEN')
        bot_token = os.getenv(bot_token_env)
        if not bot_token:
            raise ValueError(f"Discord bot token not found in environment: {bot_token_env}")
        return DiscordTools(bot_token=bot_token)
    
    def _select_tools(self, tools: DiscordTools, strategy: str, test_case: Dict[str, Any] = None) -> List:
        """Select Discord tools based on strategy."""
        if strategy == 'include_all':
            return [tools]
        elif strategy == 'include_specific':
            return [
                tools.send_message,
                tools.get_channel_messages,
                tools.list_channels
            ]
        return [tools]
    
    def _build_query(self, test_case: Dict[str, Any]) -> str:
        """Build query string from test case input, including all input fields."""
        input_data = test_case.get('input', {})
        query = input_data.get('query', '')
        
        # Append all other input fields to the query
        other_fields = []
        for key, value in input_data.items():
            if key not in ['query', 'task']:
                other_fields.append(f"{key}: {value}")
        
        if other_fields:
            query += "\n\nContext:\n" + "\n".join(other_fields)
        
        return query
