"""
{Service} Agent Runner Template

Copy this file to your_service/your_agent_runner.py and fill in the service-specific logic.

Instructions:
1. Replace all {SERVICE} placeholders with your service name
2. Replace all {service} placeholders with your service name (lowercase)
3. Replace all YourService references with your actual service
4. Implement the 5 required methods below
5. Customize INSTRUCTION_STYLES for your service
6. Add service-specific tool imports

"""

import os
import sys
import logging
from typing import Dict, Any, List

# TODO: Replace with your actual service tools
from agno.tools.yourservice import YourServiceTools
AGNO_AVAILABLE = True

# Import base class from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from base_agent_runner import BaseAgentRunner

logger = logging.getLogger(__name__)


class {Service}AgentRunner(BaseAgentRunner):
    """
    {Service} agent runner - inherits common logic from BaseAgentRunner.
    
    Implements only {Service}-specific functionality:
    - {Service} credential validation
    - {Service}Tools initialization
    - {Service}-specific tool selection
    - {Service}-specific query building
    
    All other logic (model creation, test execution, response parsing) inherited from base class.
    """
    
    # TODO: Customize these instruction styles for your service
    INSTRUCTION_STYLES = {
        'minimal': [
            "You are a {Service} assistant.",
            "Use {Service} tools to complete tasks accurately."
        ],
        'detailed': [
            "You are a specialized {Service} assistant with access to the API.",
            "When asked to interact with {Service}:",
            "1. Choose the appropriate tool for the task",
            "2. Use correct parameters",
            "3. Extract and present relevant information from results",
            "4. Ensure data completeness",
            "5. Provide accurate, factual responses",
            "6. If multiple tools are needed, execute them in sequence"
        ],
        'structured': [
            "You are a {Service} data assistant with structured data requirements.",
            "For each query, you must:",
            "- Identify the correct tool to use",
            "- Call the tool with precise parameters",
            "- Return results in a structured format with all available fields",
            "- Format output as clear, complete JSON-like data structures"
        ]
    }
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize {Service} agent runner with configuration."""
        if not AGNO_AVAILABLE:
            raise ImportError(
                "Agno package not installed. Install with: pip install agno"
            )
        
        # TODO: Add any service-specific validation here
        # For example:
        # if not _SERVICE_AVAILABLE:
        #     raise ImportError("Required package not installed")
        
        super().__init__(config)
    
    def _get_service_config(self) -> Dict[str, Any]:
        """
        Get {Service}-specific configuration from agent_config.
        
        TODO: Replace 'yourservice_auth' with your service's config key.
        This key should match the section in your YAML config file.
        """
        return self.agent_config.get('{service}_auth', {})
    
    def _validate_credentials(self) -> None:
        """
        Validate that {Service} API credentials are available.
        
        TODO: Replace with your actual credential validation logic.
        Common patterns:
        - Simple API key: Check for single env var
        - OAuth: Check for client_id and client_secret
        - Token file: Check for token file existence
        """
        # Example: Simple API key validation
        api_key = os.getenv(self.service_config.get('api_key_env', '{SERVICE}_API_KEY'))
        
        if not api_key:
            raise ValueError(
                "{Service} API key not found. Set environment variable:\n"
                f"  export {self.service_config.get('api_key_env', '{SERVICE}_API_KEY')}='your_api_key'\n"
                "Get your API key from: https://yourservice.com/developers"
            )
        
        # TODO: Add any additional credential validation here
        # For example:
        # client_secret = os.getenv(self.service_config.get('client_secret_env'))
        # if not client_secret:
        #     raise ValueError("Client secret not found")
    
    def _initialize_tools(self) -> YourServiceTools:
        """
        Initialize {Service}Tools with credentials.
        
        TODO: Replace with your actual tool initialization logic.
        Common patterns:
        - Simple API key: return ServiceTools(api_key=api_key)
        - OAuth: return ServiceTools(client_id=..., client_secret=...)
        - Token file: return ServiceTools(token_path=token_path)
        """
        # Example: Simple API key initialization
        api_key = os.getenv(self.service_config.get('api_key_env', '{SERVICE}_API_KEY'))
        
        # TODO: Extract any additional configuration parameters
        extra_config = {}
        # For example:
        # if self.service_config.get('api_url'):
        #     extra_config['api_url'] = self.service_config['api_url']
        
        # Initialize tools
        logger.info(f"Initializing {Service} tools...")
        return YourServiceTools(api_key=api_key, **extra_config)
    
    def _select_tools(self, tools: YourServiceTools, strategy: str, test_case: Dict[str, Any] = None) -> List:
        """
        Select {Service} tools based on strategy and test case restrictions.
        
        TODO: Customize tool selection logic for your service.
        Common patterns:
        - Read-only tools for safe testing
        - Specific tools for specific test cases
        - All tools by default
        """
        # Check for tool restrictions in test case metadata
        if test_case and 'metadata' in test_case:
            tool_restriction = test_case['metadata'].get('tool_restriction')
            
            if tool_restriction == 'read_only':
                # Only read operations
                logger.info("🔧 Tool restriction: read-only operations")
                # TODO: Return read-only tools
                # return [tools.get_data, tools.list_items]
                return [tools]
            
            elif tool_restriction == 'write_only':
                # Only write operations
                logger.info("🔧 Tool restriction: write-only operations")
                # TODO: Return write tools
                # return [tools.create_item, tools.update_item]
                return [tools]
            
            elif tool_restriction == 'all_tools_available':
                # All tools available - agent must choose
                logger.info("🔧 Tool restriction: all tools available - agent chooses")
                return [tools]
        
        # Fallback to strategy-based selection
        if strategy == 'include_all':
            # Include all tools
            return [tools]
        elif strategy == 'include_specific':
            # TODO: Return specific tools based on test case or config
            # return [tools.get_data, tools.create_item]
            return [tools]
        else:
            # Default: all tools
            return [tools]
    
    def _build_query(self, test_case: Dict[str, Any]) -> str:
        """
        Build query string from test case input.
        
        TODO: Customize query building logic for your service.
        Common patterns:
        - Simple: Just return test_case['input']['query']
        - With context: Append additional fields as context
        - With task: Combine query and task fields
        """
        input_data = test_case.get('input', {})
        query = input_data.get('query', '')
        
        # Example: Append additional context fields
        other_fields = []
        for key, value in input_data.items():
            if key not in ['query', 'task']:
                other_fields.append(f"{key}: {value}")
        
        if other_fields:
            query += "\n\nContext:\n" + "\n".join(other_fields)
        
        # TODO: Add any service-specific query formatting here
        # For example:
        # task = input_data.get('task', '')
        # if task:
        #     query += f"\n\nTask details: {task}"
        
        return query


# Convenience function for testing
def test_{service}_connection():
    """
    Test {Service} API connection with credentials.
    
    TODO: Replace with your actual connection test logic.
    """
    # TODO: Get credentials from environment
    api_key = os.getenv('{SERVICE}_API_KEY')
    
    if not api_key:
        print("❌ {Service} API key not found")
        print("Set environment variable:")
        print(f"  export {SERVICE}_API_KEY='your_api_key'")
        return False
    
    try:
        # TODO: Test connection to your service
        tools = YourServiceTools(api_key=api_key)
        # result = tools.some_test_method()
        print(f"✅ Successfully connected to {Service}")
        print(f"   Connection test successful")
        return True
        
    except Exception as e:
        print(f"❌ {Service} connection failed: {e}")
        return False


if __name__ == "__main__":
    # Quick test
    print(f"Testing {Service} API connection...")
    test_{service}_connection()

