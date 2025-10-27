"""
Agno Gmail Agent Runner with Azure OpenAI Integration (Agno 2.1.8+)

Wraps Agno agents with Gmail toolkit for optimization testing.
Handles agent creation, Azure model configuration, and execution.

"""

import os
import sys
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, List

from agno.tools.gmail import GmailTools
AGNO_AVAILABLE = True

# Import base class from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from base_agent_runner import BaseAgentRunner

logger = logging.getLogger(__name__)


class GmailAgentRunner(BaseAgentRunner):
    """
    Gmail agent runner - inherits common logic from BaseAgentRunner.
    
    Implements only Gmail-specific functionality:
    - Gmail credential validation
    - GmailTools initialization with token path
    - Gmail-specific tool selection
    - Gmail-specific query building
    
    All other logic (model creation, test execution, response parsing) inherited from base class.
    """
    
    INSTRUCTION_STYLES = {
        'minimal': [
            "You are a Gmail assistant.",
            "Use Gmail tools to manage emails accurately and efficiently."
        ],
        'detailed': [
            "You are a specialized Gmail assistant with access to your email account.",
            "When asked to manage emails:",
            "1. Choose the appropriate Gmail tool for the task (get_latest_emails, get_unread_emails, search_emails, send_email, etc.)",
            "2. Use correct parameters - email addresses, dates, search queries",
            "3. Extract and present the most relevant information from results",
            "4. Ensure data completeness - include all important fields like from, subject, date, body",
            "5. Provide accurate, factual responses based on actual email data",
            "6. If multiple tools are needed, execute them in logical sequence"
        ],
        'structured': [
            "You are a Gmail data organizer with structured data requirements.",
            "For each query, you must:",
            "- Identify the correct Gmail tool to use",
            "- Call the tool with precise parameters",
            "- Return results in a structured format with all available fields",
            "- Include: from, to, subject, date, body, thread_id, references",
            "- For searches: return multiple relevant results",
            "- For info requests: return complete email objects",
            "- Format output as clear, complete email data structures"
        ]
    }
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Gmail agent runner with configuration."""
        if not AGNO_AVAILABLE:
            raise ImportError(
                "Agno package not installed. Install with: pip install agno"
            )
        super().__init__(config)
        self._temp_token_path = None  # Track temp token file for cleanup
    
    def _get_service_config(self) -> Dict[str, Any]:
        """Get Gmail-specific configuration from agent_config."""
        return self.agent_config.get('gmail_auth', {})
    
    def _validate_credentials(self) -> None:
        """Validate that Gmail API credentials are available."""
        client_id = os.getenv(self.service_config.get('client_id_env', 'GOOGLE_CLIENT_ID'))
        client_secret = os.getenv(self.service_config.get('client_secret_env', 'GOOGLE_CLIENT_SECRET'))
        project_id = os.getenv(self.service_config.get('project_id_env', 'GOOGLE_PROJECT_ID'))
        
        if not client_id or not client_secret or not project_id:
            logger.warning(
                "Gmail API environment variables not set. GmailTools will attempt to use credentials.json file.\n"
                f"To use env vars, set:\n"
                f"  export {self.service_config.get('client_id_env', 'GOOGLE_CLIENT_ID')}='your_client_id'\n"
                f"  export {self.service_config.get('client_secret_env', 'GOOGLE_CLIENT_SECRET')}='your_client_secret'\n"
                f"  export {self.service_config.get('project_id_env', 'GOOGLE_PROJECT_ID')}='your_project_id'\n"
                "Get credentials from: https://console.cloud.google.com"
            )
    
    def _initialize_tools(self) -> GmailTools:
        """Initialize GmailTools with token from environment variable or file path."""
        gmail_dir = Path(__file__).parent
        credentials_path = self.service_config.get('credentials_path', None)
        token_env = self.service_config.get('token_env', 'GMAIL_TOKEN')
        token_path_config = self.service_config.get('token_path', None)
        
        # Check if token is provided via environment variable
        token_content = os.getenv(token_env)
        
        if token_content:
            # Token from environment - write to temporary file
            logger.info("Using Gmail token from environment variable")
            temp_token_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            try:
                # Write token content to temp file
                temp_token_file.write(token_content)
                temp_token_file.close()
                token_path = temp_token_file.name
                logger.info(f"Using temporary token file: {token_path}")
            except Exception as e:
                logger.error(f"Failed to write token to temp file: {e}")
                raise
        elif token_path_config:
            # Token from config path
            if not Path(token_path_config).is_absolute():
                token_path = str(gmail_dir / token_path_config)
            else:
                token_path = token_path_config
            logger.info(f"Using token file: {token_path}")
        else:
            # Fallback to default token.json in gmail directory
            token_path = str(gmail_dir / 'token.json')
            logger.info(f"Using default token file: {token_path}")
        
        # Try to find credentials file
        if credentials_path:
            if Path(credentials_path).exists():
                credentials_path = str(Path(credentials_path).absolute())
            elif (gmail_dir / credentials_path).exists():
                credentials_path = str(gmail_dir / credentials_path)
            else:
                logger.info(f"Credentials file not found - will use environment variables")
                credentials_path = None
        
        if credentials_path:
            logger.info(f"Using credentials file: {credentials_path}")
        else:
            logger.info("Using environment variables for credentials")
        
        # Initialize Gmail tools
        gmail_tools_config = {
            'token_path': token_path,
        }
        
        # Only add credentials_path if we have it
        if credentials_path:
            gmail_tools_config['credentials_path'] = credentials_path
        
        # Add port for OAuth flow if no token exists yet
        if not Path(token_path).exists():
            gmail_tools_config['port'] = 8080
            logger.info("Token file not found - will trigger OAuth flow on first API call")
        
        try:
            gmail_tools = GmailTools(**gmail_tools_config)
        except Exception as e:
            logger.error(f"Failed to initialize Gmail tools: {e}")
            logger.error("Make sure you have:")
            logger.error("  1. Generated token with: python generate_gmail_token.py")
            logger.error(f"  2. Set GMAIL_TOKEN environment variable or set token_path in config")
            logger.error("  3. Set GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_PROJECT_ID env vars")
            raise
        
        # Clean up temp file if we created one
        if token_content and Path(token_path).exists():
            # Don't delete immediately - GmailTools might need it
            # Store path for cleanup later
            self._temp_token_path = token_path
        
        return gmail_tools
    
    def _select_tools(self, gmail_tools: Any, strategy: str, test_case: Dict[str, Any] = None) -> List:
        """Select Gmail tools based on strategy and test case restrictions."""
        # Check for tool restrictions in test case metadata
        if test_case and 'metadata' in test_case:
            tool_restriction = test_case['metadata'].get('tool_restriction')
            
            if tool_restriction == 'get_latest_emails_only':
                # Only allow get_latest_emails function
                logger.info("🔧 Tool restriction: get_latest_emails only")
                return [gmail_tools.get_latest_emails]
            
            elif tool_restriction == 'read_only':
                # Only read operations
                logger.info("🔧 Tool restriction: read-only operations")
                return [
                    gmail_tools.get_latest_emails,
                    gmail_tools.get_unread_emails,
                    gmail_tools.get_starred_emails,
                    gmail_tools.get_emails_by_context,
                    gmail_tools.get_emails_by_date,
                    gmail_tools.get_emails_by_thread,
                    gmail_tools.search_emails
                ]
            
            elif tool_restriction == 'all_tools_available':
                # All tools available - agent must choose
                logger.info("🔧 Tool restriction: all tools available - agent chooses")
                return [gmail_tools]
        
        # Fallback to strategy-based selection
        if strategy == 'include_all':
            return [gmail_tools]
        elif strategy == 'include_specific':
            return [gmail_tools]
        else:
            return [gmail_tools]
    
    def _build_query(self, test_case: Dict[str, Any]) -> str:
        """Build query string from test case input."""
        input_data = test_case.get('input', {})
        query = input_data.get('query', '')
        task = input_data.get('task', '')
        
        # Combine query and task
        full_query = query
        if task:
            full_query += f"\n\nTask details: {task}"
        
        return full_query


# Convenience function for testing
def test_gmail_connection():
    """Test Gmail API connection with credentials."""
    client_id = os.getenv('GOOGLE_CLIENT_ID')
    client_secret = os.getenv('GOOGLE_CLIENT_SECRET')
    project_id = os.getenv('GOOGLE_PROJECT_ID')
    
    if not client_id or not client_secret or not project_id:
        print("❌ Gmail credentials not found")
        print("Set environment variables:")
        print("  export GOOGLE_CLIENT_ID='your_client_id'")
        print("  export GOOGLE_CLIENT_SECRET='your_client_secret'")
        print("  export GOOGLE_PROJECT_ID='your_project_id'")
        return False
    
    try:
        gmail_tools = GmailTools()
        # Simple test - try to get latest email
        result = gmail_tools.get_latest_emails(1)
        print(f"✅ Successfully connected to Gmail")
        print(f"   Got email result: {result[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ Gmail connection failed: {e}")
        return False


if __name__ == "__main__":
    # Quick test
    print("Testing Gmail API connection...")
    test_gmail_connection()

