"""
Agno Reddit Agent Runner with Azure OpenAI Integration (Agno 2.1.8+)

Wraps Agno agents with Reddit toolkit for optimization testing.
Handles agent creation, Azure model configuration, and execution.

"""

import os
import sys
import logging
from typing import Dict, Any, List

from agno.tools.reddit import RedditTools
AGNO_AVAILABLE = True

# Import base class from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from base_agent_runner import BaseAgentRunner

logger = logging.getLogger(__name__)

# Use Async PRAW for async environment compatibility
try:
    import asyncpraw
    _PRAW_AVAILABLE = True
except ImportError:
    logger.error(
        "Async PRAW not installed. Install with: pip install asyncpraw\n"
        "https://asyncpraw.readthedocs.io/en/stable/"
    )
    _PRAW_AVAILABLE = False


class RedditAgentRunner(BaseAgentRunner):
    """
    Reddit agent runner - inherits common logic from BaseAgentRunner.
    
    Implements only Reddit-specific functionality:
    - Reddit credential validation
    - RedditTools initialization
    - Reddit-specific tool selection with restrictions
    - Reddit-specific query building
    
    All other logic (model creation, test execution, response parsing) inherited from base class.
    """
    
    INSTRUCTION_STYLES = {
        'minimal': [
            "You are a Reddit research assistant.",
            "Use Reddit tools to answer questions accurately and completely."
        ],
        'detailed': [
            "You are a specialized Reddit research assistant with access to Reddit's API.",
            "When asked to search or retrieve Reddit data:",
            "1. Choose the appropriate Reddit tool for the task (search_subreddits, get_subreddit_info, get_post_details, etc.)",
            "2. Use correct parameters - subreddit names should be without 'r/' prefix",
            "3. Extract and present the most relevant information from results",
            "4. Ensure data completeness - include all important fields like subscribers, descriptions, timestamps",
            "5. Provide accurate, factual responses based on actual Reddit data",
            "6. If multiple tools are needed, execute them in logical sequence"
        ],
        'structured': [
            "You are a Reddit data analyst with structured data requirements.",
            "For each query, you must:",
            "- Identify the correct Reddit tool to use",
            "- Call the tool with precise parameters",
            "- Return results in a structured format with all available fields",
            "- Include: names, descriptions, subscriber counts, timestamps, URLs",
            "- For searches: return multiple relevant results",
            "- For info requests: return complete data objects",
            "- Format output as clear, complete JSON-like data structures"
        ]
    }
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Reddit agent runner with configuration."""
        if not AGNO_AVAILABLE:
            raise ImportError(
                "Agno package not installed. Install with: pip install agno"
            )
        if not _PRAW_AVAILABLE:
            raise ImportError("asyncpraw not installed. Install with: pip install asyncpraw")
        super().__init__(config)
    
    def _get_service_config(self) -> Dict[str, Any]:
        """Get Reddit-specific configuration from agent_config."""
        return self.agent_config.get('reddit_auth', {})
    
    def _validate_credentials(self) -> None:
        """Validate that Reddit API credentials are available."""
        client_id = os.getenv(self.service_config.get('client_id_env', 'REDDIT_CLIENT_ID'))
        client_secret = os.getenv(self.service_config.get('client_secret_env', 'REDDIT_CLIENT_SECRET'))
        
        if not client_id or not client_secret:
            raise ValueError(
                "Reddit API credentials not found. Set environment variables:\n"
                f"  export {self.service_config.get('client_id_env', 'REDDIT_CLIENT_ID')}='your_client_id'\n"
                f"  export {self.service_config.get('client_secret_env', 'REDDIT_CLIENT_SECRET')}='your_client_secret'\n"
                "Get credentials from: https://www.reddit.com/prefs/apps"
            )
    
    def _initialize_tools(self) -> RedditTools:
        """Initialize RedditTools with credentials."""
        client_id = os.getenv(self.service_config.get('client_id_env', 'REDDIT_CLIENT_ID'))
        client_secret = os.getenv(self.service_config.get('client_secret_env', 'REDDIT_CLIENT_SECRET'))
        user_agent = self.service_config.get('user_agent', 'agno-reddit-tester/1.0')
        
        # Initialize Reddit tools
        reddit_tools_config = {
            'client_id': client_id,
            'client_secret': client_secret,
            'user_agent': user_agent
        }
        
        # Optional: username/password for authenticated access
        username_env = self.service_config.get('username_env')
        password_env = self.service_config.get('password_env')
        if username_env and password_env:
            username = os.getenv(username_env)
            password = os.getenv(password_env)
            if username and password:
                reddit_tools_config['username'] = username
                reddit_tools_config['password'] = password
        
        # Create Reddit tools
        logger.info("Initializing Reddit tools...")
        reddit_tools = RedditTools(**reddit_tools_config)
        
        return reddit_tools
    
    def _select_tools(self, reddit_tools: Any, strategy: str, test_case: Dict[str, Any] = None) -> List:
        """Select Reddit tools based on strategy and test case restrictions."""
        # Check for tool restrictions in test case metadata
        if test_case and 'metadata' in test_case:
            tool_restriction = test_case['metadata'].get('tool_restriction')
            
            if tool_restriction == 'get_subreddit_info_only':
                # Only allow get_subreddit_info function
                logger.info("🔧 Tool restriction: get_subreddit_info only")
                return [reddit_tools.get_subreddit_info]
            
            elif tool_restriction == 'all_tools_available':
                # All tools available - agent must choose
                logger.info("🔧 Tool restriction: all tools available - agent chooses")
                return [reddit_tools]
        
        # Fallback to strategy-based selection
        if strategy == 'include_all':
            return [reddit_tools]
        elif strategy == 'include_specific':
            return [reddit_tools]
        else:
            return [reddit_tools]
    
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
def test_reddit_connection():
    """Test Reddit API connection with credentials."""
    client_id = os.getenv('REDDIT_CLIENT_ID')
    client_secret = os.getenv('REDDIT_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        print("❌ Reddit credentials not found")
        print("Set environment variables:")
        print("  export REDDIT_CLIENT_ID='your_client_id'")
        print("  export REDDIT_CLIENT_SECRET='your_client_secret'")
        return False
    
    try:
        import asyncpraw
        import asyncio
        
        async def test_async():
            reddit = asyncpraw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent='test-connection/1.0'
            )
            
            # Test by getting r/technology info
            subreddit = await reddit.subreddit('technology')
            info = await subreddit.load()
            print(f"✅ Successfully connected to Reddit")
            print(f"   r/technology has {info.subscribers:,} subscribers")
            await reddit.close()
            return True
        
        return asyncio.run(test_async())
        
    except Exception as e:
        print(f"❌ Reddit connection failed: {e}")
        return False


if __name__ == "__main__":
    # Quick test
    print("Testing Reddit API connection...")
    test_reddit_connection()
