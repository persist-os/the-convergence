"""
Agno Reddit Agent Runner with Azure OpenAI Integration (Agno 2.1.8+)

Wraps Agno agents with Reddit toolkit for optimization testing.
Handles agent creation, Azure model configuration, and execution.

Architecture:
- **Agno 2.1.8 API**: Uses AzureOpenAI model class and Agent.run() method
- Supports multiple Azure deployments for MAB-based model comparison
- Configurable instruction styles (minimal, detailed, structured)
- Tool selection strategies (include_all, include_specific)
- Response parsing from RunOutput objects
- Token usage tracking from metrics
- stream_intermediate_steps for tool call visibility

API Updates from Agno 1.x → 2.1.8:
- Model configuration moved to AzureOpenAI class (temperature, max_tokens)
- Agent.run() replaces agent.print_response() for programmatic access
- show_tool_calls → stream_intermediate_steps
- Response parsing uses RunOutput.content, .messages, .metrics
"""

import os
import time
import json
import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from agno.agent import Agent
from agno.models.azure import AzureOpenAI
from agno.tools.reddit import RedditTools
AGNO_AVAILABLE = True

try:
    import praw
    _PRAW_AVAILABLE = True 
except ImportError:
    _PRAW_AVAILABLE = False
    

logger = logging.getLogger(__name__)


class RedditAgentRunner:
    """
    Wrapper for executing Agno Reddit agents with Azure OpenAI.
    
    Features:
    - Dynamic agent creation with Azure deployment selection
    - Multiple instruction styles for prompt engineering
    - Tool selection strategies
    - Response parsing and formatting
    - Token usage tracking
    - MAB-ready for model comparison
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
        """
        Initialize Reddit agent runner.
        
        Args:
            config: Configuration dict from YAML (contains agent and search_space)
        """
        if not AGNO_AVAILABLE:
            raise ImportError(
                "Agno package not installed. Install with: pip install agno"
            )
        
        self.config = config
        self.agent_config = config.get('agent', {})
        self.reddit_auth = self.agent_config.get('reddit_auth', {})
        
        # Validate Reddit credentials
        self._validate_reddit_credentials()
    
    def _validate_reddit_credentials(self) -> None:
        """Validate that Reddit API credentials are available."""
        client_id = os.getenv(self.reddit_auth.get('client_id_env', 'REDDIT_CLIENT_ID'))
        client_secret = os.getenv(self.reddit_auth.get('client_secret_env', 'REDDIT_CLIENT_SECRET'))
        
        if not client_id or not client_secret:
            raise ValueError(
                "Reddit API credentials not found. Set environment variables:\n"
                f"  export {self.reddit_auth.get('client_id_env', 'REDDIT_CLIENT_ID')}='your_client_id'\n"
                f"  export {self.reddit_auth.get('client_secret_env', 'REDDIT_CLIENT_SECRET')}='your_client_secret'\n"
                "Get credentials from: https://www.reddit.com/prefs/apps"
            )
    
    def create_agent(self, params: Dict[str, Any], test_case: Dict[str, Any] = None) -> Agent:
        """
        Create Agno agent with specified parameters and model from registry.
        
        Args:
            params: Agent parameters from search_space:
                - model: Model key from agent.models registry (e.g., "gpt-4-1", "o4-mini")
                - temperature: Sampling temperature
                - max_completion_tokens: Max tokens for response
                - instruction_style: Prompt style (minimal/detailed/structured)
                - tool_strategy: Tool selection strategy
        
        Returns:
            Configured Agno Agent instance
        """
        # Get Reddit credentials
        client_id = os.getenv(self.reddit_auth.get('client_id_env', 'REDDIT_CLIENT_ID'))
        client_secret = os.getenv(self.reddit_auth.get('client_secret_env', 'REDDIT_CLIENT_SECRET'))
        user_agent = self.reddit_auth.get('user_agent', 'agno-reddit-tester/1.0')
        
        # Initialize Reddit tools
        reddit_tools_config = {
            'client_id': client_id,
            'client_secret': client_secret,
            'user_agent': user_agent
        }
        
        # Optional: username/password for authenticated access
        username_env = self.reddit_auth.get('username_env')
        password_env = self.reddit_auth.get('password_env')
        if username_env and password_env:
            username = os.getenv(username_env)
            password = os.getenv(password_env)
            if username and password:
                reddit_tools_config['username'] = username
                reddit_tools_config['password'] = password
        
        # Create Reddit tools
        logger.info("Initializing Reddit tools...")
        reddit_tools = RedditTools(**reddit_tools_config)
        
        # Select tools based on strategy and test case restrictions
        tool_strategy = params.get('tool_strategy', 'include_all')
        tools = self._select_tools(reddit_tools, tool_strategy, test_case)
        
        # Get instructions based on style
        instruction_style = params.get('instruction_style', 'detailed')
        instructions = self.INSTRUCTION_STYLES.get(
            instruction_style,
            self.INSTRUCTION_STYLES['detailed']
        )
        
        # Get model configuration from registry
        model_key = params.get('model', 'gpt-4-1')  # Default to gpt-4-1
        model_registry = self.agent_config.get('models', {})
        
        if model_key not in model_registry:
            raise ValueError(
                f"Model '{model_key}' not found in agent.models registry. "
                f"Available models: {list(model_registry.keys())}"
            )
        
        model_config = model_registry[model_key]
        
        # Extract model configuration
        azure_deployment = model_config.get('azure_deployment')
        azure_endpoint = model_config.get('azure_endpoint')
        api_key_env = model_config.get('api_key_env', 'AZURE_API_KEY')
        api_version = model_config.get('api_version', '2024-12-01-preview')
        
        # Get API key from environment
        azure_api_key = os.getenv(api_key_env)
        
        if not azure_api_key:
            raise ValueError(f"API key environment variable '{api_key_env}' not set")
        if not azure_deployment:
            raise ValueError(f"azure_deployment not specified in model config for '{model_key}'")
        if not azure_endpoint:
            raise ValueError(f"azure_endpoint not specified in model config for '{model_key}'")
        
        logger.info(f"Creating agent with model: {model_key}")
        logger.info(f"  Azure deployment: {azure_deployment}")
        logger.info(f"  Azure endpoint: {azure_endpoint}")
        logger.info(f"  API version: {api_version}")
        logger.info(f"  Temperature: {params.get('temperature', 0.7)}")
        logger.info(f"  Max tokens: {params.get('max_completion_tokens', 1000)}")
        logger.info(f"  Instruction style: {instruction_style}")
        logger.info(f"  Tool strategy: {tool_strategy}")
        
        # Create Azure OpenAI model (Agno 2.1.8 API)
        try:
            model = AzureOpenAI(
                id=azure_deployment,
                azure_deployment=azure_deployment,
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                api_version=api_version,  # From model config
                temperature=params.get('temperature', 0.7),
                max_completion_tokens=params.get('max_completion_tokens', 1000)
            )
            
            logger.info("✅ Azure OpenAI model created")
            
        except Exception as e:
            logger.error(f"Failed to create Azure model: {e}")
            raise RuntimeError(f"Azure model creation failed: {e}") from e
        
        # Create agent (Agno 2.1.8 API)
        try:
            agent = Agent(
                name="RedditResearcher",
                model=model,  # Pass Model object, not string
                instructions=instructions,
                tools=tools,
                markdown=False,
                stream_intermediate_steps=True,  # Changed from show_tool_calls
                # Note: temperature and max_tokens are in the model, not here
            )
            
            logger.info("✅ Agent created successfully")
            return agent
            
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            raise RuntimeError(f"Agent creation failed: {e}") from e
    
    def _select_tools(self, reddit_tools: Any, strategy: str, test_case: Dict[str, Any] = None) -> List[Any]:
        """
        Select tools based on strategy and test case restrictions.
        
        Args:
            reddit_tools: RedditTools instance
            strategy: Tool selection strategy
            test_case: Test case metadata for tool restrictions
        
        Returns:
            List of tool functions
        """
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
            # Include all RedditTools functions
            return [reddit_tools]
        
        elif strategy == 'include_specific':
            # For now, return all tools
            # In future, could filter to specific functions based on test case
            return [reddit_tools]
        
        else:
            # Default: all tools
            return [reddit_tools]
    
    def run_test(self, test_case: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a test case with specified agent parameters.
        
        Args:
            test_case: Test case from reddit_test_cases.json
            params: Agent configuration parameters
        
        Returns:
            Result dict with tool calls, response, metrics
        """
        test_id = test_case.get('id', 'unknown')
        logger.info(f"="*80)
        logger.info(f"Running test: {test_id}")
        logger.info(f"Function: {test_case.get('function', 'unknown')}")
        
        # Create agent with test-specific tool restrictions
        try:
            agent = self.create_agent(params, test_case)
        except Exception as e:
            logger.error(f"Agent creation failed: {e}")
            return {
                'test_id': test_id,
                'error': f"Agent creation failed: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
        
        # Build query
        query = self._build_query(test_case)
        logger.info(f"Query: {query[:100]}...")
        
        # Execute with timing
        start_time = time.time()
        
        try:
            # Run agent (Agno 2.1.8 API)
            logger.info("Executing agent...")
            run_output = agent.run(query, stream=False)
            
            end_time = time.time()
            latency = end_time - start_time
            
            logger.info(f"✅ Test completed in {latency:.2f}s")
            
            # Parse response
            result = self._parse_response(run_output)
            
            # Log detailed response for debugging
            logger.info(f"Parsed response: {len(result.get('tool_calls', []))} tool calls, {len(result.get('tool_results', []))} results")
            
            if result.get('final_response'):
                logger.info(f"📝 Final response preview: {result['final_response'][:200]}...")
            else:
                logger.warning("⚠️ No final response content")
                
            if result.get('tool_calls'):
                for i, tc in enumerate(result['tool_calls']):
                    tool_name = tc.get('function', {}).get('name', 'unknown')
                    logger.info(f"🔧 Tool call {i+1}: {tool_name}")
            else:
                logger.warning("⚠️ No tool calls detected")
            
            # Add metadata
            result['test_id'] = test_id
            result['test_function'] = test_case.get('function', 'unknown')
            result['query'] = query
            result['latency_seconds'] = latency
            result['timestamp'] = datetime.now().isoformat()
            result['params'] = params
            
            return result
            
        except Exception as e:
            end_time = time.time()
            logger.error(f"Test failed: {e}", exc_info=True)
            return {
                'test_id': test_id,
                'test_function': test_case.get('function', 'unknown'),
                'error': str(e),
                'latency_seconds': end_time - start_time,
                'timestamp': datetime.now().isoformat(),
                'params': params
            }
    
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
    
    def _parse_response(self, run_output: Any) -> Dict[str, Any]:
        """
        Parse agent RunOutput to extract tool calls and results.
        
        Args:
            run_output: RunOutput object from agent.run() (Agno 2.1.8)
        
        Returns:
            Structured result dict matching evaluator expectations
        """
        result = {
            'final_response': '',
            'tool_calls': [],
            'tool_results': [],
            'tokens_used': {},
            'latency_seconds': 0.0
        }
        
        # Extract response content (Agno 2.1.8 API)
        if hasattr(run_output, 'content'):
            result['final_response'] = run_output.content
        elif hasattr(run_output, 'get_content_as_string'):
            result['final_response'] = run_output.get_content_as_string()
        else:
            result['final_response'] = str(run_output)
        
        # Extract tool calls and results from messages (Agno 2.1.8 API)
        if hasattr(run_output, 'messages') and run_output.messages:
            for msg in run_output.messages:
                # Check for tool calls in the message
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_call_data = {
                            'function': {
                                'name': tc.function.name if hasattr(tc, 'function') else str(tc),
                                'arguments': tc.function.arguments if hasattr(tc, 'function') and hasattr(tc.function, 'arguments') else {}
                            }
                        }
                        result['tool_calls'].append(tool_call_data)
                        logger.debug(f"Found tool call: {tool_call_data['function']['name']}")
                
                # Check for tool results in the message
                if hasattr(msg, 'content') and msg.content:
                    # If this is a tool result message, add it to tool_results
                    if hasattr(msg, 'role') and msg.role == 'tool':
                        result['tool_results'].append({
                            'content': msg.content,
                            'tool_call_id': getattr(msg, 'tool_call_id', None)
                        })
                        logger.debug(f"Found tool result: {msg.content[:100]}...")
        
        # Extract metrics (token usage) from RunOutput (Agno 2.1.8 API)
        if hasattr(run_output, 'metrics') and run_output.metrics:
            metrics = run_output.metrics
            if isinstance(metrics, dict):
                result['tokens_used'] = {
                    'prompt_tokens': metrics.get('prompt_tokens', 0),
                    'completion_tokens': metrics.get('completion_tokens', 0),
                    'total_tokens': metrics.get('total_tokens', 0)
                }
            elif hasattr(metrics, 'prompt_tokens'):
                result['tokens_used'] = {
                    'prompt_tokens': getattr(metrics, 'prompt_tokens', 0),
                    'completion_tokens': getattr(metrics, 'completion_tokens', 0),
                    'total_tokens': getattr(metrics, 'total_tokens', 0)
                }
        
        # Try to extract tool results from response text
        # Look for JSON-like structures that might be Reddit data
        response_text = result['final_response']
        
        if response_text:
            try:
                # Try to find JSON objects in response
                json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                matches = re.findall(json_pattern, response_text, re.DOTALL)
                
                for match in matches:
                    try:
                        parsed = json.loads(match)
                        # Check if it looks like Reddit data
                        if isinstance(parsed, dict) and any(
                            key in parsed for key in ['subscribers', 'display_name', 'subreddit', 'author', 'name', 'url']
                        ):
                            result['tool_results'].append(parsed)
                            logger.debug(f"Found tool result with keys: {list(parsed.keys())}")
                    except json.JSONDecodeError:
                        pass
            except Exception as e:
                logger.debug(f"Could not extract tool results from response: {e}")
        
        logger.info(f"Parsed response: {len(result['tool_calls'])} tool calls, {len(result['tool_results'])} results")
        
        return result


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
        import praw
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent='test-connection/1.0'
        )
        
        # Test by getting r/technology info
        subreddit = reddit.subreddit('technology')
        print(f"✅ Successfully connected to Reddit")
        print(f"   r/technology has {subreddit.subscribers:,} subscribers")
        return True
        
    except Exception as e:
        print(f"❌ Reddit connection failed: {e}")
        return False


if __name__ == "__main__":
    # Quick test
    print("Testing Reddit API connection...")
    test_reddit_connection()

