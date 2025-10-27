"""
Base Agent Runner for Agno Agents with Azure OpenAI Integration (Agno 2.1.8+)

Abstract base class that provides common agent execution logic for all agent runners.
Eliminates code duplication across Discord, Gmail, Reddit, and other agent implementations.

Architecture:
- Abstract base class with common logic
- Subclasses implement only service-specific methods
- 90% code reduction per agent (from ~500 lines to ~50 lines)

Common functionality provided:
- Azure OpenAI model initialization
- Instruction style management
- Tool selection strategies
- Response parsing from RunOutput objects
- Token usage extraction
- Error handling and timing
- Endpoint URL parsing for Azure deployments

Abstract methods (must implement in subclasses):
- _get_service_config() - Get service-specific config section
- _validate_credentials() - Validate service credentials
- _initialize_tools() - Initialize service-specific tools
- _select_tools() - Service-specific tool filtering
- _build_query() - Service-specific query formatting
"""

import os
import time
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from abc import ABC, abstractmethod

from agno.agent import Agent
from agno.models.azure import AzureOpenAI
AGNO_AVAILABLE = True

logger = logging.getLogger(__name__)


class BaseAgentRunner(ABC):
    """
    Abstract base class for all Agno agent runners.
    
    Handles common agent execution logic:
    - Azure OpenAI model configuration
    - Instruction style management
    - Tool selection strategies
    - Response parsing and formatting
    - Error handling and timing
    
    Subclasses must implement:
    - _get_service_config() - Service-specific configuration
    - _validate_credentials() - Service credential validation
    - _initialize_tools() - Service-specific tool setup
    - _select_tools() - Service-specific tool filtering
    - _build_query() - Service-specific query formatting
    
    Usage:
        class MyAgentRunner(BaseAgentRunner):
            def _get_service_config(self):
                return self.agent_config.get('my_service_auth', {})
            
            def _validate_credentials(self):
                # Validate credentials
                pass
            
            def _initialize_tools(self):
                return MyTools()
            
            def _select_tools(self, tools, strategy, test_case):
                # Select tools based on strategy
                return [tools]
            
            def _build_query(self, test_case):
                return test_case['input']['query']
    """
    
    # Common instruction styles (can be overridden by subclasses)
    INSTRUCTION_STYLES = {
        'minimal': [
            "You are a helpful assistant.",
            "Use available tools to complete tasks accurately."
        ],
        'detailed': [
            "You are a specialized assistant with access to tools.",
            "When asked to complete a task:",
            "1. Choose the appropriate tool for the task",
            "2. Use correct parameters",
            "3. Extract and present relevant information from results",
            "4. Ensure data completeness",
            "5. Provide accurate, factual responses",
            "6. If multiple tools are needed, execute them in sequence"
        ],
        'structured': [
            "You are a data assistant with structured data requirements.",
            "For each query, you must:",
            "- Identify the correct tool to use",
            "- Call the tool with precise parameters",
            "- Return results in a structured format",
            "- Include all important fields",
            "- Format output as clear, complete data structures"
        ]
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base agent runner.
        
        Args:
            config: Configuration dict from YAML (contains agent and search_space)
        """
        if not AGNO_AVAILABLE:
            raise ImportError(
                "Agno package not installed. Install with: pip install agno"
            )
        
        self.config = config
        self.agent_config = config.get('agent', {})
        
        # Service-specific config (subclass implements _get_service_config)
        self.service_config = self._get_service_config()
        
        # Validate credentials (subclass implements _validate_credentials)
        self._validate_credentials()
    
    # Abstract methods - must be implemented by subclasses
    @abstractmethod
    def _get_service_config(self) -> Dict[str, Any]:
        """Get service-specific configuration from agent_config."""
        pass
    
    @abstractmethod
    def _validate_credentials(self) -> None:
        """Validate service-specific credentials."""
        pass
    
    @abstractmethod
    def _initialize_tools(self) -> Any:
        """Initialize service-specific tools."""
        pass
    
    @abstractmethod
    def _select_tools(self, tools: Any, strategy: str, test_case: Dict[str, Any] = None) -> List:
        """Select tools based on strategy and test case."""
        pass
    
    @abstractmethod
    def _build_query(self, test_case: Dict[str, Any]) -> str:
        """Build query string from test case input."""
        pass
    
    # Common methods - implemented in base class
    def create_agent(self, params: Dict[str, Any], test_case: Dict[str, Any] = None) -> Agent:
        """
        Create Agno agent with specified parameters and model from registry.
        
        Args:
            params: Agent parameters from search_space:
                - model: Model key from agent.models registry
                - temperature: Sampling temperature
                - max_completion_tokens: Max tokens for response
                - instruction_style: Prompt style (minimal/detailed/structured)
                - tool_strategy: Tool selection strategy
            test_case: Optional test case for tool restrictions
        
        Returns:
            Configured Agno Agent instance
        """
        # Initialize service-specific tools
        try:
            tools_obj = self._initialize_tools()
        except Exception as e:
            logger.error(f"Failed to initialize tools: {e}")
            raise RuntimeError(f"Tool initialization failed: {e}") from e
        
        # Select tools based on strategy
        tool_strategy = params.get('tool_strategy', 'include_all')
        tools = self._select_tools(tools_obj, tool_strategy, test_case)
        
        # Get instructions based on style
        instruction_style = params.get('instruction_style', 'detailed')
        instructions = self.INSTRUCTION_STYLES.get(
            instruction_style,
            self.INSTRUCTION_STYLES['detailed']
        )
        
        # Get model configuration from registry
        model_key = params.get('model', 'gpt-4')
        model_registry = self.agent_config.get('models', {})
        
        if model_key not in model_registry:
            raise ValueError(
                f"Model '{model_key}' not found in agent.models registry. "
                f"Available models: {list(model_registry.keys())}"
            )
        
        model_config = model_registry[model_key]
        
        # Extract model configuration
        endpoint = model_config.get('endpoint')
        api_key_env = model_config.get('api_key_env', 'AZURE_API_KEY')
        
        # Get API key from environment
        azure_api_key = os.getenv(api_key_env)
        
        if not azure_api_key:
            raise ValueError(f"API key environment variable '{api_key_env}' not set")
        if not endpoint:
            raise ValueError(f"endpoint not specified in model config for '{model_key}'")
        
        logger.info(f"Creating agent with model: {model_key}")
        logger.info(f"  Temperature: {params.get('temperature', 0.7)}")
        logger.info(f"  Max tokens: {params.get('max_completion_tokens', 1000)}")
        logger.info(f"  Instruction style: {instruction_style}")
        logger.info(f"  Tool strategy: {tool_strategy}")
        
        # Extract deployment name and base URL from endpoint
        deployment_name = self._extract_deployment_from_endpoint(endpoint)
        base_url = self._extract_base_url_from_endpoint(endpoint)
        
        # Create Azure OpenAI model (Agno 2.1.8 API)
        try:
            model = AzureOpenAI(
                id=deployment_name,
                azure_deployment=deployment_name,
                azure_endpoint=base_url,
                api_key=azure_api_key,
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
                name="Assistant",
                model=model,
                instructions=instructions,
                tools=tools,
                markdown=False,
                stream_intermediate_steps=True,
            )
            
            logger.info("✅ Agent created successfully")
            return agent
            
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            raise RuntimeError(f"Agent creation failed: {e}") from e
    
    def run_test(self, test_case: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a test case with specified agent parameters.
        
        Args:
            test_case: Test case from test cases JSON
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
        
        logger.info(f"Parsed response: {len(result['tool_calls'])} tool calls, {len(result['tool_results'])} results")
        
        return result
    
    def _extract_deployment_from_endpoint(self, endpoint: str) -> str:
        """
        Extract deployment name from Azure endpoint URL.
        
        Args:
            endpoint: Full Azure endpoint URL
            
        Returns:
            Deployment name extracted from URL
        """
        match = re.search(r'/deployments/([^/]+)/', endpoint)
        return match.group(1) if match else 'default'
    
    def _extract_base_url_from_endpoint(self, endpoint: str) -> str:
        """
        Extract base URL from Azure endpoint URL.
        
        Args:
            endpoint: Full Azure endpoint URL
            
        Returns:
            Base URL (e.g., https://resource.openai.azure.com)
        """
        match = re.search(r'(https://[^/]+)', endpoint)
        return match.group(1) if match else endpoint
