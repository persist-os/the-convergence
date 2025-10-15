"""
LiteLLM Provider - Universal LLM interface for The Convergence.

Supports 100+ LLM providers through litellm: OpenAI, Anthropic, Cohere, etc.
"""

import os
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
import weave

from convergence.generator.constants import DEFAULT_LLM_MODEL
from convergence.core.env_loader import ensure_api_key, get_api_key

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


class LiteLLMConfig(BaseModel):
    """Configuration for LiteLLM provider."""

    model: str = Field(
        default=DEFAULT_LLM_MODEL,
        description="Model name in format: provider/model (e.g., gemini/gemini-2.5-flash, openai/gpt-4o)"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key (can also use provider-specific env vars like GEMINI_API_KEY, OPENAI_API_KEY)"
    )
    api_base: Optional[str] = Field(
        default=None,
        description="Custom API base URL (optional)"
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1)
    timeout: int = Field(default=60, description="Request timeout in seconds")


class LiteLLMProvider:
    """
    Universal LLM provider using litellm.
    
    Supports:
    - Google Gemini (gemini-2.5-flash, gemini-2.5-pro, etc.) - Default
    - OpenAI (gpt-4o, gpt-4-turbo, etc.)
    - Anthropic (claude-3-opus, claude-3-sonnet, etc.)
    - Cohere, Replicate, Hugging Face, and 100+ more
    
    Environment Variables (Optional):
        LLM_MODEL: Model name (imported from convergence.generator.constants.DEFAULT_LLM_MODEL)
        GEMINI_API_KEY: API key for Gemini models
        OPENAI_API_KEY: API key for OpenAI models
        ANTHROPIC_API_KEY: API key for Anthropic models
    
    Usage:
        # Use default (from constants.DEFAULT_LLM_MODEL)
        provider = LiteLLMProvider()
        
        # Or specify a model
        provider = LiteLLMProvider(model="openai/gpt-4o")
        
        # Or use env vars
        os.environ["LLM_MODEL"] = "gemini/gemini-2.5-pro"
        os.environ["GEMINI_API_KEY"] = "your-key"
        provider = LiteLLMProvider()
        
        response = await provider.generate("Hello, world!")
    """
    
    def __init__(self, config: Optional[LiteLLMConfig] = None, **kwargs):
        """
        Initialize LiteLLM provider.
        
        Reads from environment variables if not specified:
        - LLM_MODEL: Override default model
        - <PROVIDER>_API_KEY: API key for the provider
        
        Args:
            config: LiteLLM configuration
            **kwargs: Override config values
        """
        if not LITELLM_AVAILABLE:
            raise ImportError(
                "litellm is not installed. Install with: pip install litellm"
            )
        
        self.config = config or LiteLLMConfig()
        
        # Allow environment variable to override model
        if "LLM_MODEL" in os.environ and not config:
            self.config.model = os.environ["LLM_MODEL"]
        elif not config:
            # Try to get model from environment
            model = os.environ.get("LLM_MODEL")
            if model:
                self.config.model = model
                
        # Auto-load API key if not provided
        if not self.config.api_key:
            provider = self.config.model.split('/')[0] if '/' in self.config.model else 'unknown'
            api_key = get_api_key(provider)
            if api_key:
                self.config.api_key = api_key
        
        # Allow kwargs to override config
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    @weave.op()
    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (overrides config)
            max_tokens: Max tokens to generate (overrides config)
            **kwargs: Additional litellm parameters
            
        Returns:
            Dict with 'content' and optional 'metadata'
        """
        
        # Use config defaults if not specified
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        
        try:
            # Prepare litellm parameters
            litellm_params = {
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": self.config.timeout,
            }
            
            # Add API key if provided in config
            if self.config.api_key:
                litellm_params["api_key"] = self.config.api_key
            
            # Add custom API base if provided
            if self.config.api_base:
                litellm_params["api_base"] = self.config.api_base
            
            # Merge any additional kwargs
            litellm_params.update(kwargs)
            
            # Call litellm (supports async)
            response = await litellm.acompletion(**litellm_params)
            
            # Extract response (handle None content)
            content = response.choices[0].message.content or ""
            
            # Extract usage safely (different providers return different formats)
            usage = {}
            if response.usage:
                if hasattr(response.usage, '_asdict'):
                    usage = response.usage._asdict()
                elif hasattr(response.usage, 'model_dump'):
                    usage = response.usage.model_dump()
                elif hasattr(response.usage, 'dict'):
                    usage = response.usage.dict()
                else:
                    # Fallback: extract common attributes
                    usage = {
                        'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0),
                        'completion_tokens': getattr(response.usage, 'completion_tokens', 0),
                        'total_tokens': getattr(response.usage, 'total_tokens', 0),
                    }
            
            return {
                'content': content,
                'metadata': {
                    'model': self.config.model,
                    'usage': usage,
                    'finish_reason': response.choices[0].finish_reason
                }
            }
        
        except Exception as e:
            # Log error clearly with helpful context
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"LLM generation failed: {e}")
            
            # Provide helpful error message based on error type
            error_msg = str(e)
            if "AuthenticationError" in error_msg or "api_key" in error_msg.lower():
                provider = self.config.model.split('/')[0] if '/' in self.config.model else 'unknown'
                env_var = f"{provider.upper()}_API_KEY"
                logger.error(
                    f"API key not set. Please set {env_var} environment variable "
                    f"or pass api_key in config. Model: {self.config.model}"
                )
            
            # Return error in expected format
            return {
                'content': '',
                'metadata': {'error': str(e), 'model': self.config.model}
            }
    
    @weave.op()
    async def generate_structured(
        self,
        prompt: str,
        schema: type[BaseModel],
        **kwargs: Any
    ) -> BaseModel:
        """
        Generate structured output matching Pydantic schema.
        
        Uses function calling or JSON mode depending on model support.
        
        Args:
            prompt: Input prompt
            schema: Pydantic model class
            **kwargs: Additional parameters
            
        Returns:
            Instance of schema with generated data
        """
        
        # Add JSON instruction to prompt
        json_prompt = f"""{prompt}

Respond with valid JSON matching this schema:
{schema.model_json_schema()}

JSON:"""
        
        response = await self.generate(
            prompt=json_prompt,
            **kwargs
        )
        
        content = response.get('content', '{}')
        
        # Try to parse as JSON and validate with schema
        try:
            import json
            data = json.loads(content)
            return schema(**data)
        except Exception as e:
            # Fallback: return empty instance
            return schema()


class MockLLMProvider:
    """
    Mock LLM provider for testing without API calls.
    
    Returns simple mock responses based on prompt patterns.
    """
    
    def __init__(self, **kwargs):
        """Initialize mock provider."""
        pass
    
    @weave.op()
    async def generate(
        self,
        prompt: str,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Generate mock response."""
        
        # Simple pattern-based responses
        prompt_lower = prompt.lower()
        
        if 'reasoning' in prompt_lower or 'think' in prompt_lower:
            content = "Let me think through this step by step: First, I'll analyze the situation. Second, I'll consider the options. Third, I'll choose the best approach."
        elif 'question' in prompt_lower or 'request' in prompt_lower:
            content = "How can I improve my understanding of this topic?"
        elif 'response' in prompt_lower:
            content = "I can help you with that. Here's a detailed explanation of the concept you're asking about."
        elif 'evaluate' in prompt_lower or 'judge' in prompt_lower:
            content = "A is better because it provides more comprehensive information and addresses the question directly."
        else:
            content = f"Mock response to prompt: {prompt[:50]}..."
        
        return {
            'content': content,
            'metadata': {'model': 'mock', 'mock': True}
        }
    
    @weave.op()
    async def generate_structured(
        self,
        prompt: str,
        schema: type[BaseModel],
        **kwargs: Any
    ) -> BaseModel:
        """Generate mock structured response."""
        # Return empty instance of schema
        return schema()


# Convenience function
def get_llm_provider(
    model: Optional[str] = None,
    mock: bool = False,
    **kwargs
) -> LiteLLMProvider | MockLLMProvider:
    """
    Get LLM provider instance.
    
    Reads model from (in order of priority):
    1. model parameter
    2. LLM_MODEL environment variable
    3. DEFAULT_LLM_MODEL from constants
    
    Args:
        model: Model name (e.g., "gemini/gemini-2.5-flash", "openai/gpt-4o")
               If None, uses LLM_MODEL env var or DEFAULT_LLM_MODEL constant
        mock: If True, return mock provider for testing
        **kwargs: Additional config parameters
        
    Returns:
        LLM provider instance
    """
    if mock:
        return MockLLMProvider(**kwargs)
    
    # Allow env var to override default, but explicit model param takes precedence
    if model is None:
        model = os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL)
    
    config = LiteLLMConfig(model=model, **kwargs)
    return LiteLLMProvider(config=config)


__all__ = ['LiteLLMProvider', 'MockLLMProvider', 'LiteLLMConfig', 'get_llm_provider']

