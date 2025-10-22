"""
Constants for The Convergence generator module.

These values can be overridden by environment variables.
"""

import os

# Default LLM model for RLP/SAO reasoning
# Can be overridden with LLM_MODEL environment variable
DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL", "gemini/gemini-2.0-flash-exp")

# Alternative models (uncomment to change default):
# DEFAULT_LLM_MODEL = "gemini/gemini-2.5-pro"  # More capable but slower
# DEFAULT_LLM_MODEL = "openai/gpt-4o"  # OpenAI GPT-4
# DEFAULT_LLM_MODEL = "openai/gpt-3.5-turbo"  # Cheaper OpenAI option
# DEFAULT_LLM_MODEL = "anthropic/claude-3-5-sonnet-20241022"  # Claude

__all__ = ['DEFAULT_LLM_MODEL']

