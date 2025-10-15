"""OpenAI API adapter (default base behavior)."""
from typing import Dict, Any
from . import APIAdapter
from ..models import APIResponse


class OpenAIAdapter(APIAdapter):
    """
    Default adapter for OpenAI APIs.
    
    Supports:
    - Chat Completions API (v1/chat/completions)
    - Responses API (v1/responses)
    - Completions API (v1/completions)
    
    This is the baseline behavior - most other LLM APIs follow OpenAI's format.
    """
    
    def transform_request(
        self,
        optimization_params: Dict[str, Any],
        test_case: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Transform params into OpenAI request format.
        
        OpenAI uses a flat structure where:
        - Test case input contains: messages, input, or prompt
        - Optimization params are merged directly: model, temperature, max_tokens, etc.
        
        Args:
            optimization_params: Optimization parameters being tested
            test_case: Test case with input data
            
        Returns:
            OpenAI API request payload
        """
        # Default behavior: merge test case input with optimization params
        # This works for most OpenAI-compatible APIs
        request = {**test_case.get("input", {}), **optimization_params}
        return request
    
    def transform_response(
        self,
        api_response: APIResponse,
        optimization_params: Dict[str, Any]
    ) -> APIResponse:
        """
        Transform OpenAI response (no transformation needed by default).
        
        OpenAI responses are already in a standard format that evaluators expect.
        
        Args:
            api_response: Raw OpenAI API response
            optimization_params: Optimization parameters used in request
            
        Returns:
            Same APIResponse (no transformation needed)
        """
        # OpenAI responses are the baseline - no transformation needed
        return api_response

