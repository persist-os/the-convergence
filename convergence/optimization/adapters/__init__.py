"""
API-specific adapters for provider request/response transformations.

Adapters handle differences between API providers, transforming:
- Requests: Convert optimization params into provider-specific format
- Responses: Extract and normalize response data for evaluators

The base system assumes OpenAI-compatible format (the industry standard).
Adapters are only needed for providers that deviate from this format.

Available Adapters:
- OpenAIAdapter: Baseline behavior (explicit adapter, not required)
- AzureOpenAIAdapter: Azure-specific auth and endpoints
- GeminiAdapter: Google Gemini's nested request/response structure
- BrowserBaseAdapter: Browser automation session API
- UniversalAgentAdapter: Universal adapter for all Agno agent implementations (Discord, Gmail, Reddit, etc.)
- LocalFunctionAdapter: Adapter for optimizing local Python functions without HTTP endpoints

To add a new adapter:
1. Create adapter file in this directory
2. Inherit from APIAdapter base class
3. Implement transform_request() and transform_response()
4. Register in runner._detect_adapter()
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from ..models import APIResponse


class APIAdapter(ABC):
    """
    Base class for API-specific request/response transformations.
    
    Adapters bridge the gap between The Convergence's optimization framework
    and provider-specific API formats.
    """
    
    @abstractmethod
    def transform_request(
        self,
        optimization_params: Dict[str, Any],
        test_case: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Transform optimization params into API-specific format.
        
        Args:
            optimization_params: Optimization parameters being tested
                                 (e.g., temperature, max_tokens, topK)
            test_case: Test case with input data and expected results
                       (e.g., {"input": {"prompt": "..."}, "expected": {...}})
            
        Returns:
            API-specific request payload ready for HTTP POST
        """
        pass
    
    @abstractmethod
    def transform_response(
        self,
        api_response: APIResponse,
        optimization_params: Dict[str, Any]
    ) -> APIResponse:
        """
        Transform API response for evaluator consumption.
        
        Args:
            api_response: Raw API response from the provider
            optimization_params: Optimization parameters used in request
            
        Returns:
            Transformed APIResponse with extracted/normalized data
        """
        pass


__all__ = ['APIAdapter']

