"""Azure OpenAI API adapter."""
from typing import Dict, Any
from . import APIAdapter
from ..models import APIResponse


class AzureOpenAIAdapter(APIAdapter):
    """
    Adapter for Azure OpenAI Service.
    
    Azure OpenAI is mostly compatible with OpenAI's API, but has:
    - Different authentication (api-key header instead of Bearer)
    - Different endpoint structure (includes deployment name)
    - Slightly different response format for some endpoints
    
    This adapter handles these differences.
    """
    
    def transform_request(
        self,
        optimization_params: Dict[str, Any],
        test_case: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Transform params into Azure OpenAI request format.
        
        Azure OpenAI follows OpenAI's format closely, so we just merge
        test case input with optimization parameters.
        
        Args:
            optimization_params: Optimization parameters being tested
            test_case: Test case with input data
            
        Returns:
            Azure OpenAI request payload
        """
        # Azure follows OpenAI format - simple merge
        request = {**test_case.get("input", {}), **optimization_params}
        return request
    
    def transform_response(
        self,
        api_response: APIResponse,
        optimization_params: Dict[str, Any]
    ) -> APIResponse:
        """
        Transform Azure OpenAI response.
        
        Azure OpenAI responses are nearly identical to OpenAI's.
        No transformation needed in most cases.
        
        Args:
            api_response: Raw Azure OpenAI API response
            optimization_params: Optimization parameters used in request
            
        Returns:
            Same APIResponse (Azure format matches OpenAI)
        """
        # Azure responses match OpenAI format
        return api_response

