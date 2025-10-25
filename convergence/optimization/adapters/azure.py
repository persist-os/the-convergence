"""Azure OpenAI API adapter with model registry support."""
import os
from typing import Dict, Any
from . import APIAdapter
from ..models import APIResponse


class AzureOpenAIAdapter(APIAdapter):
    """
    Azure OpenAI adapter with model registry support.
    
    Azure OpenAI is mostly compatible with OpenAI's API, but has:
    - Different authentication (api-key header instead of Bearer)
    - Different endpoint structure (includes deployment name)
    - Slightly different response format for some endpoints
    
    This adapter handles these differences and provides multi-model support.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Azure adapter with model registry config."""
        self.config = config
        self.api_config = self.config.get('api', {})
        self.models = self.api_config.get('models', {})
    
    def transform_request(
        self,
        optimization_params: Dict[str, Any],
        test_case: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Transform params into Azure OpenAI request format."""
        return {**test_case.get("input", {}), **optimization_params}
    
    def transform_response(
        self,
        api_response: APIResponse,
        optimization_params: Dict[str, Any]
    ) -> APIResponse:
        """Transform Azure OpenAI response."""
        return api_response
    
    def get_endpoint_for_model(self, model_key: str) -> str:
        """Get endpoint for specific model from registry."""
        model_config = self.models.get(model_key, {})
        return model_config.get('endpoint', '')
    
    def get_api_key_for_model(self, model_key: str) -> str:
        """Get API key env var name for the Azure resource (same key for all models)."""
        # All models in the same Azure resource share the same API key
        # Return the env var name, not the value - APICaller will look it up
        token_env = self.api_config.get('auth', {}).get('token_env', 'AZURE_API_KEY')
        return token_env

