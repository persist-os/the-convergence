"""
Configuration Validator for The Convergence

Validates optimization configurations and provides helpful error messages
for common configuration issues.
"""
from typing import Dict, List, Any, Optional
from rich.console import Console

console = Console()


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ConfigValidator:
    """Validates optimization configurations for common issues."""
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """
        Validate a configuration and raise helpful errors for common issues.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ConfigValidationError: If configuration has issues
        """
        validator = ConfigValidator()
        
        # Check for missing required fields
        validator._validate_required_fields(config)
        
        # Check API configuration
        validator._validate_api_config(config.get('api', {}))
        
        # Check search space
        validator._validate_search_space(config.get('search_space', {}))
        
        # Check evaluation configuration
        validator._validate_evaluation_config(config.get('evaluation', {}))
    
    def _validate_required_fields(self, config: Dict[str, Any]) -> None:
        """Validate that required fields are present."""
        required_fields = ['api', 'search_space', 'evaluation']
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            raise ConfigValidationError(
                f"Missing required configuration sections: {', '.join(missing_fields)}\n"
                f"Please check your optimization.yaml file and ensure all required sections are present."
            )
    
    def _validate_api_config(self, api_config: Dict[str, Any]) -> None:
        """Validate API configuration."""
        if not api_config:
            raise ConfigValidationError(
                "API configuration is missing or empty.\n"
                "Please add an 'api' section to your optimization.yaml file."
            )
        
        # Check for either endpoint OR models registry
        endpoint = api_config.get('endpoint')
        models = api_config.get('models')
        
        if not endpoint and not models:
            raise ConfigValidationError(
                "API endpoint or models registry is missing.\n"
                "Please specify either:\n"
                "  1. An 'endpoint' in the 'api' section (single endpoint), OR\n"
                "  2. A 'models' registry in the 'api' section (multi-model support).\n"
                "For example:\n"
                "  api:\n"
                "    endpoint: https://api.openai.com/v1/chat/completions\n"
                "  OR\n"
                "  api:\n"
                "    models:\n"
                "      model-name:\n"
                "        endpoint: https://..."
            )
        
        # Validate endpoint format if provided
        if endpoint:
            # Check for template endpoints
            if 'api.example.com' in endpoint:
                raise ConfigValidationError(
                    f"Template endpoint detected: {endpoint}\n"
                    f"Please replace this with your actual API endpoint in optimization.yaml.\n"
                    f"For example:\n"
                    f"  - Groq: https://api.groq.com/openai/v1/chat/completions\n"
                    f"  - OpenAI: https://api.openai.com/v1/chat/completions\n"
                    f"  - Anthropic: https://api.anthropic.com/v1/messages"
                )
            
            # Check for placeholder endpoints
            if 'your-resource' in endpoint or 'your-model' in endpoint:
                raise ConfigValidationError(
                    f"Placeholder endpoint detected: {endpoint}\n"
                    f"Please replace the placeholder values with your actual resource and model names."
                )
        
        # Validate models registry if provided
        if models:
            if not isinstance(models, dict):
                raise ConfigValidationError(
                    "The 'models' field must be a dictionary mapping model names to their configurations.\n"
                    "For example:\n"
                    "  models:\n"
                    "    gpt-4:\n"
                    "      endpoint: https://..."
                )
            
            if not models:
                raise ConfigValidationError(
                    "The 'models' registry is empty.\n"
                    "Please add at least one model to the models registry."
                )
            
            # Validate each model has an endpoint
            for model_name, model_config in models.items():
                if not isinstance(model_config, dict):
                    raise ConfigValidationError(
                        f"Model '{model_name}' configuration must be a dictionary."
                    )
                
                model_endpoint = model_config.get('endpoint')
                if not model_endpoint:
                    raise ConfigValidationError(
                        f"Model '{model_name}' is missing an 'endpoint' field.\n"
                        f"Please specify the endpoint for '{model_name}' in the models registry."
                    )
                
                # Check for placeholder endpoints in model registry
                if 'your-resource' in model_endpoint or 'your-model' in model_endpoint:
                    raise ConfigValidationError(
                        f"Placeholder endpoint detected for model '{model_name}': {model_endpoint}\n"
                        f"Please replace the placeholder values with your actual resource and model names."
                    )
        
        # Check authentication
        auth_config = api_config.get('auth', {})
        if not auth_config:
            raise ConfigValidationError(
                "API authentication configuration is missing.\n"
                "Please add an 'auth' section under 'api' in your optimization.yaml file."
            )
        
        token_env = auth_config.get('token_env', '')
        if not token_env:
            raise ConfigValidationError(
                "API key environment variable is not specified.\n"
                "Please specify 'token_env' in the 'auth' section of your optimization.yaml file.\n"
                "For example: token_env: 'OPENAI_API_KEY'"
            )
    
    def _validate_search_space(self, search_space: Dict[str, Any]) -> None:
        """Validate search space configuration."""
        if not search_space:
            raise ConfigValidationError(
                "Search space configuration is missing.\n"
                "Please add a 'search_space' section to your optimization.yaml file."
            )
        
        parameters = search_space.get('parameters', {})
        if not parameters:
            raise ConfigValidationError(
                "No parameters defined in search space.\n"
                "Please add parameters to optimize in the 'search_space.parameters' section."
            )
        
        # Check for model parameter (required for LLM APIs only)
        # Note: Some APIs like BrowserBase don't use model parameters
        if 'model' in parameters:
            model_config = parameters.get('model', {})
            model_values = model_config.get('values', [])
            if not model_values:
                raise ConfigValidationError(
                    "Model parameter has no values defined.\n"
                    "Please specify the models to test in the 'model.values' array.\n"
                    "Example:\n"
                    "  model:\n"
                    "    type: categorical\n"
                    "    values: ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant']"
                )
        
        # Validate other parameters
        for param_name, param_config in parameters.items():
            if param_name == 'model':
                continue  # Already validated
                
            if not isinstance(param_config, dict):
                raise ConfigValidationError(
                    f"Parameter '{param_name}' configuration must be a dictionary.\n"
                    f"Please check the parameter configuration in your optimization.yaml file."
                )
            
            param_type = param_config.get('type', '')
            if not param_type:
                raise ConfigValidationError(
                    f"Parameter '{param_name}' is missing a 'type' field.\n"
                    f"Please specify the parameter type (continuous, discrete, or categorical)."
                )
    
    def _validate_evaluation_config(self, evaluation_config: Dict[str, Any]) -> None:
        """Validate evaluation configuration."""
        if not evaluation_config:
            raise ConfigValidationError(
                "Evaluation configuration is missing.\n"
                "Please add an 'evaluation' section to your optimization.yaml file."
            )
        
        # Check test cases
        test_cases = evaluation_config.get('test_cases', {})
        if not test_cases:
            raise ConfigValidationError(
                "Test cases configuration is missing.\n"
                "Please add test cases to the 'evaluation.test_cases' section."
            )
        
        # Check metrics
        metrics = evaluation_config.get('metrics', {})
        if not metrics:
            raise ConfigValidationError(
                "Evaluation metrics are missing.\n"
                "Please add metrics to the 'evaluation.metrics' section."
            )
    
    @staticmethod
    def validate_and_suggest_fixes(config: Dict[str, Any], config_file_path: Optional[str] = None) -> None:
        """
        Validate configuration and provide specific suggestions for fixes.
        
        Args:
            config: Configuration dictionary to validate
            config_file_path: Path to the config file for context
            
        Raises:
            ConfigValidationError: If configuration has issues
        """
        try:
            ConfigValidator.validate_config(config)
        except ConfigValidationError as e:
            # Add file context to the error message
            if config_file_path:
                error_msg = f"Configuration validation failed in {config_file_path}:\n\n{e}"
            else:
                error_msg = f"Configuration validation failed:\n\n{e}"
            
            console.print(f"[red]❌ {error_msg}[/red]")
            console.print("\n[yellow]💡 Tip: Run 'convergence setup' to generate a working configuration.[/yellow]")
            
            raise ConfigValidationError(error_msg) from e
