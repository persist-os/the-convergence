"""Gemini API adapter."""
from typing import Dict, Any
from . import APIAdapter
from ..models import APIResponse


class GeminiAdapter(APIAdapter):
    """Transforms generic optimization into Gemini generateContent API format."""
    
    def transform_request(
        self,
        optimization_params: Dict[str, Any],
        test_case: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Transform params into Gemini generateContent request.
        
        Converts optimization parameters to Gemini API format:
        - prompt → contents[0].parts[0].text
        - temperature/topK/topP/maxOutputTokens → generationConfig.*
        
        Args:
            optimization_params: Optimization parameters being tested
            test_case: Test case with 'prompt' in input
            
        Returns:
            Gemini generateContent payload
        """
        # Get prompt from test case input
        test_input = test_case.get("input", {})
        prompt = test_input.get("prompt", "")
        
        # Build contents structure (required by Gemini)
        contents = [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
        
        # Build generationConfig from optimization parameters
        generation_config = {}
        
        # Add parameters if they exist
        if "temperature" in optimization_params:
            generation_config["temperature"] = optimization_params["temperature"]
        
        if "topK" in optimization_params:
            generation_config["topK"] = int(optimization_params["topK"])
        
        if "topP" in optimization_params:
            generation_config["topP"] = optimization_params["topP"]
        
        if "maxOutputTokens" in optimization_params:
            generation_config["maxOutputTokens"] = int(optimization_params["maxOutputTokens"])
        
        # Build final request
        request = {
            "contents": contents
        }
        
        # Only add generationConfig if we have parameters
        if generation_config:
            request["generationConfig"] = generation_config
        
        return request
    
    def transform_response(
        self,
        api_response: APIResponse,
        optimization_params: Dict[str, Any]
    ) -> APIResponse:
        """
        Transform Gemini response for evaluator.
        
        Extracts text from Gemini's nested response structure:
        candidates[0].content.parts[0].text
        
        Args:
            api_response: Raw Gemini API response
            optimization_params: Optimization parameters used in request
            
        Returns:
            Transformed APIResponse with extracted text
        """
        if not api_response.success:
            return api_response
        
        result = api_response.result
        
        # Extract text from Gemini response structure
        if isinstance(result, dict) and "candidates" in result:
            try:
                candidates = result.get("candidates", [])
                if candidates and len(candidates) > 0:
                    candidate = candidates[0]
                    content = candidate.get("content", {})
                    parts = content.get("parts", [])
                    
                    if parts and len(parts) > 0:
                        text = parts[0].get("text", "")
                        
                        # Return simplified response with extracted text
                        return APIResponse(
                            success=True,
                            result=text,  # Just the text for easier evaluation
                            latency_ms=api_response.latency_ms,
                            estimated_cost_usd=api_response.estimated_cost_usd,
                            metadata={
                                "gemini_response": True,
                                "original_response": result
                            }
                        )
            except (KeyError, IndexError, TypeError) as e:
                # If parsing fails, return original response
                return api_response
        
        return api_response

