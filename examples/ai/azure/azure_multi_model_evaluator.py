"""
Custom evaluator for Azure multi-model optimization
"""

import json
import time
import re
from typing import Dict, Any, Optional


def score_azure_response(result: Any, expected: Any, params: Dict[str, Any], metric: Optional[str] = None) -> float:
    """
    Score Azure API response for multi-model optimization tasks
    
    Args:
        result: API response from Azure OpenAI
        expected: Expected outcomes from test case
        params: Parameters used for the request
        metric: Specific metric to evaluate
    
    Returns:
        float: Score between 0.0 and 1.0 for the requested metric
    
    Raises:
        RuntimeError: If result is None or API call fundamentally failed
    """
    # FAIL HARD if no result (API call failed)
    if result is None:
        raise RuntimeError(
            "API call returned None. This means the Azure OpenAI API call failed. "
            "Check your AZURE_API_KEY, endpoint, and deployment name."
        )
    
    # Check for problematic finish reasons
    finish_reason = None
    if isinstance(result, dict) and "choices" in result and len(result["choices"]) > 0:
        finish_reason = result["choices"][0].get("finish_reason")
    
    # Extract response content
    response_text = extract_response_text(result)
    
    # Handle cases where content is empty but we can still evaluate
    if not response_text:
        # Check if this is due to token limits or content filtering
        if finish_reason == "length":
            print(f"⚠️  Warning: Response truncated due to token limit (finish_reason: 'length')")
            print(f"   Consider increasing max_completion_tokens in your config")
            # Return partial score for metrics we can still evaluate
            if metric == "latency_sec":
                return evaluate_latency(result, params)
            elif metric == "cost_per_task":
                return evaluate_cost(result, params)
            else:
                # For content-based metrics, return 0.0 but don't crash
                return 0.0
        
        elif finish_reason == "content_filter":
            raise RuntimeError(
                "Response blocked by content filter. "
                "Your prompt or response may have triggered Azure's content safety filters."
            )
        
        else:
            # Unknown reason for empty content - provide detailed error
            raise RuntimeError(
                f"No content extracted from API response. "
                f"Finish reason: {finish_reason}, "
                f"Response type: {type(result)}, "
                f"Response preview: {str(result)[:300]}"
            )
    
    # Route to appropriate evaluator based on metric
    if metric == "response_quality":
        return evaluate_response_quality(response_text, expected)
    elif metric == "response_length":
        return evaluate_response_length(response_text, expected)
    elif metric == "latency_sec":
        return evaluate_latency(result, params)
    elif metric == "cost_per_task":
        return evaluate_cost(result, params)
    else:
        # Default: average all metrics
        return (
            evaluate_response_quality(response_text, expected) * 0.4 +
            evaluate_response_length(response_text, expected) * 0.2 +
            evaluate_latency(result, params) * 0.2 +
            evaluate_cost(result, params) * 0.2
        )


def extract_response_text(result: Any) -> str:
    """
    Extract text content from Azure OpenAI response
    
    Handles multiple response formats and edge cases robustly.
    """
    try:
        if isinstance(result, dict):
            # Azure OpenAI format: {"choices": [{"message": {"content": "..."}}]}
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                
                # Try to get content from message
                if "message" in choice:
                    content = choice["message"].get("content")
                    # Handle None or empty string
                    if content:
                        return str(content)
                
                # Fallback: check for text field (older API versions)
                if "text" in choice:
                    text = choice.get("text")
                    if text:
                        return str(text)
        
        elif isinstance(result, str):
            return result
        
        # If we get here, no valid content was found
        return ""
    
    except (KeyError, IndexError, AttributeError, TypeError) as e:
        # Log the error but don't crash - return empty string
        print(f"⚠️  Warning: Error extracting response text: {e}")
        return ""


def evaluate_response_quality(response: str, expected: Any) -> float:
    """Evaluate response quality based on content and structure"""
    if not response:
        return 0.0
    
    score = 0.0
    
    # Length check (reasonable response)
    if len(response) > 50:
        score += 0.3
    if len(response) > 150:
        score += 0.2
    
    # Check for structured content
    if any(word in response.lower() for word in ["answer", "result", "solution", "therefore"]):
        score += 0.3
    
    # Check for reasoning indicators
    reasoning_keywords = ["because", "therefore", "so", "thus", "step", "first", "then", "finally"]
    reasoning_count = sum(1 for kw in reasoning_keywords if kw in response.lower())
    score += min(reasoning_count / 3.0, 0.2)  # Max 0.2 for reasoning
    
    return min(score, 1.0)


def evaluate_response_length(response: str, expected: Any) -> float:
    """Evaluate if response length is appropriate"""
    if not response:
        return 0.0
    
    length = len(response)
    
    # Score based on length appropriateness
    if 100 <= length <= 500:
        return 1.0
    elif 50 <= length <= 1000:
        return 0.8
    elif 20 <= length <= 2000:
        return 0.6
    else:
        return 0.3


def evaluate_latency(result: Any, params: Dict[str, Any]) -> float:
    """
    Evaluate latency (lower is better, but we return higher_is_better score)
    """
    try:
        # Extract latency from result
        latency_ms = 0.0
        
        if isinstance(result, dict):
            # Check various fields that might contain timing
            if "latency_ms" in result:
                latency_ms = float(result["latency_ms"])
            elif "response_time_ms" in result:
                latency_ms = float(result["response_time_ms"])
        
        # If no latency found, estimate based on token count
        if latency_ms == 0.0:
            # Rough estimate: ~50ms per 100 tokens
            if isinstance(result, dict) and "usage" in result:
                total_tokens = result["usage"].get("total_tokens", 1000)
                latency_ms = (total_tokens / 100) * 50
            else:
                latency_ms = 2000  # Default 2s estimate
        
        latency_sec = latency_ms / 1000.0
        
        # Score: 1.0 for < 1s, 0.5 for 5s, 0.0 for > 10s
        if latency_sec < 1.0:
            return 1.0
        elif latency_sec < 5.0:
            return 1.0 - ((latency_sec - 1.0) / 8.0)
        elif latency_sec < 10.0:
            return 0.5 - ((latency_sec - 5.0) / 10.0)
        else:
            return 0.0
    
    except (ValueError, TypeError, KeyError) as e:
        # If we can't evaluate latency, return neutral score
        print(f"⚠️  Warning: Error evaluating latency: {e}")
        return 0.5


def evaluate_cost(result: Any, params: Dict[str, Any]) -> float:
    """
    Evaluate cost (lower is better, but we return higher_is_better score)
    """
    try:
        cost_usd = 0.0
        
        if isinstance(result, dict) and "usage" in result:
            usage = result["usage"]
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            completion_tokens = int(usage.get("completion_tokens", 0))
            
            # Generic Azure pricing estimate (adjust as needed)
            # $0.15/1M input tokens, $0.60/1M output tokens
            cost_usd = (prompt_tokens * 0.00000015) + (completion_tokens * 0.00000060)
        
        # Score: 1.0 for < $0.001, 0.5 for $0.01, 0.0 for > $0.10
        if cost_usd < 0.001:
            return 1.0
        elif cost_usd < 0.01:
            return 1.0 - ((cost_usd - 0.001) / 0.018)
        elif cost_usd < 0.10:
            return 0.5 - ((cost_usd - 0.01) / 0.18)
        else:
            return 0.0
    
    except (ValueError, TypeError, KeyError) as e:
        # If we can't evaluate cost, return neutral score
        print(f"⚠️  Warning: Error evaluating cost: {e}")
        return 0.5
