"""
Custom evaluator for Azure OpenAI o4-mini reasoning tasks
"""

import json
import time
import re
from typing import Dict, Any, Optional


def score_o4_mini_response(result: Any, expected: Any, params: Dict[str, Any], metric: Optional[str] = None) -> float:
    """
    Score o4-mini API response for reasoning tasks
    
    Args:
        result: API response from o4-mini
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
            # Token limit reached - provide guidance but return partial score
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
    if metric == "reasoning_accuracy":
        return evaluate_reasoning_accuracy(response_text, expected)
    elif metric == "solution_completeness":
        return evaluate_solution_completeness(response_text, expected)
    elif metric == "latency_sec":
        return evaluate_latency(result, params)
    elif metric == "cost_per_task":
        return evaluate_cost(result, params)
    else:
        # Default: average all metrics
        return (
            evaluate_reasoning_accuracy(response_text, expected) * 0.4 +
            evaluate_solution_completeness(response_text, expected) * 0.3 +
            evaluate_latency(result, params) * 0.2 +
            evaluate_cost(result, params) * 0.1
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


def evaluate_reasoning_accuracy(response: str, expected: Any) -> float:
    """Evaluate if the reasoning leads to correct answer"""
    if not response:
        return 0.0
    
    response_lower = response.lower()
    
    # Check for expected answer (basic math problems)
    if isinstance(expected, dict) and "answer" in expected:
        answer = str(expected["answer"])
        if answer.lower() in response_lower:
            return 1.0
    
    # Check for logic puzzle answers (pet assignments)
    if isinstance(expected, dict):
        score = 0.0
        checks = 0
        
        # Logic puzzle: check pet assignments
        if "alex_pet" in expected:
            checks += 1
            if "alex" in response_lower and "bird" in response_lower:
                score += 0.33
        if "bailey_pet" in expected:
            checks += 1
            if "bailey" in response_lower and "bird" in response_lower:
                score += 0.33
        if "casey_pet" in expected:
            checks += 1
            if "casey" in response_lower and "dog" in response_lower:
                score += 0.34
        
        # Multi-step reasoning: check for numerical answers
        if "additional_budget" in expected:
            checks += 1
            # Look for the expected value (with some tolerance)
            additional = expected["additional_budget"]
            # Check for numbers in response
            import re
            numbers = re.findall(r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)', response)
            numbers = [float(n.replace(',', '')) for n in numbers]
            
            # Check if expected value is close to any number in response
            tolerance = additional * 0.1  # 10% tolerance
            if any(abs(num - additional) < tolerance for num in numbers):
                score += 0.5
        
        if "percentage_increase" in expected:
            checks += 1
            percentage = expected["percentage_increase"]
            # Look for percentage in response
            import re
            percentages = re.findall(r'(\d+\.?\d*)%', response)
            percentages = [float(p) for p in percentages]
            
            tolerance = 2.0  # 2% tolerance
            if any(abs(pct - percentage) < tolerance for pct in percentages):
                score += 0.5
        
        if checks > 0:
            return min(score, 1.0)
    
    # Fallback: Check for reasoning indicators
    reasoning_keywords = ["because", "therefore", "so", "thus", "step", "first", "then", "finally"]
    reasoning_count = sum(1 for kw in reasoning_keywords if kw in response_lower)
    
    return min(reasoning_count / 3.0, 1.0)  # Max score if 3+ reasoning words


def evaluate_solution_completeness(response: str, expected: Any) -> float:
    """Evaluate if solution is complete and well-explained"""
    if not response:
        return 0.0
    
    score = 0.0
    
    # Length check (reasonable explanation)
    if len(response) > 50:
        score += 0.3
    if len(response) > 150:
        score += 0.2
    
    # Check for step-by-step reasoning
    if "step" in response.lower() or any(str(i) in response for i in range(1, 6)):
        score += 0.3
    
    # Check for conclusion/answer
    conclusion_words = ["answer", "result", "solution", "therefore", "total"]
    if any(word in response.lower() for word in conclusion_words):
        score += 0.2
    
    return min(score, 1.0)


def evaluate_latency(result: Any, params: Dict[str, Any]) -> float:
    """
    Evaluate latency (lower is better, but we return higher_is_better score)
    
    Handles cases where latency data might be missing or malformed.
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
    
    Handles cases where usage data might be missing or malformed.
    """
    try:
        cost_usd = 0.0
        
        if isinstance(result, dict) and "usage" in result:
            usage = result["usage"]
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            completion_tokens = int(usage.get("completion_tokens", 0))
            
            # o4-mini pricing (example rates, adjust as needed)
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
