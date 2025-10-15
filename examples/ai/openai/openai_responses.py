"""
🤖 OpenAI Responses API Evaluator

Simple evaluator for OpenAI's Responses API (gpt-4o, gpt-4o-mini, gpt-4-turbo, etc.)

The Responses API returns:
{
    "output_text": "Under the soft glow of the moon...",
    "output": [...],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30
    }
}

This evaluator scores based on:
- Content completeness (has required keywords?)
- Response quality (basic text quality checks)
- Length appropriateness (within expected bounds)
"""
import re
from typing import Dict, Any, Optional


def score_openai_response(
    result: Any,
    expected: Dict[str, Any],
    params: Dict[str, Any],
    metric: Optional[str] = None
) -> float:
    """
    Score OpenAI Responses API output.
    
    Scoring based on:
    - Response quality: completeness, quality, length (50%)
    - Latency: response time performance (25%)
    - Cost: token efficiency (25%)
    
    Args:
        result: API response from OpenAI Responses API
            Can include:
            - output/output_text: The response text
            - usage: Token usage stats
            - latency_ms: Response latency (if provided externally)
            - cost_usd: Response cost (if provided externally)
        expected: Expected criteria:
            - contains: List of required keywords (e.g. ["Paris", "France"])
            - min_length: Minimum character count (optional)
            - max_length: Maximum character count (optional)
            - max_latency_ms: Target latency threshold (default: 2000)
            - max_cost_usd: Target cost threshold (default: 0.01)
        params: API parameters (model, temperature, etc.)
        metric: Specific metric name to return (completeness, quality, length, latency, cost)
    
    Returns:
        Quality score 0.0-1.0
    
    Example:
        result = {
            "output_text": "Paris is the capital of France.",
            "usage": {"total_tokens": 50},
            "latency_ms": 850,
            "cost_usd": 0.002
        }
        expected = {"contains": ["Paris"], "max_latency_ms": 1000}
        score = score_openai_response(result, expected, {})
        # Returns ~0.95 (has keyword, good quality, fast response, low cost)
    """
    # Extract text from response
    text = _extract_text(result)
    
    if not text or len(text.strip()) < 3:
        return 0.0
    
    # Calculate component scores
    scores = {}
    
    # 1. Response Quality Components (combined 50%)
    scores['completeness'] = _score_completeness(text, expected)
    scores['quality'] = _score_quality(text)
    scores['length'] = _score_length(text, expected)
    
    # Combined quality score
    quality_score = (
        scores['completeness'] * 0.6 +
        scores['quality'] * 0.25 +
        scores['length'] * 0.15
    )
    
    # 2. Latency Score (25%)
    scores['latency'] = _score_latency(result, expected)
    
    # 3. Cost/Efficiency Score (25%)
    scores['cost'] = _score_cost(result, expected)
    
    # If specific metric requested, return it
    if metric:
        metric_lower = metric.lower()
        if metric_lower in scores:
            return round(min(1.0, max(0.0, scores[metric_lower])), 3)
        if metric_lower in ['response_quality', 'quality_score']:
            return round(min(1.0, max(0.0, quality_score)), 3)
    
    # Final weighted score
    final_score = (
        quality_score * 0.50 +
        scores['latency'] * 0.25 +
        scores['cost'] * 0.25
    )
    
    return round(min(1.0, max(0.0, final_score)), 3)


def _extract_text(result: Any) -> str:
    """Extract text from OpenAI Responses API output."""
    if isinstance(result, str):
        return result
    
    if isinstance(result, dict):
        # Try output_text first (convenience property)
        if 'output_text' in result:
            return str(result['output_text'])
        
        # Fall back to parsing output array
        if 'output' in result and isinstance(result['output'], list):
            texts = []
            for item in result['output']:
                if isinstance(item, dict) and 'content' in item:
                    content = item['content']
                    if isinstance(content, list):
                        for content_item in content:
                            if isinstance(content_item, dict):
                                if content_item.get('type') == 'output_text':
                                    texts.append(content_item.get('text', ''))
                    elif isinstance(content, str):
                        texts.append(content)
            return '\n'.join(texts)
        
        # Try old Chat Completions format
        if 'choices' in result and isinstance(result['choices'], list):
            if len(result['choices']) > 0:
                choice = result['choices'][0]
                if 'message' in choice:
                    return choice['message'].get('content', '')
                elif 'text' in choice:
                    return choice['text']
        
        # Generic fallback
        if 'text' in result:
            return str(result['text'])
        if 'content' in result:
            return str(result['content'])
    
    return str(result)


def _score_completeness(text: str, expected: Dict[str, Any]) -> float:
    """
    Check if response contains required keywords.
    Returns percentage of required keywords found.
    """
    required = expected.get('contains', [])
    if not required:
        return 1.0  # No requirements = perfect score
    
    text_lower = text.lower()
    found = sum(1 for keyword in required if keyword.lower() in text_lower)
    
    return found / len(required)




def _score_quality(text: str) -> float:
    """
    Basic text quality check.
    Looks for: proper capitalization, reasonable word variety, not too repetitive.
    """
    if len(text) < 5:
        return 0.2
    
    score = 0.0
    
    # Check 1: Starts with capital letter (basic grammar)
    if text[0].isupper():
        score += 0.3
    else:
        score += 0.1
    
    # Check 2: Has reasonable word variety (not super repetitive)
    words = text.lower().split()
    if words:
        unique_ratio = len(set(words)) / len(words)
        score += 0.4 * min(1.0, unique_ratio * 1.5)
    else:
        score += 0.2
    
    # Check 3: Not too short, not ridiculously long
    if 10 <= len(text) <= 1000:
        score += 0.3
    elif len(text) < 10:
        score += 0.1
    else:
        score += 0.2
    
    return min(1.0, score)


def _score_length(text: str, expected: Dict[str, Any]) -> float:
    """
    Check if response length is appropriate.
    """
    min_length = expected.get('min_length', 0)
    max_length = expected.get('max_length', float('inf'))
    
    actual_length = len(text)
    
    # Perfect if within bounds
    if min_length <= actual_length <= max_length:
        return 1.0
    
    # Too short: scale proportionally
    if actual_length < min_length:
        if min_length > 0:
            return max(0.3, actual_length / min_length)  # At least 0.3
        return 0.8
    
    # Too long: small penalty
    if actual_length > max_length and max_length < float('inf'):
        overage = (actual_length - max_length) / max_length
        return max(0.5, 1.0 - overage * 0.5)  # Max 50% penalty
    
    return 0.9  # Default for edge cases


def _score_latency(result: Any, expected: Dict[str, Any]) -> float:
    """
    Score based on response latency.
    
    Faster responses score higher. Target is under 2000ms by default.
    
    Args:
        result: API response (may contain latency_ms field)
        expected: Expected criteria (may contain max_latency_ms)
    
    Returns:
        Latency score 0.0-1.0
    """
    # Extract latency from result
    latency_ms = None
    if isinstance(result, dict):
        latency_ms = result.get('latency_ms')
    
    # If no latency provided, return neutral score
    if latency_ms is None:
        return 0.8
    
    # Get target latency (default 2000ms = 2 seconds)
    target_latency = expected.get('max_latency_ms', 2000)
    
    # Score based on how close to target
    if latency_ms <= target_latency:
        # Perfect if under target
        # Bonus points for being much faster
        if latency_ms <= target_latency * 0.5:
            return 1.0  # Excellent: under half the target
        else:
            # Linear scale from 0.85 to 1.0
            ratio = latency_ms / target_latency
            return 0.85 + (1.0 - ratio) * 0.15
    else:
        # Penalty for being over target
        # Gradual penalty up to 2x target, then steeper
        overage_ratio = latency_ms / target_latency
        if overage_ratio <= 2.0:
            # 0.85 at target, 0.5 at 2x target
            return max(0.5, 0.85 - (overage_ratio - 1.0) * 0.35)
        else:
            # Steep penalty for very slow responses
            return max(0.2, 0.5 - (overage_ratio - 2.0) * 0.1)


def _score_cost(result: Any, expected: Dict[str, Any]) -> float:
    """
    Score based on cost/token efficiency.
    
    Lower cost and better token efficiency score higher.
    
    Args:
        result: API response (may contain cost_usd and/or usage fields)
        expected: Expected criteria (may contain max_cost_usd)
    
    Returns:
        Cost efficiency score 0.0-1.0
    """
    # Extract cost and usage from result
    cost_usd = None
    total_tokens = None
    
    if isinstance(result, dict):
        cost_usd = result.get('cost_usd')
        
        # Try to extract token usage
        if 'usage' in result and isinstance(result['usage'], dict):
            usage = result['usage']
            total_tokens = usage.get('total_tokens')
    
    # If no cost/usage info, return neutral score
    if cost_usd is None and total_tokens is None:
        return 0.8
    
    # Get target cost (default $0.01 = 1 cent)
    target_cost = expected.get('max_cost_usd', 0.01)
    
    scores = []
    
    # Score based on cost if available
    if cost_usd is not None:
        if cost_usd <= target_cost:
            # Perfect if under target
            if cost_usd <= target_cost * 0.5:
                scores.append(1.0)  # Excellent: under half the target
            else:
                # Linear scale
                ratio = cost_usd / target_cost
                scores.append(0.85 + (1.0 - ratio) * 0.15)
        else:
            # Penalty for being over budget
            overage_ratio = cost_usd / target_cost
            if overage_ratio <= 2.0:
                scores.append(max(0.5, 0.85 - (overage_ratio - 1.0) * 0.35))
            else:
                scores.append(max(0.2, 0.5 - (overage_ratio - 2.0) * 0.1))
    
    # Score based on token efficiency if available
    if total_tokens is not None:
        # Reward efficient token usage (fewer tokens = better)
        # Target: ~100 tokens for simple responses, ~500 for complex
        target_tokens = expected.get('target_tokens', 200)
        
        if total_tokens <= target_tokens:
            # Good efficiency
            scores.append(1.0)
        elif total_tokens <= target_tokens * 2:
            # Acceptable
            ratio = total_tokens / target_tokens
            scores.append(max(0.7, 1.0 - (ratio - 1.0) * 0.3))
        else:
            # Poor efficiency (too verbose)
            scores.append(0.6)
    
    # Return average of available scores, or neutral if none
    if scores:
        return sum(scores) / len(scores)
    return 0.8


# Optional: Score reasoning models differently (o3, o1, etc.)
def score_reasoning_response(
    result: Any,
    expected: Dict[str, Any],
    params: Dict[str, Any],
    metric: Optional[str] = None
) -> float:
    """
    Score reasoning model responses (o3, o1, etc.)
    
    For reasoning models, we weight correctness more heavily and are more
    tolerant of longer response times (since reasoning takes time).
    
    Scoring:
    - Correctness/completeness: 50%
    - Quality: 20%
    - Cost efficiency: 20%
    - Latency: 10% (less important for reasoning)
    
    Args:
        result: API response (including latency_ms, cost_usd if available)
        expected: Expected criteria (especially 'contains' keywords)
        params: API parameters
        metric: Specific metric to return
    
    Returns:
        Score 0.0-1.0
    """
    text = _extract_text(result)
    
    if not text:
        return 0.0
    
    # Calculate component scores
    completeness = _score_completeness(text, expected)
    quality = _score_quality(text)
    length = _score_length(text, expected)
    latency = _score_latency(result, expected)
    cost = _score_cost(result, expected)
    
    # Return specific metric if requested
    if metric:
        metric_map = {
            'completeness': completeness,
            'quality': quality,
            'length': length,
            'latency': latency,
            'cost': cost
        }
        if metric.lower() in metric_map:
            return round(min(1.0, max(0.0, metric_map[metric.lower()])), 3)
    
    # For reasoning models: prioritize correctness over speed
    # Reasoning takes time, so be more lenient on latency
    final_score = (
        completeness * 0.50 +
        quality * 0.20 +
        cost * 0.20 +
        latency * 0.10
    )
    
    return round(min(1.0, max(0.0, final_score)), 3)

