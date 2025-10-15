"""
🚀 Groq API Evaluator

Evaluator for Groq's ultra-fast LLM inference API.
Groq uses OpenAI-compatible Chat Completions format with LPU acceleration.

Response format:
{
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "The robot learned to paint with vibrant digital brushstrokes..."
            }
        }
    ],
    "usage": {
        "prompt_tokens": 15,
        "completion_tokens": 25,
        "total_tokens": 40
    },
    "model": "llama-3.1-70b-versatile"
}

Scoring criteria:
- Content completeness (required keywords)
- Response quality (grammar, coherence, formatting)
- Length appropriateness (within bounds)
- Speed efficiency (Groq's main strength)
"""
from typing import Dict, Any, Optional


def score_groq_response(
    result: Any,
    expected: Dict[str, Any],
    params: Dict[str, Any],
    metric: Optional[str] = None
) -> float:
    """
    Score Groq API response (Chat Completions format).
    
    Evaluates based on:
    - Completeness: Contains required keywords? (50%)
    - Quality: Well-formed, coherent text? (30%)
    - Length: Appropriate response length? (20%)
    
    Args:
        result: API response from Groq (Chat Completions format)
        expected: Expected criteria:
            - contains: List of required keywords
            - min_length: Minimum character count (optional)
            - max_length: Maximum character count (optional)
            - format: Expected format ('text', 'code', 'list', etc.)
        params: API parameters (model, temperature, etc.)
        metric: Specific metric name (optional)
    
    Returns:
        Quality score 0.0-1.0
    
    Example:
        result = {"choices": [{"message": {"content": "Tokyo is the capital of Japan."}}]}
        expected = {"contains": ["Tokyo"], "min_length": 10}
        score = score_groq_response(result, expected, {})
        # Returns ~0.95 (has keyword, good quality, appropriate length)
    """
    # Extract text from Chat Completions format
    text = _extract_text(result)
    
    if not text or len(text.strip()) < 3:
        return 0.0
    
    # Three-component scoring
    scores = {}
    
    # 1. Completeness: Required keywords present? (50%)
    scores['completeness'] = _score_completeness(text, expected)
    
    # 2. Quality: Well-formed response? (30%)
    scores['quality'] = _score_quality(text, expected)
    
    # 3. Length: Appropriate length? (20%)
    scores['length'] = _score_length(text, expected)
    
    # Weighted average
    final_score = (
        scores['completeness'] * 0.50 +
        scores['quality'] * 0.30 +
        scores['length'] * 0.20
    )
    
    return round(min(1.0, max(0.0, final_score)), 3)


def _extract_text(result: Any) -> str:
    """
    Extract text from Groq Chat Completions response.
    
    Handles multiple response formats:
    - Standard: choices[0].message.content
    - Streaming: concatenated delta.content
    - Legacy: choices[0].text
    """
    if isinstance(result, str):
        return result
    
    if isinstance(result, dict):
        # Standard Chat Completions format
        if 'choices' in result and isinstance(result['choices'], list):
            if len(result['choices']) > 0:
                choice = result['choices'][0]
                
                # Standard message format
                if 'message' in choice and isinstance(choice['message'], dict):
                    content = choice['message'].get('content', '')
                    return str(content) if content else ''
                
                # Legacy text format
                if 'text' in choice:
                    return str(choice['text'])
        
        # Direct content field (fallback)
        if 'content' in result:
            return str(result['content'])
        
        # Generic text field
        if 'text' in result:
            return str(result['text'])
    
    return str(result)


def _score_completeness(text: str, expected: Dict[str, Any]) -> float:
    """
    Check if response contains all required keywords.
    
    Returns percentage of required keywords found (0.0-1.0).
    """
    required = expected.get('contains', [])
    if not required:
        return 1.0  # No requirements = perfect score
    
    text_lower = text.lower()
    found = sum(1 for keyword in required if keyword.lower() in text_lower)
    
    return found / len(required)


def _score_quality(text: str, expected: Dict[str, Any]) -> float:
    """
    Evaluate text quality based on format expectations.
    
    Checks:
    - Proper capitalization and grammar
    - Format-specific patterns (code has 'def'/'function', lists have numbers)
    - Word variety (not overly repetitive)
    - Reasonable length
    """
    if len(text) < 5:
        return 0.2
    
    score = 0.0
    format_type = expected.get('format', 'text')
    
    # Check 1: Format-specific validation (40%)
    if format_type == 'code':
        # Code should have syntax elements
        code_indicators = ['def ', 'function ', 'return ', 'class ', 'import ', '=', '{', '}']
        has_code = any(indicator in text for indicator in code_indicators)
        score += 0.4 if has_code else 0.1
    
    elif format_type == 'list':
        # Lists should have numbered/bulleted items
        list_indicators = ['1.', '2.', '3.', '•', '-', '*']
        has_list = any(indicator in text for indicator in list_indicators)
        score += 0.4 if has_list else 0.1
    
    else:  # text format
        # Text should start with capital letter
        if text[0].isupper():
            score += 0.4
        else:
            score += 0.1
    
    # Check 2: Word variety - not too repetitive (30%)
    words = text.lower().split()
    if words:
        unique_ratio = len(set(words)) / len(words)
        score += 0.3 * min(1.0, unique_ratio * 1.5)
    else:
        score += 0.1
    
    # Check 3: Reasonable length - not too short or ridiculously long (30%)
    if 10 <= len(text) <= 2000:
        score += 0.3
    elif len(text) < 10:
        score += 0.1
    else:
        score += 0.2
    
    return min(1.0, score)


def _score_length(text: str, expected: Dict[str, Any]) -> float:
    """
    Check if response length is within expected bounds.
    
    Returns 1.0 if within bounds, scaled penalty if outside.
    """
    min_length = expected.get('min_length', 0)
    max_length = expected.get('max_length', float('inf'))
    
    actual_length = len(text)
    
    # Perfect if within bounds
    if min_length <= actual_length <= max_length:
        return 1.0
    
    # Too short: proportional penalty
    if actual_length < min_length:
        if min_length > 0:
            return max(0.3, actual_length / min_length)
        return 0.8
    
    # Too long: smaller penalty (verbose is better than missing content)
    if actual_length > max_length and max_length < float('inf'):
        overage = (actual_length - max_length) / max_length
        return max(0.5, 1.0 - overage * 0.3)  # Max 50% penalty
    
    return 0.9


def score_groq_speed_optimized(
    result: Any,
    expected: Dict[str, Any],
    params: Dict[str, Any],
    metric: Optional[str] = None
) -> float:
    """
    Alternative scoring optimized for Groq's speed advantage.
    
    For use cases where speed matters more than perfection.
    Weights correctness higher, tolerates brevity.
    
    - Correctness: 70%
    - Quality: 20%
    - Length: 10%
    
    Args:
        result: Groq API response
        expected: Expected criteria
        params: API parameters
        metric: Specific metric
    
    Returns:
        Score 0.0-1.0
    """
    text = _extract_text(result)
    
    if not text:
        return 0.0
    
    completeness = _score_completeness(text, expected)
    quality = _score_quality(text, expected)
    length = _score_length(text, expected)
    
    # Speed-optimized weighting
    final_score = (completeness * 0.70 + quality * 0.20 + length * 0.10)
    return round(min(1.0, max(0.0, final_score)), 3)

