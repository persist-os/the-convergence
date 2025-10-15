"""
Response processing utilities for optimization runs.
"""
from typing import Any, Optional


def extract_response_text(result: Any) -> Optional[str]:
    """
    Extract human-readable text from API response.
    
    Handles various API response formats:
    - OpenAI Responses API (output array)
    - OpenAI Chat Completions (choices array)
    - Direct text responses
    - Generic dict responses
    
    Args:
        result: API response result (dict, str, or other)
    
    Returns:
        Extracted text or None if not extractable
    """
    if not result:
        return None
    
    # Already a string
    if isinstance(result, str):
        return result
    
    if not isinstance(result, dict):
        return str(result)
    
    # OpenAI Responses API format
    # {"output": [{"content": [{"type": "output_text", "text": "..."}]}]}
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
        if texts:
            return '\n'.join(texts)
    
    # OpenAI Chat Completions format
    # {"choices": [{"message": {"content": "..."}}]}
    if 'choices' in result and isinstance(result['choices'], list):
        if len(result['choices']) > 0:
            choice = result['choices'][0]
            if isinstance(choice, dict):
                if 'message' in choice and isinstance(choice['message'], dict):
                    return choice['message'].get('content', '')
                elif 'text' in choice:
                    return choice['text']
    
    # Generic response with text/content fields
    if 'text' in result:
        return str(result['text'])
    if 'content' in result:
        return str(result['content'])
    if 'output_text' in result:
        return str(result['output_text'])
    
    # Last resort: convert to string
    return str(result)


def truncate_text(text: str, max_length: int = 200) -> str:
    """
    Truncate text for display with ellipsis.
    
    Args:
        text: Text to truncate
        max_length: Maximum length before truncation
    
    Returns:
        Truncated text with "..." if needed
    """
    if not text:
        return ""
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length] + "..."


def format_metrics_for_display(metrics: dict) -> str:
    """
    Format metrics dictionary for human-readable display.
    
    Args:
        metrics: Dictionary of metric name -> value
    
    Returns:
        Formatted string like "latency: 123ms, cost: $0.001"
    """
    if not metrics:
        return "N/A"
    
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            if key.endswith('_usd') or key.startswith('cost'):
                parts.append(f"{key}: ${value:.6f}")
            elif key.endswith('_ms'):
                parts.append(f"{key}: {value:.1f}ms")
            else:
                parts.append(f"{key}: {value:.4f}")
        else:
            parts.append(f"{key}: {value}")
    
    return ", ".join(parts)

