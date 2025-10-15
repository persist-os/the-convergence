"""
📝 Text Quality Evaluator

Evaluates text quality using readability, structure, clarity, and vocabulary metrics.
Useful for evaluating LLM text generation, content quality, and writing assessment.
"""
import re
import math
from typing import Dict, Any, Optional


def score_text_quality(
    result: Any,
    expected: Dict[str, Any],
    params: Dict[str, Any],
    metric: Optional[str] = None
) -> float:
    """
    Score text quality based on multiple dimensions.
    
    Args:
        result: Text result (string or dict with 'text' field)
        expected: Expected criteria:
            - target_audience: 'general'|'technical'|'academic' (default: 'general')
            - min_words: Minimum word count
            - max_words: Maximum word count
            - contains: List of required keywords
        params: API parameters (unused)
        metric: Specific metric name (unused)
    
    Returns:
        Quality score 0.0-1.0
    
    Example:
        result = "This is a well-written explanation..."
        expected = {
            'target_audience': 'general',
            'min_words': 50,
            'contains': ['explanation', 'example']
        }
        score = score_text_quality(result, expected, {})
    """
    # Extract text
    if isinstance(result, dict):
        text = result.get('text', result.get('content', str(result)))
    else:
        text = str(result)
    
    if not text or len(text.strip()) < 10:
        return 0.0
    
    # Get parameters
    target_audience = expected.get('target_audience', 'general')
    min_words = expected.get('min_words', 0)
    max_words = expected.get('max_words', float('inf'))
    required_keywords = expected.get('contains', [])
    
    scores = {}
    
    # 1. Word count check
    words = text.split()
    word_count = len(words)
    
    if min_words <= word_count <= max_words:
        scores['length'] = 1.0
    elif word_count < min_words:
        scores['length'] = word_count / min_words if min_words > 0 else 0.5
    else:
        scores['length'] = max_words / word_count if word_count > 0 else 0.5
    
    # 2. Keyword coverage
    if required_keywords:
        text_lower = text.lower()
        found = sum(1 for kw in required_keywords if kw.lower() in text_lower)
        scores['keywords'] = found / len(required_keywords)
    else:
        scores['keywords'] = 1.0
    
    # 3. Readability (basic)
    scores['readability'] = _basic_readability(text, target_audience)
    
    # 4. Structure
    scores['structure'] = _text_structure(text)
    
    # 5. Clarity
    scores['clarity'] = _text_clarity(text)
    
    # Weighted average
    weights = {
        'length': 0.15,
        'keywords': 0.25,
        'readability': 0.25,
        'structure': 0.20,
        'clarity': 0.15
    }
    
    final_score = sum(scores[k] * weights[k] for k in scores)
    return round(min(1.0, max(0.0, final_score)), 3)


def _basic_readability(text: str, target_audience: str) -> float:
    """Calculate basic readability score."""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if not sentences:
        return 0.5
    
    words = text.split()
    if not words:
        return 0.5
    
    avg_sentence_length = len(words) / len(sentences)
    avg_word_length = sum(len(w) for w in words) / len(words)
    
    # Rough grade level estimate
    estimated_grade = (0.4 * avg_sentence_length) + (0.5 * avg_word_length) - 2
    
    # Target grades
    target_grades = {
        'general': 10,
        'technical': 12,
        'academic': 14
    }
    target = target_grades.get(target_audience, 10)
    
    # Score based on deviation from target
    deviation = abs(estimated_grade - target)
    if deviation <= 2:
        return 1.0
    elif deviation <= 4:
        return 0.8
    elif deviation <= 6:
        return 0.6
    else:
        return 0.4


def _text_structure(text: str) -> float:
    """Evaluate text structure."""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    
    if len(sentences) < 2:
        return 0.5
    
    # Sentence length variation
    lengths = [len(s.split()) for s in sentences]
    avg_length = sum(lengths) / len(lengths)
    
    if len(lengths) > 1:
        variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        std_dev = math.sqrt(variance)
        variation_score = min(std_dev / 10, 1.0)
    else:
        variation_score = 0.5
    
    # Paragraph organization (look for line breaks)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if len(paragraphs) >= 2:
        org_score = 1.0
    elif len(paragraphs) == 1:
        org_score = 0.7
    else:
        org_score = 0.5
    
    return (variation_score * 0.5 + org_score * 0.5)


def _text_clarity(text: str) -> float:
    """Evaluate text clarity."""
    text_lower = text.lower()
    words = text.split()
    
    if not words:
        return 0.5
    
    # Passive voice indicators (lower is better)
    passive_markers = ['was', 'were', 'been', 'being', 'is', 'are']
    passive_count = sum(1 for word in text_lower.split() if word in passive_markers)
    passive_ratio = passive_count / len(words)
    passive_score = max(0, 1.0 - (passive_ratio * 3))
    
    # Vague words (lower is better)
    vague_words = ['thing', 'stuff', 'something', 'various', 'many', 'some']
    vague_count = sum(text_lower.count(vw) for vw in vague_words)
    vague_ratio = vague_count / len(words)
    vague_score = max(0, 1.0 - (vague_ratio * 10))
    
    # Specific numbers (higher is better)
    number_count = len(re.findall(r'\b\d+\b', text))
    specificity_score = min(number_count / 5, 1.0)
    
    return (passive_score * 0.4 + vague_score * 0.3 + specificity_score * 0.3)


