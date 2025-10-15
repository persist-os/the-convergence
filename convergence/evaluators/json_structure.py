"""
🏗️  JSON Structure Evaluator

Validates JSON structure, required fields, and data consistency.
Useful for evaluating structured API responses and data validation.
"""
import json
from typing import Dict, Any, Optional, List


def score_json_structure(
    result: Any,
    expected: Dict[str, Any],
    params: Dict[str, Any],
    metric: Optional[str] = None
) -> float:
    """
    Score JSON structure quality.
    
    Args:
        result: JSON result (string or dict)
        expected: Expected criteria:
            - required_fields: List of required field names
            - field_types: Dict of field -> expected type ('string', 'number', 'boolean', 'array', 'object')
            - min_fields: Minimum number of fields
            - max_fields: Maximum number of fields
        params: API parameters (unused)
        metric: Specific metric name (unused)
    
    Returns:
        Structure score 0.0-1.0
    
    Example:
        result = {"name": "Alice", "age": 30, "active": true}
        expected = {
            'required_fields': ['name', 'age'],
            'field_types': {'name': 'string', 'age': 'number'},
            'min_fields': 2
        }
        score = score_json_structure(result, expected, {})
    """
    # Parse JSON if string
    if isinstance(result, str):
        try:
            if result.strip().startswith('```'):
                # Extract from markdown code block
                result = result.split('```')[1]
                if result.startswith('json'):
                    result = result[4:]
                result = result.strip()
            data = json.loads(result)
        except json.JSONDecodeError:
            return 0.0
    elif isinstance(result, dict):
        data = result
    else:
        return 0.3  # Unknown structure
    
    scores = {}
    
    # 1. JSON validity (already passed if we got here)
    scores['valid'] = 1.0
    
    # 2. Required fields
    required_fields = expected.get('required_fields', [])
    if required_fields and isinstance(data, dict):
        present = sum(1 for field in required_fields if field in data)
        scores['required'] = present / len(required_fields)
    else:
        scores['required'] = 1.0
    
    # 3. Field types
    field_types = expected.get('field_types', {})
    if field_types and isinstance(data, dict):
        type_map = {
            'string': str,
            'str': str,
            'number': (int, float),
            'int': int,
            'float': float,
            'boolean': bool,
            'bool': bool,
            'array': list,
            'list': list,
            'object': dict,
            'dict': dict
        }
        
        correct = 0
        for field, expected_type in field_types.items():
            if field not in data:
                continue
            
            python_type = type_map.get(expected_type.lower(), str)
            if isinstance(data[field], python_type):
                correct += 1
        
        scores['types'] = correct / len(field_types) if field_types else 1.0
    else:
        scores['types'] = 1.0
    
    # 4. Field count
    if isinstance(data, dict):
        field_count = len(data)
        min_fields = expected.get('min_fields', 0)
        max_fields = expected.get('max_fields', float('inf'))
        
        if min_fields <= field_count <= max_fields:
            scores['count'] = 1.0
        elif field_count < min_fields:
            scores['count'] = field_count / min_fields if min_fields > 0 else 0.5
        else:
            scores['count'] = max_fields / field_count if field_count > 0 else 0.5
    else:
        scores['count'] = 0.7  # Not a dict
    
    # 5. Completeness (no null/empty values)
    if isinstance(data, dict):
        total_fields = len(data)
        if total_fields > 0:
            empty = sum(1 for v in data.values() if v is None or v == '' or v == [] or v == {})
            scores['complete'] = 1.0 - (empty / total_fields)
        else:
            scores['complete'] = 0.5
    else:
        scores['complete'] = 1.0
    
    # Weighted average
    weights = {
        'valid': 0.25,
        'required': 0.30,
        'types': 0.20,
        'count': 0.10,
        'complete': 0.15
    }
    
    final_score = sum(scores[k] * weights[k] for k in scores)
    return round(min(1.0, max(0.0, final_score)), 3)


def score_json_validity(
    result: Any,
    expected: Dict[str, Any],
    params: Dict[str, Any],
    metric: Optional[str] = None
) -> float:
    """
    Simple check: is it valid JSON?
    
    Returns 1.0 if valid, 0.0 if invalid.
    """
    if isinstance(result, (dict, list)):
        return 1.0
    
    if isinstance(result, str):
        try:
            json.loads(result)
            return 1.0
        except json.JSONDecodeError:
            return 0.0
    
    return 0.0

