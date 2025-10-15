"""
💎 Code Quality Evaluator

Evaluates code quality using syntax checking, complexity, and structure analysis.
Useful for evaluating code generation APIs and programming assistants.
"""
import ast
import re
from typing import Dict, Any, Optional


def score_code_quality(
    result: Any,
    expected: Dict[str, Any],
    params: Dict[str, Any],
    metric: Optional[str] = None
) -> float:
    """
    Score code quality based on multiple dimensions.
    
    Args:
        result: Code result (string or dict with 'code' field)
        expected: Expected criteria:
            - language: 'python'|'javascript'|'other' (default: 'python')
            - contains: List of required elements (functions, classes, keywords)
            - min_lines: Minimum line count
            - max_complexity: Maximum acceptable complexity
        params: API parameters (unused)
        metric: Specific metric name (unused)
    
    Returns:
        Quality score 0.0-1.0
    
    Example:
        result = "def hello():\\n    print('world')"
        expected = {
            'language': 'python',
            'contains': ['def', 'print'],
            'min_lines': 2
        }
        score = score_code_quality(result, expected, {})
    """
    # Extract code
    if isinstance(result, dict):
        code = result.get('code', result.get('content', str(result)))
    else:
        code = str(result)
    
    if not code or len(code.strip()) < 5:
        return 0.0
    
    # Get parameters
    language = expected.get('language', 'python')
    required_elements = expected.get('contains', [])
    min_lines = expected.get('min_lines', 0)
    max_complexity = expected.get('max_complexity', 20)
    
    scores = {}
    
    # Language-specific evaluation
    if language == 'python':
        scores = _eval_python_code(code, required_elements, min_lines, max_complexity)
    elif language == 'javascript':
        scores = _eval_javascript_code(code, required_elements, min_lines)
    else:
        scores = _eval_generic_code(code, required_elements, min_lines)
    
    # Weighted average
    weights = {
        'syntax': 0.30,
        'completeness': 0.25,
        'complexity': 0.20,
        'structure': 0.15,
        'quality': 0.10
    }
    
    final_score = sum(scores.get(k, 0.5) * weights[k] for k in weights)
    return round(min(1.0, max(0.0, final_score)), 3)


def _eval_python_code(
    code: str,
    required_elements: list,
    min_lines: int,
    max_complexity: int
) -> Dict[str, float]:
    """Evaluate Python code specifically."""
    scores = {}
    
    # 1. Syntax check
    try:
        tree = ast.parse(code)
        scores['syntax'] = 1.0
    except SyntaxError:
        return {'syntax': 0.0, 'completeness': 0.0, 'complexity': 0.0, 'structure': 0.0, 'quality': 0.0}
    
    # 2. Required elements
    if required_elements:
        code_lower = code.lower()
        found = sum(1 for elem in required_elements if elem.lower() in code_lower)
        scores['completeness'] = found / len(required_elements)
    else:
        scores['completeness'] = 1.0
    
    # 3. Complexity (count decision points)
    complexity = 0
    functions = 0
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
            complexity += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions += 1
    
    avg_complexity = complexity / functions if functions > 0 else complexity
    
    if avg_complexity <= max_complexity * 0.5:
        scores['complexity'] = 1.0
    elif avg_complexity <= max_complexity:
        scores['complexity'] = 0.7
    else:
        scores['complexity'] = 0.4
    
    # 4. Structure
    line_count = len(code.split('\n'))
    
    # Check for functions/classes
    has_functions = bool([n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))])
    has_classes = bool([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
    
    structure_score = 0.5
    if has_functions or has_classes:
        structure_score += 0.3
    if line_count >= min_lines:
        structure_score += 0.2
    
    scores['structure'] = min(1.0, structure_score)
    
    # 5. Quality indicators
    quality_score = 0.5
    
    # Has docstrings?
    docstrings = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if ast.get_docstring(node):
                docstrings += 1
    
    if docstrings > 0:
        quality_score += 0.2
    
    # Has type hints?
    type_hints = len(re.findall(r':\s*\w+', code))
    if type_hints > 0:
        quality_score += 0.2
    
    # Has error handling?
    if 'try:' in code or 'except' in code:
        quality_score += 0.1
    
    scores['quality'] = min(1.0, quality_score)
    
    return scores


def _eval_javascript_code(
    code: str,
    required_elements: list,
    min_lines: int
) -> Dict[str, float]:
    """Evaluate JavaScript code (basic)."""
    scores = {}
    
    # 1. Basic syntax check
    # Check for matching braces/brackets/parens
    if code.count('{') == code.count('}') and code.count('(') == code.count(')'):
        scores['syntax'] = 0.8  # Can't fully validate without JS parser
    else:
        scores['syntax'] = 0.3
    
    # 2. Required elements
    if required_elements:
        code_lower = code.lower()
        found = sum(1 for elem in required_elements if elem.lower() in code_lower)
        scores['completeness'] = found / len(required_elements)
    else:
        scores['completeness'] = 1.0
    
    # 3. Complexity (rough estimate)
    complexity_keywords = ['if', 'for', 'while', 'switch', 'case']
    complexity = sum(len(re.findall(rf'\b{kw}\b', code)) for kw in complexity_keywords)
    
    if complexity <= 10:
        scores['complexity'] = 1.0
    elif complexity <= 20:
        scores['complexity'] = 0.7
    else:
        scores['complexity'] = 0.4
    
    # 4. Structure
    line_count = len(code.split('\n'))
    has_functions = bool(re.search(r'\bfunction\b|\(\)\s*=>', code))
    
    structure_score = 0.5
    if has_functions:
        structure_score += 0.3
    if line_count >= min_lines:
        structure_score += 0.2
    
    scores['structure'] = min(1.0, structure_score)
    
    # 5. Quality
    quality_score = 0.5
    if 'const' in code or 'let' in code:
        quality_score += 0.2
    if '// ' in code or '/*' in code:
        quality_score += 0.2
    if 'try' in code or 'catch' in code:
        quality_score += 0.1
    
    scores['quality'] = min(1.0, quality_score)
    
    return scores


def _eval_generic_code(
    code: str,
    required_elements: list,
    min_lines: int
) -> Dict[str, float]:
    """Evaluate generic code (language-agnostic)."""
    scores = {}
    
    # 1. Basic syntax - just check it's not empty
    scores['syntax'] = 0.7 if code.strip() else 0.0
    
    # 2. Required elements
    if required_elements:
        code_lower = code.lower()
        found = sum(1 for elem in required_elements if elem.lower() in code_lower)
        scores['completeness'] = found / len(required_elements)
    else:
        scores['completeness'] = 1.0
    
    # 3. Complexity (very rough)
    control_flow = len(re.findall(r'\b(if|for|while|loop|case|when)\b', code, re.IGNORECASE))
    if control_flow <= 10:
        scores['complexity'] = 1.0
    else:
        scores['complexity'] = 0.6
    
    # 4. Structure
    line_count = len(code.split('\n'))
    scores['structure'] = 1.0 if line_count >= min_lines else (line_count / min_lines if min_lines > 0 else 0.5)
    
    # 5. Quality
    has_comments = bool(re.search(r'#|//|/\*', code))
    scores['quality'] = 0.7 if has_comments else 0.5
    
    return scores


def score_python_syntax(
    result: Any,
    expected: Dict[str, Any],
    params: Dict[str, Any],
    metric: Optional[str] = None
) -> float:
    """
    Simple check: is it valid Python syntax?
    
    Returns 1.0 if valid, 0.0 if invalid.
    """
    if isinstance(result, dict):
        code = result.get('code', result.get('content', str(result)))
    else:
        code = str(result)
    
    try:
        ast.parse(code)
        return 1.0
    except SyntaxError:
        return 0.0


