# 🎯 Custom Evaluators Guide

## Overview

The Convergence supports custom evaluators for scoring API responses based on your specific criteria. This guide shows you how to add and use custom evaluators.

## Three Ways to Add Evaluators

### 1. **Local File (Simplest)** ✅ Recommended for Quick Testing

Place your evaluator Python file in the **same directory as your config YAML**:

```
my-project/
  ├── my_config.yaml
  └── my_evaluator.py  ← Your evaluator here
```

**my_config.yaml:**
```yaml
evaluation:
  custom_evaluator:
    enabled: true
    module: "my_evaluator"  # ← Just the filename (no .py)
    function: "score_response"
```

**my_evaluator.py:**
```python
def score_response(result, expected, params, metric=None):
    """
    Score the API response.
    
    Args:
        result: API response result
        expected: Expected output from test case
        params: API parameters used
        metric: Optional metric name
    
    Returns:
        float: Score between 0.0 and 1.0
    """
    # Your evaluation logic
    if "success" in str(result):
        return 0.9
    return 0.5
```

### 2. **Built-in Evaluators** (Professional)

Use evaluators that ship with The Convergence:

```yaml
evaluation:
  custom_evaluator:
    enabled: true
    module: "gemini_evaluator"  # ← Built-in
    function: "score_task_decomposition"
```

Available built-in evaluators:
- `gemini_evaluator.score_task_decomposition` - Task breakdown quality

### 3. **Installed Python Package** (Reusable)

Install your evaluator as a package and import it normally:

```bash
pip install my-evaluator-package
```

```yaml
evaluation:
  custom_evaluator:
    enabled: true
    module: "my_evaluator_package"
    function: "evaluate"
```

## Writing Custom Evaluators

### Minimal Example

```python
def score_my_api(result, expected, params, metric=None):
    """
    Simple evaluator - just return a score!
    """
    # Check if result contains expected keywords
    score = 0.0
    for keyword in expected.get('contains', []):
        if keyword.lower() in str(result).lower():
            score += 0.25
    
    return min(1.0, score)
```

### Advanced Example with Multiple Metrics

```python
import re

def score_code_generation(result, expected, params, metric=None):
    """
    Evaluate generated code quality.
    
    Checks multiple dimensions:
    - Syntax validity
    - Contains required elements
    - Complexity
    - Documentation
    """
    if not result or not isinstance(result, str):
        return 0.0
    
    scores = {}
    
    # 1. Syntax check
    try:
        compile(result, '<string>', 'exec')
        scores['syntax'] = 1.0
    except SyntaxError:
        scores['syntax'] = 0.0
    
    # 2. Required elements
    required = expected.get('contains', [])
    if required:
        found = sum(1 for req in required if req in result)
        scores['completeness'] = found / len(required)
    else:
        scores['completeness'] = 1.0
    
    # 3. Has docstrings?
    if '"""' in result or "'''" in result:
        scores['documentation'] = 1.0
    else:
        scores['documentation'] = 0.5
    
    # 4. Not too complex (heuristic)
    complexity = len(re.findall(r'\b(if|for|while|def|class)\b', result))
    if complexity <= 5:
        scores['complexity'] = 1.0
    elif complexity <= 10:
        scores['complexity'] = 0.7
    else:
        scores['complexity'] = 0.4
    
    # Weighted average
    weights = {
        'syntax': 0.3,
        'completeness': 0.3,
        'documentation': 0.2,
        'complexity': 0.2
    }
    
    final_score = sum(scores[k] * weights[k] for k in scores)
    return round(final_score, 3)
```

### Using the Base Class (Optional)

For better structure, inherit from `BaseEvaluator`:

```python
from convergence.evaluators.base import BaseEvaluator, score_wrapper

class MyEvaluator(BaseEvaluator):
    @staticmethod
    @score_wrapper  # Automatically clamps score to [0, 1]
    def evaluate(result, expected, params, metric=None):
        # Your logic here
        return some_score

# In your config, use:
# module: "my_module"
# function: "MyEvaluator.evaluate"
```

## Function Signature

All evaluators must follow this signature:

```python
def your_evaluator_function(
    result: Any,           # API response result
    expected: Any,         # Expected output from test case
    params: Dict[str, Any],  # API parameters used
    metric: Optional[str] = None  # Metric name (if checking specific metric)
) -> float:  # Return score 0.0 to 1.0
    """Your evaluator."""
    return 0.85
```

## Examples by Use Case

### JSON API Response Evaluation

```python
import json

def score_json_structure(result, expected, params, metric=None):
    """Check if JSON response has required structure."""
    try:
        # Parse result if string
        if isinstance(result, str):
            data = json.loads(result)
        else:
            data = result
        
        # Check required fields
        required_fields = expected.get('required_fields', [])
        if required_fields:
            present = sum(1 for field in required_fields if field in data)
            return present / len(required_fields)
        
        return 1.0
    except (json.JSONDecodeError, AttributeError):
        return 0.0
```

### Text Quality Evaluation

```python
import re

def score_text_quality(result, expected, params, metric=None):
    """Evaluate text quality based on length, structure, keywords."""
    text = str(result)
    
    score = 0.0
    
    # 1. Length check (50-500 words)
    word_count = len(text.split())
    if 50 <= word_count <= 500:
        score += 0.3
    elif 20 <= word_count < 50 or 500 < word_count <= 1000:
        score += 0.15
    
    # 2. Contains required keywords
    keywords = expected.get('contains', [])
    if keywords:
        keyword_score = sum(1 for kw in keywords if kw.lower() in text.lower())
        score += 0.4 * (keyword_score / len(keywords))
    else:
        score += 0.4
    
    # 3. Has proper structure (sentences, punctuation)
    sentences = len(re.findall(r'[.!?]+', text))
    if sentences >= 3:
        score += 0.3
    elif sentences >= 1:
        score += 0.15
    
    return min(1.0, score)
```

### ML Model Output Evaluation

```python
import numpy as np

def score_ml_predictions(result, expected, params, metric=None):
    """Evaluate ML model predictions."""
    try:
        # Extract predictions
        if isinstance(result, dict):
            predictions = result.get('predictions', [])
        else:
            predictions = result
        
        expected_labels = expected.get('labels', [])
        
        if not predictions or not expected_labels:
            return 0.5
        
        # Calculate accuracy
        correct = sum(1 for p, e in zip(predictions, expected_labels) if p == e)
        accuracy = correct / len(expected_labels)
        
        # Check confidence scores if available
        if isinstance(result, dict) and 'confidence' in result:
            confidences = result['confidence']
            avg_confidence = np.mean(confidences)
            
            # Weighted score: 70% accuracy, 30% confidence
            return 0.7 * accuracy + 0.3 * avg_confidence
        
        return accuracy
    except Exception:
        return 0.0
```

## Configuration in YAML

### Basic Setup

```yaml
evaluation:
  test_cases:
    inline:
      - input:
          prompt: "Generate a hello world function"
        expected:
          contains: ["def", "hello", "print"]
          min_lines: 3
  
  metrics:
    quality_score:
      weight: 0.6
      type: "higher_is_better"
      function: "custom"  # ← Use custom evaluator
  
  custom_evaluator:
    enabled: true
    module: "my_evaluator"
    function: "score_code_generation"
```

### Multiple Metrics with Custom Evaluation

```yaml
evaluation:
  metrics:
    # Built-in metrics
    success_rate:
      weight: 0.2
      type: "higher_is_better"
    
    latency_ms:
      weight: 0.2
      type: "lower_is_better"
      threshold: 5000
    
    # Custom quality metric
    quality_score:
      weight: 0.6
      type: "higher_is_better"
      function: "custom"
  
  custom_evaluator:
    enabled: true
    module: "my_evaluator"
    function: "evaluate_quality"
```

## Testing Your Evaluator

Create a test script to verify your evaluator:

```python
# test_my_evaluator.py
from my_evaluator import score_response

# Test case 1: Good response
result = {"status": "success", "data": [1, 2, 3]}
expected = {"contains": ["success"], "min_items": 3}
score = score_response(result, expected, {})
print(f"Test 1 Score: {score}")  # Should be high

# Test case 2: Bad response
result = {"status": "error"}
expected = {"contains": ["success"]}
score = score_response(result, expected, {})
print(f"Test 2 Score: {score}")  # Should be low

assert 0.0 <= score <= 1.0, "Score must be between 0 and 1"
print("✅ All tests passed!")
```

Run it:
```bash
python test_my_evaluator.py
```

## Debugging

If your evaluator isn't loading, check:

1. **File location**: Is `my_evaluator.py` in the same directory as your YAML?
2. **Function name**: Does it match exactly in your YAML?
3. **Syntax errors**: Run `python my_evaluator.py` to check for errors
4. **Return type**: Make sure you return a float between 0 and 1

The error message will tell you what's wrong:
```
Failed to load custom evaluator my_evaluator.score_response
Tried:
  1. Built-in: convergence.evaluators.my_evaluator
  2. Installed module: my_evaluator
  3. Local file: my_evaluator.py in config directory
```

## Best Practices

1. **Return float between 0.0 and 1.0** - This is enforced
2. **Handle errors gracefully** - Return 0.0 or 0.5 on error
3. **Document your function** - Add docstring explaining what you're measuring
4. **Test with edge cases** - Empty results, malformed data, etc.
5. **Keep it fast** - Evaluation runs many times during optimization
6. **Make it deterministic** - Same input should give same score

## Built-in Evaluators Reference

### gemini_evaluator.score_task_decomposition

Evaluates how well Gemini breaks down complex tasks into steps.

**What it checks:**
- Number of steps (reasonable count)
- Step clarity (action verbs, proper length)
- Coverage of expected concepts
- Logical structure (sequential, dependencies)

**Configuration:**
```yaml
evaluation:
  test_cases:
    inline:
      - input:
          prompt: "Break down: Research AI, build prototype, evaluate"
        expected:
          contains: ["research", "build", "evaluate"]
          min_steps: 3
  
  custom_evaluator:
    enabled: true
    module: "gemini_evaluator"
    function: "score_task_decomposition"
```

**Score breakdown:**
- 25%: Number of steps
- 30%: Step clarity (actionable?)
- 25%: Concept coverage
- 20%: Logical structure

## Common Patterns

### Pattern 1: Keyword Matching

```python
def score_contains_keywords(result, expected, params, metric=None):
    text = str(result).lower()
    keywords = expected.get('contains', [])
    if not keywords:
        return 1.0
    matches = sum(1 for kw in keywords if kw.lower() in text)
    return matches / len(keywords)
```

### Pattern 2: Numeric Threshold

```python
def score_within_range(result, expected, params, metric=None):
    try:
        value = float(result.get('value', 0))
        min_val = expected.get('min', 0)
        max_val = expected.get('max', 100)
        
        if min_val <= value <= max_val:
            return 1.0
        return 0.0
    except:
        return 0.0
```

### Pattern 3: Similarity Matching

```python
from difflib import SequenceMatcher

def score_similarity(result, expected, params, metric=None):
    result_text = str(result)
    expected_text = str(expected.get('text', ''))
    
    ratio = SequenceMatcher(None, result_text, expected_text).ratio()
    return ratio
```

## Next Steps

1. ✅ Create your evaluator function
2. ✅ Place it in the same directory as your config YAML
3. ✅ Update your YAML to use it
4. ✅ Test it with `convergence optimize your_config.yaml`
5. ✅ Iterate based on results

## Need Help?

- Check `examples/` for complete working examples
- Read `convergence/evaluators/gemini_evaluator.py` for a full implementation
- The framework will give detailed error messages if something's wrong

Happy evaluating! 🎯

