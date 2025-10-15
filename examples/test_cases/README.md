# Test Cases Guide

**Quick reference for creating and managing test cases for API optimization.**

---

## 📝 Test Case Format

All test cases follow this JSON structure:

```json
{
  "test_cases": [
    {
      "id": "unique_test_id",
      "description": "What this test validates",
      "input": {
        // API parameters (what you send to the API)
      },
      "expected": {
        // What you expect back (for evaluation)
      },
      "metadata": {
        "category": "qa",
        "difficulty": "easy",
        "weight": 1.0
      }
    }
  ]
}
```

---

## 🚀 Quick Start

### 1. **Copy a Template**

Choose the template matching your API type (see templates below).

### 2. **Fill in Your Data**

```json
{
  "test_cases": [
    {
      "id": "test_1",
      "description": "Test API with simple input",
      "input": {
        "prompt": "What is 2+2?"  // Your actual API input
      },
      "expected": {
        "contains": ["4"],           // What response should include
        "min_quality_score": 0.8     // Quality threshold
      },
      "metadata": {
        "category": "math",
        "weight": 1.0
      }
    }
  ]
}
```

### 3. **Save and Run**

```bash
# Save as test_cases.json
# Reference in optimization.yaml:
evaluation:
  test_cases:
    path: "test_cases.json"

# Run optimization
convergence optimize optimization.yaml
```

---

## 🧬 **NEW: Automatic Test Case Generation**

**Don't want to write 20 test cases manually?** Use evolutionary test case generation!

### How It Works

Start with **3-5 good test cases**, and the system automatically generates variants using:
- **Mutation**: Synonym replacement, number scaling, rephrasing
- **Crossover**: Combining elements from different tests
- **Complexity scaling**: Easy → Medium → Hard variations

### Example

**You write 3 tests:**
```json
{
  "test_cases": [
    {"id": "test_1", "input": {"prompt": "What is 5+3?"}},
    {"id": "test_2", "input": {"prompt": "Solve 10-4"}},
    {"id": "test_3", "input": {"prompt": "Calculate 2*6"}}
  ]
}
```

**System generates 9 more automatically:**
- `test_1_variant_1`: "Compute 7+4" (synonyms + numbers)
- `test_1_variant_2`: "Determine 3+2" (more variations)
- `test_2_variant_1`: "Figure out 15-6" (synonym + scaled numbers)
- ... and so on

### Configuration

Enable in your `optimization.yaml`:

```yaml
evaluation:
  test_cases:
    path: "test_cases.json"
    
    # Enable automatic generation
    augmentation:
      enabled: true              # Turn on/off
      mutation_rate: 0.3         # 30% chance to mutate elements
      crossover_rate: 0.2        # 20% chance of hybrid tests
      augmentation_factor: 2     # 2 variants per original
      preserve_originals: true   # Keep your original tests
```

### What Gets Mutated?

**Text prompts:**
```
Original: "Calculate the total cost of 15 apples"
Variant 1: "Compute the total cost of 22 apples"  (synonym + number)
Variant 2: "Determine the total price of 10 apples" (more changes)
```

**Numbers in text:**
```
Original: "15 apples at $2 each"
Variant 1: "22 apples at $2 each"  (scaled up 1.5x)
Variant 2: "7 apples at $3 each"   (scaled down + up)
```

**Emphasis:**
```
Original: "Solve this problem"
Variant 1: "Please solve this problem carefully"
Variant 2: "Clearly work through this question"
```

### Benefits

✅ **10x faster**: 3 test cases → 15 total in seconds  
✅ **Better coverage**: Automatic variations cover edge cases  
✅ **Less maintenance**: Update 3 files, not 15  
✅ **Robust configs**: Tested across more scenarios

### Parameters Explained

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mutation_rate` | 0.3 | Probability of mutating each element (0.1-0.5) |
| `crossover_rate` | 0.2 | Probability of combining tests (0.1-0.4) |
| `augmentation_factor` | 2 | Variants per original test (1-5) |
| `preserve_originals` | true | Keep original tests in final set |

**Total tests** = originals + (originals × factor)  
**Example**: 3 originals × factor 2 = 3 + 6 = **9 total tests**

---

## 📋 Templates by API Type

### LLM APIs (OpenAI, Anthropic, Gemini)

```json
{
  "test_cases": [
    {
      "id": "qa_basic",
      "description": "Simple question answering",
      "input": {
        "input": "What is the capital of France?"
      },
      "expected": {
        "contains": ["Paris"],
        "min_length": 5,
        "format": "text",
        "min_quality_score": 0.9
      },
      "metadata": {
        "category": "qa",
        "difficulty": "easy",
        "weight": 1.0
      }
    }
  ]
}
```

### Search APIs (Tavily, Exa, Serper)

```json
{
  "test_cases": [
    {
      "id": "search_tech",
      "description": "Technical search query",
      "input": {
        "query": "machine learning transformer models"
      },
      "expected": {
        "contains_keywords": ["transformer", "attention"],
        "min_results": 5,
        "max_latency_ms": 3000
      },
      "metadata": {
        "category": "technical",
        "weight": 1.0
      }
    }
  ]
}
```

### Web Automation (BrowserBase)

```json
{
  "test_cases": [
    {
      "id": "load_example",
      "description": "Load example.com",
      "url": "https://example.com",
      "selector": "h1",
      "expected": {
        "min_elements": 1,
        "contains_text": "Example Domain"
      },
      "success_criteria": {
        "load_time_ms": 3000,
        "requires_javascript": false
      }
    }
  ]
}
```

---

## ✅ Best Practices

### 1. **Start Small**
- Begin with 3-5 high-quality tests
- Use augmentation to generate more
- Verify generated tests make sense

### 2. **Representative Distribution**

Match your production workload:

```json
{
  "test_cases": [
    // 70% - Common production cases
    {"id": "common_1", "metadata": {"weight": 0.7}},
    {"id": "common_2", "metadata": {"weight": 0.7}},
    
    // 20% - Edge cases
    {"id": "edge_1", "metadata": {"weight": 0.2}},
    
    // 10% - Stress tests
    {"id": "stress_1", "metadata": {"weight": 0.1}}
  ]
}
```

### 3. **Clear Expected Outcomes**

✅ **Good** (specific):
```json
"expected": {
  "contains": ["Paris"],
  "min_length": 5,
  "max_latency_ms": 1000
}
```

❌ **Bad** (vague):
```json
"expected": {
  "correct": true,
  "fast": true
}
```

### 4. **Test Edge Cases**

Don't just test happy paths:

```json
{
  "test_cases": [
    {"id": "empty", "input": {"query": ""}},
    {"id": "very_long", "input": {"query": "word " * 500}},
    {"id": "special_chars", "input": {"query": "test@#$%"}},
    {"id": "non_english", "input": {"query": "测试"}}
  ]
}
```

### 5. **Use Weights**

Prioritize important tests:

```json
{
  "metadata": {
    "weight": 2.0  // 2x importance of normal tests
  }
}
```

---

## 🎯 Common Patterns

### Pattern 1: Quality vs Speed Trade-off

Test both extremes:

```json
{
  "test_cases": [
    {
      "id": "quality_focused",
      "input": {...},
      "metadata": {"category": "quality", "weight": 1.5}
    },
    {
      "id": "speed_focused",
      "input": {...},
      "metadata": {"category": "speed", "weight": 1.0}
    }
  ]
}
```

### Pattern 2: Difficulty Progression

Easy → Medium → Hard:

```json
{
  "test_cases": [
    {
      "id": "easy_1",
      "input": {"prompt": "2+2"},
      "metadata": {"difficulty": "easy"}
    },
    {
      "id": "medium_1",
      "input": {"prompt": "solve for x: 2x + 5 = 13"},
      "metadata": {"difficulty": "medium"}
    },
    {
      "id": "hard_1",
      "input": {"prompt": "prove the quadratic formula"},
      "metadata": {"difficulty": "hard"}
    }
  ]
}
```

### Pattern 3: Category Coverage

Cover all your use cases:

```json
{
  "test_cases": [
    {"id": "qa_1", "metadata": {"category": "qa"}},
    {"id": "creative_1", "metadata": {"category": "creative"}},
    {"id": "reasoning_1", "metadata": {"category": "reasoning"}},
    {"id": "code_1", "metadata": {"category": "code_generation"}}
  ]
}
```

---

## 🔧 Advanced: Custom Evaluators

For complex evaluation logic, create a custom evaluator:

```python
# evaluator.py
def score_response(result, expected, params, metric=None):
    """
    Custom scoring logic.
    
    Args:
        result: API response
        expected: Expected values from test case
        params: Config parameters used
        metric: Specific metric (optional)
    
    Returns:
        float: Score 0.0 to 1.0
    """
    score = 0.0
    
    # Check content
    if "answer" in result:
        if str(expected.get("contains")[0]) in str(result["answer"]):
            score += 0.5
    
    # Check latency
    if result.get("latency_ms", 9999) < expected.get("max_latency_ms", 1000):
        score += 0.3
    
    # Check length
    if len(str(result)) >= expected.get("min_length", 0):
        score += 0.2
    
    return min(1.0, score)
```

**Reference in YAML:**

```yaml
evaluation:
  custom_evaluator:
    enabled: true
    module: "evaluator"
    function: "score_response"
```

---

## 📊 Validation & Debugging

### Validate Test Case Format

```bash
# Check if your test cases are valid JSON
python -m json.tool test_cases.json

# Or use jq
jq . test_cases.json
```

### Test a Single Case

```python
# test_single.py
import json

with open("test_cases.json") as f:
    tests = json.load(f)

# Check structure
for test in tests["test_cases"]:
    assert "id" in test
    assert "input" in test
    assert "expected" in test
    print(f"✅ {test['id']} is valid")
```

### Debugging Tips

**Issue**: Tests failing unexpectedly  
**Solution**: Add logging to see actual vs expected

**Issue**: Too many tests (slow optimization)  
**Solution**: Reduce `augmentation_factor` or number of originals

**Issue**: Generated tests are nonsensical  
**Solution**: Lower `mutation_rate` to 0.2

**Issue**: Generated tests are too similar  
**Solution**: Increase `mutation_rate` to 0.4-0.5

---

## 📁 Example Files

See working examples in `examples/`:

- `ai/openai/openai_responses_tests.json` - LLM test cases
- `search/tavily/tavily_tests.json` - Search test cases  
- `web_browsing/browserbase/browserbase_tests.json` - Browser automation

**Copy and adapt them for your API!**

---

## 🎓 Evolution Engine Deep Dive

The test case evolution engine (`convergence/optimization/test_case_evolution.py`) uses:

### Mutation Strategies

1. **Synonym Replacement**
   - 100+ common words mapped to synonyms
   - Preserves capitalization and punctuation
   - Example: "Calculate" → "Compute", "Determine", "Find"

2. **Number Scaling**
   - Scales by factors: 0.5x, 0.75x, 1.25x, 1.5x, 2.0x
   - Preserves integer vs float types
   - Example: 15 → [7, 11, 19, 23, 30]

3. **Emphasis Adjustment**
   - Adds: "Please", "carefully", "clearly"
   - Removes existing emphasis words
   - Example: "Solve" → "Please solve carefully"

4. **Structural Mutations**
   - Sentence reordering
   - Complexity variations
   - Context adjustments

### Crossover Strategies

1. **Prompt Blending**
   - Combines parts of two prompts
   - Interleaves sentences
   - Example: "Calculate profit from 100 widgets" + "Determine revenue from 200 items" → "Calculate revenue from 100 items"

2. **Metadata Averaging**
   - Blends difficulty levels
   - Averages weights
   - Combines categories

3. **Numerical Averaging**
   - Averages expected values
   - Blends thresholds
   - Maintains reasonable ranges

### Implementation

```python
from convergence.optimization.test_case_evolution import TestCaseEvolutionEngine

# Create engine
engine = TestCaseEvolutionEngine(
    mutation_rate=0.3,
    crossover_rate=0.2,
    augmentation_factor=2,
    preserve_originals=True
)

# Generate variants
original_tests = [...]
augmented_tests = engine.augment_test_cases(original_tests)

# Analyze diversity
from convergence.optimization.test_case_evolution import TestCaseAnalyzer
diversity = TestCaseAnalyzer.analyze_diversity(augmented_tests)
print(f"Diversity score: {diversity['diversity']:.2f}")
```

---

## 💡 Pro Tips

### Tip 1: Incremental Augmentation

Start conservative, increase gradually:

```yaml
# Run 1: Conservative
augmentation_factor: 1

# Run 2: Moderate  
augmentation_factor: 2

# Run 3: Aggressive
augmentation_factor: 3
```

### Tip 2: Monitor Diversity

Check the console output:

```
🧬 Test Case Augmentation Enabled
   Original test cases: 3
   Augmented test cases: 9
   Diversity score: 0.78  ✅ Good!
```

Aim for **0.6-0.8** diversity score.

### Tip 3: Version Control

```bash
# Track both original and generated tests
git add test_cases.json
git add test_cases_augmented.json  # If saving augmented separately

# Commit with context
git commit -m "Add test cases for API optimization (3 original + augmentation)"
```

### Tip 4: Cost Awareness

More tests = more API calls:

```
3 tests × 4 configs × 3 generations = 36 API calls
9 tests × 4 configs × 3 generations = 108 API calls  (3x cost)
```

**Solution**: Reduce `population_size` or `generations` when using augmentation.

---

## 🔄 Workflow

### Recommended Process

1. **Write 3-5 high-quality tests** covering main use cases
2. **Enable augmentation** with `augmentation_factor: 2`
3. **Run optimization** and review results
4. **Check diversity score** (aim for 0.6-0.8)
5. **Adjust parameters** if needed:
   - Too similar? Increase `mutation_rate`
   - Too different? Decrease `mutation_rate`
   - Need more tests? Increase `augmentation_factor`
6. **Re-run and iterate**

---

## 📚 Further Reading

- **Main README**: `../README.md` - Project overview
- **Best Practices**: `../BEST_PRACTICES.md` - Optimization tips
- **Custom Evaluators**: `../../convergence/evaluators/README.md`
- **Evolution Engine**: `../../convergence/optimization/test_case_evolution.py` - Source code

---

**Questions?** Check `TROUBLESHOOTING.md` or open a GitHub issue.

**Ready to optimize!** 🚀
