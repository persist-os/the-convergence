# 🚀 OpenAI Responses API - Quick Start

Simple model comparison for OpenAI's Responses API.

## What This Does

Quick test of OpenAI models (gpt-4o-mini, gpt-4-turbo) with **only ~10 API calls total**.

Perfect for:
- ✅ Validating setup works
- ✅ Testing the system (~$0.02-0.05 cost)
- ✅ Quick model comparison

## Setup (2 minutes)

### 1. Set your API key
```bash
export OPENAI_API_KEY="sk-..."
```

### 2. Run optimization
```bash
cd /Users/ariahan/Documents/div/both/the_convergence
convergence optimize examples/ai/openai_responses_optimization.yaml
```

### 3. View results
Results save to `./results/openai_responses_optimization/`

## What Gets Tested

**Models:**
- `gpt-4o-mini` - Fast and cost-efficient
- `gpt-4-turbo` - More capable, higher cost

**Test Cases** (10 different prompts):
1. Creative writing (bedtime story)
2. Factual QA (capital of France)
3. Math reasoning (distance calculation)
4. Concept explanation (photosynthesis)
5. List generation (programming languages)
6. Code generation (Python function)
7. Summarization
8. Translation (English to Spanish)
9. Comparison (RAM vs ROM)
10. Analogy creation

## How It Scores

Each response is scored on 3 things:

1. **Completeness (50%)**: Does it have the right keywords?
2. **Quality (30%)**: Is it well-formed text?
3. **Length (20%)**: Is it the right length?

Plus built-in metrics:
- Latency (response time)
- Cost per call
- Token efficiency

## Results

After running, you'll get:
- Best model for your use case
- Performance breakdown by test category
- Cost vs quality analysis
- Exportable config with best settings

## Files Created

✅ `convergence/evaluators/openai_responses.py` - Built-in evaluator  
✅ `examples/test_cases/openai_responses_tests.json` - 10 test cases  
✅ `examples/ai/openai_responses_optimization.yaml` - Config file  

## For Real Optimization

This config is in **MINIMAL TEST MODE** (only 10 API calls).

For real optimization, edit the YAML:

```yaml
execution:
  experiments_per_generation: 50  # Change from 10 to 50+
  
evolution:
  generations: 5  # Change from 1 to 5+
```

This will give you:
- More comprehensive testing
- Better statistical confidence
- Full parameter exploration

## Customizing

### Add Your Own Test Cases

Edit `examples/test_cases/openai_responses_tests.json`:

```json
{
  "id": "my_test",
  "input": {
    "input": "Your prompt here"
  },
  "expected": {
    "contains": ["keyword1", "keyword2"],
    "min_length": 20
  }
}
```

### Test Different Models

Edit `openai_responses_optimization.yaml`:

```yaml
search_space:
  parameters:
    model:
      type: "categorical"
      values:
        - "gpt-4o"
        - "gpt-4o-mini"
        - "gpt-4-turbo"
```

### Adjust Scoring Weights

Edit the evaluator or modify metric weights in the YAML:

```yaml
evaluation:
  metrics:
    response_quality:
      weight: 0.40  # Change this
    latency_ms:
      weight: 0.25  # And this
    cost_per_call:
      weight: 0.20  # And this
```

## Architecture

**api_caller.py**: Makes raw HTTP POST requests to OpenAI  
→ Generic, works with any HTTP API  
→ Tracks latency, cost, success rate  
→ No model-specific logic  

**openai_responses.py**: Evaluates the response quality  
→ Parses `output_text` from response  
→ Scores on completeness, quality, length  
→ Returns 0.0-1.0 score  

**Test cases**: Your validation data  
→ Input prompts + expected outputs  
→ Easily customizable JSON file  

**YAML config**: Ties everything together  
→ API endpoint, auth, parameters  
→ Evaluation metrics and weights  
→ Optimization settings  

## Common Issues

### "Test cases file not found"
Make sure you're running from the project root:
```bash
cd /Users/ariahan/Documents/div/both/the_convergence
convergence optimize examples/ai/openai_responses_optimization.yaml
```

### "API key not found"
Export your key:
```bash
export OPENAI_API_KEY="sk-..."
```

### "Module 'openai_responses' not found"
The evaluator is built-in, but make sure you're using the latest version:
```bash
cd /Users/ariahan/Documents/div/both/the_convergence
pip install -e .
```

## Next Steps

1. ✅ Run basic optimization (2 models, 10 tests)
2. ✅ Review results in `./results/`
3. ✅ Add your own test cases
4. ✅ Test more models or parameters
5. ✅ Export best config for production

## Documentation

- **Architecture**: `examples/ARCHITECTURE_GUIDE.md`
- **API Caller vs LiteLLM**: `examples/AI_INTEGRATION_ARCHITECTURE.md`
- **Test Cases**: `examples/test_cases/README.md`
- **Evaluators**: `convergence/evaluators/README.md`

---

**Simple Goal**: Find the best OpenAI model for your specific use case by testing them on your actual prompts. 🎯

