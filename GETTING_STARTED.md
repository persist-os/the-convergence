# 🚀 Getting Started with The Convergence

**Quick start guide for optimizing your API in under 5 minutes.**

---

## 📦 Installation

```bash
pip install the-convergence
```

**Or install from source:**

```bash
git clone https://github.com/persist-os/the-convergence.git
cd the-convergence
pip install -e .
```

---

## ⚡ Quick Start (2 Minutes)

### Option 1: Interactive Setup (Recommended)

```bash
# Interactive wizard guides you through setup
convergence init

# Then run optimization
convergence optimize optimization.yaml
```

### Option 2: Use a Preset

```bash
# Copy an example config
cp examples/ai/openai/openai_responses_optimization.yaml my_config.yaml

# Set your API key
export OPENAI_API_KEY="sk-..."

# Run optimization
convergence optimize my_config.yaml
```

---

## 🎯 Your First Optimization

### Step 1: Set Up Your API Key

```bash
# For OpenAI
export OPENAI_API_KEY="sk-..."

# For other APIs, see examples/
```

### Step 2: Run Interactive Setup

```bash
convergence init
```

**This wizard will:**

1. Let you choose a template (OpenAI, BrowserBase, Groq, etc.)
2. Configure optimization settings (intensity, parallelism, output)
3. Configure Legacy System (continuous learning - enabled by default)
4. Optionally enable Agent Society (experimental)

**Output files:**

- `optimization.yaml` - Your configuration
- `test_cases.json` - Test cases
- `evaluator.py` - Evaluation logic (if needed)

### Step 3: Run Optimization

```bash
convergence optimize optimization.yaml
```

**Watch it work:**

```
🚀 STARTING API OPTIMIZATION
======================================================================
API: openai_responses
Generations: 2
Population Size: 2
======================================================================

🧬 GENERATION 1/2
📊 Evaluating 2 configurations...

🔬 Config [1/2]: model=gpt-4o-mini, temperature=0.7
   📝 Test 1/3: capital_question
   Score: 0.950 | Latency: 450ms
   📝 Test 2/3: simple_math
   Score: 0.900 | Latency: 380ms
   📝 Test 3/3: creative_simple
   Score: 0.850 | Latency: 520ms
   ✅ Aggregate Score: 0.9000

✅ Optimization Complete!
   Best config: model=gpt-4o-mini, temperature=0.7
   Score: 0.90
   Cost: $0.0003/call
```

### Step 4: Check Results

```bash
# Results are saved in:
./results/
  ├── optimization_results.json   # Raw data
  ├── optimization_report.md      # Markdown report
  └── best_config.py              # Winner config
```

---

## 📚 Example Configurations

We provide **examples** in the `examples/` directory, grouped by integration type:

### AI & LLM APIs

- OpenAI (ChatGPT) — `examples/ai/openai/`
- Groq — `examples/ai/groq/`
- Azure OpenAI — `examples/ai/azure/`

### Web Automation

- BrowserBase — `examples/web_browsing/browserbase/`

### Test Case Templates & Guides

- Test case guides — `examples/test_cases/`

### Best Practices & Tutorials

- Example best practices — `examples/BEST_PRACTICES.md`

These cover basic LLM API optimization, prompt tuning, search API evaluation, custom evaluators, advanced usage patterns, and more.

**See full list:** `examples/` directory

---

## 🎓 Understanding the Config

### Minimal Config Structure

```yaml
# 1. Your API
api:
  name: "my_api"
  endpoint: "https://api.example.com/v1/endpoint"
  auth:
    type: "bearer"
    token_env: "MY_API_KEY"  # Environment variable name

# 2. What to optimize
search_space:
  parameters:
    model: ["gpt-4o-mini", "gpt-4o"]
    temperature: [0.3, 0.5, 0.7, 0.9]

# 3. How to evaluate
evaluation:
  test_cases:
    path: "test_cases.json"
  metrics:
    quality: {weight: 0.6}
    latency_ms: {weight: 0.3}
    cost_usd: {weight: 0.1}

# 4. Optimization settings
optimization:
  algorithm: "mab_evolution"
  evolution:
    population_size: 4
    generations: 3
```

---

## 🔧 Customization

### Create Your Own Evaluator

If the built-in evaluators don't fit your needs:

```python
# evaluator.py
def score_response(result, expected, params, metric=None):
    """
    Custom evaluation logic.
    
    Args:
        result: API response
        expected: Expected values from test case
        params: Config parameters used
        metric: Specific metric being evaluated (optional)
    
    Returns:
        float: Score between 0.0 and 1.0
    """
    score = 0.0
    
    # Your custom logic here
    if "answer" in result:
        if result["answer"] == expected.get("answer"):
            score += 0.5
    
    if result.get("latency_ms", 9999) < 500:
        score += 0.5
    
    return score
```

**Then reference it in your config:**

```yaml
evaluation:
  custom_evaluator:
    enabled: true
    module: "evaluator"
    function: "score_response"
```

---

## 💡 Pro Tips

### 1. Start Small

- Begin with 2-3 test cases
- Use `population_size: 2` and `generations: 2`
- Verify it works, then scale up

### 2. Check Logs

- Logs show what's happening during optimization
- Look for errors in API calls or evaluation

### 3. Iterate

- Run optimization multiple times
- Each run builds on previous knowledge (Legacy System enabled by default)
- Gets better over time automatically

### 4. Version Control

```bash
git add optimization.yaml test_cases.json
git commit -m "API optimization config for production"
```

---

## 🐛 Troubleshooting

### "Command not found: convergence"

```bash
# Reinstall
pip install --upgrade the-convergence

# Or check PATH
which convergence
```

### "API key not found"

```bash
# Make sure environment variable is set
echo $OPENAI_API_KEY

# Set it if missing
export OPENAI_API_KEY="sk-..."
```

### "No module named 'convergence'"

```bash
# Reinstall with dependencies
pip install --upgrade the-convergence
```

### Optimization takes too long

```yaml
# Reduce these in optimization.yaml:
optimization:
  evolution:
    population_size: 2  # Fewer configs per generation
    generations: 2      # Fewer generations
  execution:
    parallel_workers: 1  # Sequential (slower but safer)
```

### Rate limit errors

```yaml
# For APIs with strict limits:
optimization:
  execution:
    parallel_workers: 1    # One at a time
    max_retries: 1         # Fewer retries
    delay_between_calls: 1 # 1 second delay
```

---

## 📖 Next Steps

1. **Read examples** - Check `examples/BEST_PRACTICES.md`
2. **Explore configs** - Browse `examples/` for your API type
3. **Join community** - GitHub Discussions
4. **Contribute** - See `CONTRIBUTING.md`

---

## 🆘 Need Help?

- **Documentation**: Check `README.md` and `TROUBLESHOOTING.md`
- **Issues**: [GitHub Issues](https://github.com/persist-os/the-convergence/issues)
- **Security**: See `SECURITY.md` for vulnerability reporting

---

**Happy optimizing! 🚀**
