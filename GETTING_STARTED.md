# Getting Started with The Convergence

**Build self-learning systems that improve with every optimization run.**

The Convergence uses reinforcement learning to find optimal configurations. Each run learns from previous runs, and your system gets smarter over time.

---

## Installation

```bash
pip install the-convergence
```

**Or install from source:**

```bash
git clone https://github.com/persist-os/the-convergence.git
cd the-convergence
pip install -e .
```

**With self-improving agents (RLP + SAO):**

```bash
pip install "the-convergence[agents]"
```

---

## Quick Start

### Option 1: Interactive Setup (Recommended)

```bash
# Interactive wizard guides you through setup
convergence init

# Run optimization - system learns from this run
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

## Your First Optimization

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
2. Configure optimization settings (generations, population, parallelism)
3. Enable the Learning System (warm-starts from previous runs)
4. Optionally enable Agent Society (RLP reasoning + SAO self-improvement)

**Output files:**

- `optimization.yaml` - Your configuration
- `test_cases.json` - Test cases
- `evaluator.py` - Evaluation logic (if needed)

### Step 3: Run Optimization

```bash
convergence optimize optimization.yaml
```

**Watch the system learn:**

```
STARTING OPTIMIZATION
======================================================================
API: openai_responses
Generations: 3 | Population: 4
Learning: Enabled (warm-start from previous runs)
======================================================================

GENERATION 1/3
Thompson Sampling: Exploring configuration space...
Testing 4 configurations in parallel...

Config [1/4]: model=gpt-4o-mini, temperature=0.7
   Test 1/3: capital_question → Score: 0.950 | Latency: 450ms
   Test 2/3: simple_math → Score: 0.900 | Latency: 380ms
   Test 3/3: creative_simple → Score: 0.850 | Latency: 520ms
   Aggregate Score: 0.900

GENERATION 2/3
Evolution: Breeding from top performers...
   Mutation: temperature 0.7 → 0.65
   Crossover: Combining gpt-4o-mini traits with gpt-4o

...

Optimization Complete!
   Best config: model=gpt-4o-mini, temperature=0.65
   Score: 0.92
   Improvement over Generation 1: +2.2%

Learning System: Results saved for next run
   Next optimization will start from these winners
```

### Step 4: Check Results

```bash
./results/
  ├── optimization_results.json   # Full history
  ├── optimization_report.md      # Analysis
  └── best_config.py              # Winning configuration
```

### Step 5: Run Again (Watch It Learn)

```bash
convergence optimize optimization.yaml
```

The second run starts from your previous winners and explores from there - that's the learning loop in action.

---

## How Learning Works

The Convergence improves across runs through three mechanisms:

### 1. Thompson Sampling (Per-Decision Learning)

Every configuration maintains a Beta distribution over expected rewards. The system samples from these distributions, naturally balancing:
- **Exploration** - Trying uncertain configurations
- **Exploitation** - Using known good configurations

### 2. Evolution (Per-Generation Learning)

Each generation breeds from the previous:
- **Elitism** - Top performers survive unchanged
- **Mutation** - Random parameter changes explore nearby space
- **Crossover** - Successful traits combine

### 3. Warm-Start (Cross-Run Learning)

Enabled by default. Previous optimization results seed the next run:
- Winners become starting population
- New runs explore from proven configurations
- Each run builds on all previous runs

---

## Example Configurations

We provide **examples** in the `examples/` directory:

### AI & LLM APIs

- OpenAI (ChatGPT) — `examples/ai/openai/`
- Groq — `examples/ai/groq/`
- Azure OpenAI — `examples/ai/azure/`

### Web Automation

- BrowserBase — `examples/web_browsing/browserbase/`

### Test Case Templates

- Test case guides — `examples/test_cases/`

### Best Practices

- Example patterns — `examples/BEST_PRACTICES.md`

---

## Understanding the Config

### Minimal Config Structure

```yaml
# 1. Your API
api:
  name: "my_api"
  endpoint: "https://api.example.com/v1/endpoint"
  auth:
    type: "bearer"
    token_env: "MY_API_KEY"

# 2. What to optimize (search space)
search_space:
  parameters:
    model: ["gpt-4o-mini", "gpt-4o"]
    temperature: [0.3, 0.5, 0.7, 0.9]

# 3. How to evaluate (reward signal)
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

# 5. Learning (enabled by default)
legacy:
  enabled: true
  storage_path: "./legacy"
```

---

## Customization

### Create Your Own Evaluator

If the built-in evaluators don't fit your needs:

```python
# evaluator.py
def score_response(result, expected, params, metric=None):
    """
    Custom evaluation logic - this becomes your reward signal.

    Args:
        result: API response
        expected: Expected values from test case
        params: Config parameters used
        metric: Specific metric being evaluated

    Returns:
        float: Reward between 0.0 and 1.0
    """
    reward = 0.0

    # Your domain logic
    if "answer" in result:
        if result["answer"] == expected.get("answer"):
            reward += 0.5

    if result.get("latency_ms", 9999) < 500:
        reward += 0.5

    return reward
```

**Reference it in your config:**

```yaml
evaluation:
  custom_evaluator:
    enabled: true
    module: "evaluator"
    function: "score_response"
```

---

## Pro Tips

### 1. Think in Episodes

Each optimization run is a learning episode. Run multiple times:
- First run: Broad exploration
- Second run: Refinement around winners
- Third run: Fine-tuning

### 2. Start Small, Scale Up

- Begin with 2-3 test cases
- Use `population_size: 4` and `generations: 3`
- Verify learning is happening, then scale

### 3. Monitor Learning Progress

```bash
# Check what the system learned
cat results/optimization_report.md

# View learning history
ls legacy/
```

### 4. Enable Agent Society for Complex Optimizations

```yaml
society:
  enabled: true
  learning:
    rlp_enabled: true   # Agents think before selecting configs
    sao_enabled: true   # Agents generate their own training data
```

### 5. Version Control Your Configs

```bash
git add optimization.yaml test_cases.json
git commit -m "Optimization config v1"
```

---

## Troubleshooting

### "Command not found: convergence"

```bash
pip install --upgrade the-convergence
which convergence
```

### "API key not found"

```bash
echo $OPENAI_API_KEY
export OPENAI_API_KEY="sk-..."
```

### "No module named 'convergence'"

```bash
pip install --upgrade the-convergence
```

### Optimization takes too long

```yaml
optimization:
  evolution:
    population_size: 2
    generations: 2
  execution:
    parallel_workers: 1
```

### Rate limit errors

```yaml
optimization:
  execution:
    parallel_workers: 1
    max_retries: 1
    delay_between_calls: 1
```

---

## Next Steps

1. **Run multiple episodes** - Watch learning compound
2. **Enable Agent Society** - Add RLP reasoning and SAO self-improvement
3. **Read examples** - Check `examples/BEST_PRACTICES.md`
4. **Production runtime** - See `SDK_USAGE.md` for per-request optimization

---

## Need Help?

- **Documentation**: `README.md`, `SDK_USAGE.md`, `TROUBLESHOOTING.md`
- **Issues**: [GitHub Issues](https://github.com/persist-os/the-convergence/issues)
- **Security**: See `SECURITY.md`

---

**Stop tuning. Start evolving.**
