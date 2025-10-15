# Best Practices for API Optimization

**Quick reference for getting optimal results with The Convergence.**

---

## 🎯 Quick Wins

### 1. Start with a Good Preset

```bash
convergence init  # Choose template matching your API
```

### 2. Use Representative Test Cases

**Good**: 5-10 real examples from production  
**Bad**: 100 random synthetic tests

### 3. Set Clear Metric Weights

```yaml
metrics:
  quality: {weight: 0.5}   # Most important
  latency_ms: {weight: 0.3}
  cost_usd: {weight: 0.2}  # Least important
```

---

## 📊 Test Case Design

### Cover Your Use Cases

```json
{
  "test_cases": [
    // 70% - Common cases (daily production traffic)
    {"id": "common_1", "input": {...}},
    {"id": "common_2", "input": {...}},
    
    // 20% - Edge cases (corner cases you've seen)
    {"id": "edge_1", "input": {...}},
    
    // 10% - Stress cases (worst-case scenarios)
    {"id": "stress_1", "input": {...}}
  ]
}
```

### Quality Over Quantity

✅ **5 high-quality tests** > ❌ **50 low-quality tests**

Each test should:
- Represent real usage
- Have clear expected output
- Be measurable

---

## 🔧 Search Space Configuration

### Start Broad, Then Narrow

**First run** - Explore:
```yaml
temperature: [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
```

**Second run** - Exploit:
```yaml
temperature: [0.6, 0.65, 0.7, 0.75, 0.8]  # Around best from first run
```

### Only Optimize What Varies

❌ **Don't do this:**
```yaml
api_version: ["v1"]  # Only one option!
```

✅ **Do this:**
```yaml
model: ["gpt-4o-mini", "gpt-4o", "claude-3-sonnet"]  # Multiple options
```

---

## ⚙️ Optimization Settings

### Quick Settings for Different Scenarios

#### Fast Demo (2-5 minutes)
```yaml
optimization:
  evolution:
    population_size: 2
    generations: 2
  execution:
    parallel_workers: 1
```
**Use for**: Initial testing, quick validation

#### Balanced (10-20 minutes)
```yaml
optimization:
  evolution:
    population_size: 4
    generations: 5
  execution:
    parallel_workers: 2
```
**Use for**: Most production use cases

#### Thorough (30-60 minutes)
```yaml
optimization:
  evolution:
    population_size: 8
    generations: 10
  execution:
    parallel_workers: 4
```
**Use for**: Critical production systems, complex search spaces

### Rule of Thumb

**Total experiments** = `population_size × generations`

- **Quick**: ~10-20 total experiments
- **Balanced**: ~20-50 total experiments
- **Thorough**: ~80-150 total experiments

### Always Enable Early Stopping

```yaml
optimization:
  execution:
    early_stopping:
      enabled: true
      patience: 3  # Stop if no improvement for 3 generations
      min_improvement: 0.01  # Must improve by 1%
```

**Saves time** when convergence is reached!

---

## 💰 Cost Management

### Set a Budget

```yaml
evaluation:
  metrics:
    cost_usd:
      weight: 0.2
      type: "lower_is_better"
      threshold: 0.01  # Reject configs over $0.01/call
```

### Start Cheap

1. **Optimize with cheap model** first (e.g., `gpt-4o-mini`)
2. **Test top 3 configs** on expensive model (e.g., `gpt-4o`)
3. **Deploy best** overall config

### Use Parallel Workers Wisely

```yaml
parallel_workers: 4  # 4× faster, but 4× the API calls per minute
```

**Match your rate limits:**
- Free tier API: `parallel_workers: 1`
- Paid tier: `parallel_workers: 2-4`
- Enterprise: `parallel_workers: 4-10`

---

## 📈 Metric Selection

### Weight by Business Value

**E-commerce chatbot:**
```yaml
metrics:
  customer_satisfaction: {weight: 0.5}  # Revenue impact
  latency_ms: {weight: 0.3}             # UX impact
  cost_per_chat: {weight: 0.2}          # Operating cost
```

**Batch processing:**
```yaml
metrics:
  accuracy: {weight: 0.5}     # Quality matters
  cost_per_item: {weight: 0.4}  # Volume × cost = big impact
  throughput: {weight: 0.1}     # Speed less critical
```

### Use Thresholds for Hard Constraints

```yaml
metrics:
  success_rate:
    weight: 0.3
    threshold: 0.95  # MUST be > 95%, or reject config
  
  latency_ms:
    weight: 0.4
    threshold: 1000  # MUST be < 1 second, or reject config
```

---

## 🚀 Execution Strategy

### Iterative Approach

**Don't optimize everything at once!**

**Week 1**: Optimize major parameters (model choice)  
**Week 2**: Optimize generation params (temperature, tokens)  
**Week 3**: Optimize advanced features (caching, retries)

### Before Deploying to Production

1. **Baseline**: Run current config, record metrics
2. **Optimize**: Run convergence, get best config
3. **Validate**: Test best config in staging
4. **A/B test**: 10% → 50% → 100% rollout
5. **Monitor**: Watch metrics for regressions

---

## 🔍 Troubleshooting

### Optimization Not Improving?

**Check**:
- ✅ Test cases are realistic
- ✅ Metrics are weighted correctly
- ✅ Search space isn't too narrow
- ✅ Evaluation function works

**Try**:
- Increase `population_size`
- Increase `generations`
- Widen search space
- Add more diverse test cases

### Too Slow?

**Reduce**:
```yaml
optimization:
  evolution:
    population_size: 2    # Fewer configs
    generations: 2        # Fewer iterations
  execution:
    parallel_workers: 1   # Sequential (if hitting rate limits)
```

### Hitting Rate Limits?

```yaml
optimization:
  execution:
    parallel_workers: 1        # One at a time
    max_retries: 1             # Fewer retries
```

### Results Not Reproducible?

Ensure:
- Same test cases
- Same evaluation logic
- Same API version
- Deterministic evaluation (no randomness)

---

## 📝 Documentation & Version Control

### Save Your Configs

```bash
git add optimization.yaml test_cases.json evaluator.py
git commit -m "feat: API optimization config for chatbot v2"
```

### Document Why

```yaml
search_space:
  parameters:
    model: ["gpt-4o-mini", "gpt-4o"]
    # Note: Excluded gpt-3.5-turbo - failed quality threshold
    # in baseline testing (accuracy < 0.80)
```

### Track Results

```bash
# Save results with descriptive names
convergence optimize config.yaml \
  --output results/chatbot_v2_optimization_2024_10_15
```

---

## 🎓 Advanced Patterns

### Conditional Configs

```python
# Use different optimized configs for different scenarios
if user_type == "premium":
    config = best_config_premium  # Higher quality
elif query_complexity == "simple":
    config = best_config_fast     # Lower cost
else:
    config = best_config_balanced
```

---

## ✅ Pre-Launch Checklist

Before deploying optimized config:

- [ ] Tested in staging environment
- [ ] Validated on real production traffic (A/B test)
- [ ] Monitored for 24-48 hours
- [ ] No regressions in key metrics
- [ ] Cost impact is acceptable
- [ ] Team trained on new config
- [ ] Rollback plan ready

---

## 🔄 Continuous Improvement

**Re-optimize regularly:**

- ✅ Quarterly (APIs and use cases evolve)
- ✅ After major API updates (new models, pricing)
- ✅ After traffic pattern changes
- ✅ When adding new features

**Each run builds on previous knowledge** via the legacy system!

---

## 💡 Key Takeaways

1. **Start simple** - Use presets, few test cases, quick settings
2. **Iterate** - Don't try to optimize everything at once
3. **Monitor** - Track results, A/B test, watch for regressions
4. **Improve** - Re-run optimization as your needs evolve
5. **Document** - Save configs, results, and decisions

---

**Remember**: Perfect is the enemy of good. Start with a baseline, improve iteratively! 🚀
