# Azure OpenAI o4-mini Optimization

## Overview
This configuration optimizes Azure OpenAI's o4-mini reasoning model parameters for complex reasoning tasks.

## Recent Improvements (2025-10-15)

### 1. **Expanded Test Cases** (1 → 3 tests)
- ✅ **Basic Math** (easy): Simple arithmetic word problem
- ✅ **Logic Puzzle** (medium): Deductive reasoning with constraints
- ✅ **Multi-Step Reasoning** (hard): Complex calculations with percentages

### 2. **Enhanced Search Space** (1 → 3 parameters)

#### Before:
```yaml
max_completion_tokens: [1000, 5000, 10000]  # Only 1 parameter
```

#### After:
```yaml
max_completion_tokens: [500, 1000, 2000, 4000, 8000]  # 5 values
presence_penalty: [-2.0 to 2.0, step 0.5]              # 9 values  
frequency_penalty: [-2.0 to 2.0, step 0.5]             # 9 values
```

**Total search space**: 5 × 9 × 9 = **405 possible configurations**

### 3. **Improved Evaluator**
Enhanced `azure_o4_mini_evaluator.py` to handle:
- ✅ Logic puzzle validation (checks pet assignments)
- ✅ Numerical answer extraction with tolerance
- ✅ Percentage calculation verification
- ✅ Regex-based answer extraction

### 4. **Optimized Algorithm Settings**
```yaml
population_size: 6      # Up from 3
generations: 5          # Up from 3
parallel_workers: 3     # Parallel execution
experiments_per_gen: 18 # 6 configs × 3 tests
```

## How to Run

```bash
convergence optimize examples/ai/azure/azure_o4_mini/azure_o4_mini_optimization.yaml
```

## Expected Results
- **Better reasoning diversity** via presence/frequency penalties
- **More comprehensive evaluation** with 3 test cases
- **Optimal token allocation** across different problem complexities
- **5 generations × 18 experiments** = 90 total API calls

## Metrics Tracked
1. **Reasoning Accuracy** (40%) - Correctness of final answer
2. **Solution Completeness** (30%) - Quality of step-by-step explanation
3. **Latency** (20%) - Response time optimization
4. **Cost** (10%) - Token usage efficiency

## Test Case Details

### Test 1: Basic Math
Simple arithmetic to establish baseline performance.

### Test 2: Logic Puzzle
Requires deductive reasoning:
- Alex is allergic to fur → must have bird
- Bailey's pet can fly → has bird (CONFLICT!)
- Casey's pet barks → has dog

*Note: This has an intentional conflict to test error handling*

### Test 3: Multi-Step Reasoning
Complex calculation requiring:
1. Calculate department sizes (40%, 30%, 30%)
2. Calculate current training budgets
3. Apply percentage increases
4. Sum total increase
5. Calculate percentage of original budget

**Expected**: $45,600 additional, 10.6% increase

## Next Steps
1. Run optimization to collect results
2. Analyze which parameter combinations work best for each difficulty level
3. Consider adding temperature if o4-mini supports it
4. Add more test cases for edge cases
