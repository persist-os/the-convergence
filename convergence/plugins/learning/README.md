# Learning Systems - Core Self-Improvement Components

## Status: Core Components

These learning systems are **integrated** into the optimization pipeline and represent the core self-improvement capabilities of The Convergence.

---

## Overview

The learning systems enable autonomous optimization through cutting-edge research:

| System | Research Paper | Purpose |
|--------|---------------|---------|
| **RLP** | [NVIDIA 2024](https://arxiv.org/abs/2510.01265) | Think before acting (internal reasoning) |
| **SAO** | [Hugging Face 2024](https://arxiv.org/abs/2510.06652) | Self-generate training data |

Together, these create agents that reason about decisions and continuously improve without external supervision.

---

## RLP: Reinforcement Learning on Policy

**Paper**: [Reinforcement as a Pretraining Objective](https://arxiv.org/abs/2510.01265) (NVIDIA, October 2024)

### Core Idea

Reward agents for generating thoughts that improve prediction accuracy:
- **Before** making a decision: Generate internal reasoning
- **Reward** = Information gain (does the thought help predict outcomes?)
- **Learn** from every decision (dense reward signal)
- **No external verifier needed** - self-supervised!

### How It Works in The Convergence

```
Generation Flow:
1. RLP generates internal reasoning about the search space
2. Reasoning biases config selection toward promising regions
3. Configs are tested and scored
4. RLP receives reward based on how much reasoning helped
5. Policy improves over generations → smarter config selection
```

### Example Output

```
Generation 1:
RLP: Internal reasoning active...
   "Based on previous runs, focusing on temperature range 0.6-0.8
    appears most promising for quality metrics..."

Config tested: temperature=0.7 → Score: 0.85
RLP: Training policy from results...
   Stats: Mean reward=0.12, Episodes=5

Generation 2:
RLP: Internal reasoning active...
   "Temperature 0.7 worked well. Let's explore nearby values
    while also testing different model combinations..."
```

### Features

- Internal reasoning generation via LLM
- Information gain reward calculation
- Experience replay buffer (10,000 experiences)
- Reward normalization for stable learning
- GAE (Generalized Advantage Estimation)
- Weave integration for tracking

### Configuration

```yaml
society:
  enabled: true
  learning:
    rlp_enabled: true
```

---

## SAO: Self-Alignment Optimization

**Paper**: [Aligning LLMs via Fully Self-Synthetic Data](https://arxiv.org/abs/2510.06652) (Hugging Face, October 2024)

### Core Idea

LLMs generate their own training data - **no external labels needed**:
1. **Generate prompts** via persona role-play (PersonaHub)
2. **Generate response pairs** for each prompt
3. **Self-judge** to create preference labels
4. **Train** on self-generated data

**No GPT-4, no human labeling, fully self-synthetic!**

### How It Works in The Convergence

```
Every few generations:
1. SAO analyzes optimization history
2. Generates synthetic preference pairs:
   - Prompt: "How should I optimize this API?"
   - Response A: "Try lowering temperature"
   - Response B: "Test multiple models first"
   - Self-judgment: "B is better - more systematic"
3. Stores in dataset for policy improvements
```

### Example Output

```
Generation 3:
SAO: Self-improvement active...
   Generating synthetic preference pairs...
   Generated 3 preference pairs
   Dataset: 12 total samples

Sample generated:
  Prompt: "For API optimization with quality focus, what strategy works best?"
  Chosen: "Systematically test multiple models, then fine-tune the winner"
  Rejected: "Just increase temperature until quality improves"
```

### Features

- Persona-based prompt generation (100+ personas)
- Response pair generation with temperature variation
- Self-judgment for preference labeling
- Quality filtering (length, diversity, duplicates)
- Iterative refinement (multi-round SAO)
- Dataset export/import (JSONL, JSON)
- Weave integration for tracking

### Configuration

```yaml
society:
  enabled: true
  learning:
    sao_enabled: true
```

---

## How They Work Together

### Integration in the Optimization Loop

```python
for generation in range(total_generations):
    # 1. RLP: Think before selecting configs
    if rlp_enabled:
        reasoning = await rlp.generate_internal_reasoning(state)
        # Reasoning guides config selection

    # 2. Test configs (existing optimization)
    results = await test_configs(population)

    # 3. RLP: Learn from results
    if rlp_enabled:
        reward = calculate_information_gain(reasoning, results)
        rlp.update_policy(reasoning, reward)

    # 4. SAO: Generate training data (periodically)
    if sao_enabled and generation % 3 == 0:
        synthetic_data = await sao.generate_synthetic_dataset(n=3)

    # 5. Evolve population for next generation
    population = evolve(population, results)
```

### Console Output

When agent society is enabled:

```
STARTING OPTIMIZATION
...
AGENT SOCIETY: 2 agents active
   - RLP (Reasoning-based Learning)
   - SAO (Self-Alignment Optimization)

GENERATION 1/3
RLP: Internal reasoning active...
...

RLP: Training policy from generation results...
   Stats: Mean reward=0.12, Episodes=5

GENERATION 3/3
SAO: Self-improvement active...
   Generating synthetic preference pairs...
   Generated 3 preference pairs
   Dataset: 12 total samples

AGENT SOCIETY CONTRIBUTIONS:
   - RLP (Reasoning): Active
   - SAO (Self-Improvement): Active
```

---

## Research Background

### RLP (NVIDIA Research, Oct 2024)

**Key Innovation**: Reward models for generating useful thoughts, not just final answers.

**Insight**: Traditional RL rewards final outcomes. RLP rewards intermediate reasoning steps. This creates a dense learning signal and enables "thinking before acting."

**Results**: Models trained with RLP show improved reasoning, better generalization, and don't require external verifiers.

### SAO (Hugging Face Research, Oct 2024)

**Key Innovation**: Models generate their own training data - prompts, responses, AND preference labels.

**Insight**: Instead of human labeling or GPT-4 judging, the model judges its own responses. Surprisingly, this works BETTER than external judges when done within the SAO framework.

**Results**: Models trained with SAO achieve strong alignment without any external supervision.

---

## Usage Examples

### Enable Agent Society

```yaml
# optimization.yaml
society:
  enabled: true
  auto_generate_agents: true
  learning:
    rlp_enabled: true
    sao_enabled: true
  llm:
    model: "openai/gpt-4o-mini"
    api_key_env: "OPENAI_API_KEY"
```

### Run Optimization

```bash
export OPENAI_API_KEY="your-key-here"
convergence optimize optimization.yaml
```

### Programmatic Access

```python
from convergence.plugins.learning.rlp import RLPMixin, RLPConfig
from convergence.plugins.learning.sao import SAOMixin, SAOConfig

# Create RLP agent
rlp = RLPMixin(
    config=RLPConfig(temperature=0.7),
    llm_provider=your_llm_provider
)

# Generate reasoning
thought = await rlp.generate_internal_reasoning(
    state={"goal": "optimize_api"},
    context="Testing different temperature values"
)

# Calculate reward
reward = rlp.information_gain_reward(
    thought=thought,
    prediction="temperature=0.7",
    outcome="score=0.85"
)

# Update policy
updated_state = rlp.update_rlp_policy(
    thought=thought,
    reward=reward,
    state=state
)

# Create SAO agent
sao = SAOMixin(
    config=SAOConfig(n_personas=100),
    llm_provider=your_llm_provider
)

# Generate synthetic data
dataset = await sao.generate_synthetic_dataset(n_samples=10)

# Export for later use
sao.export_dataset("training_data.jsonl", format="jsonl")
```

---

## Configuration Options

### RLP Configuration

```python
class RLPConfig:
    model_name: str = "gpt2"              # Model for reasoning
    thought_length: int = 500             # Max tokens for thoughts
    temperature: float = 0.7              # Sampling temperature
    buffer_size: int = 10000              # Experience replay size
    gamma: float = 0.99                   # Discount factor
    gae_lambda: float = 0.95              # GAE lambda
    normalize_rewards: bool = True        # Reward normalization
    use_logprobs: bool = True             # Extract log-probs
```

### SAO Configuration

```python
class SAOConfig:
    n_personas: int = 100                 # Number of personas
    temperature: float = 0.6              # Generation temperature
    response_pairs: int = 2               # Responses per prompt
    min_response_length: int = 50         # Quality filter
    similarity_threshold: float = 0.9     # Duplicate detection
    quality_filter: bool = True           # Enable filtering
    diversity_factor: float = 0.7         # Diversity weight
    iterative_rounds: int = 3             # Refinement rounds
    batch_size: int = 10                  # Parallel generation
```

---

## Contributing

Want to enhance the learning systems? Check `CONTRIBUTING.md` and:

1. Review the research papers
2. Test features
3. Propose improvements
4. Submit PR with Weave tracking
5. Update documentation

---

## Further Reading

- **RLP Paper**: https://arxiv.org/abs/2510.01265
- **SAO Paper**: https://arxiv.org/abs/2510.06652

---

**Status**: Core self-improvement components
**Research**: October 2024 (NVIDIA + Hugging Face)
**Last Updated**: 2026-01-12
