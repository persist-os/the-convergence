# Learning Systems - Experimental Agent Society Features

## ⚠️  Status: EXPERIMENTAL - Active in v0.1.0 with Agent Society

These learning systems **ARE integrated** into the optimization pipeline when Agent Society is enabled. They are experimental features under active development.

---

## 📚 Overview

The learning systems provide advanced AI learning capabilities for agent societies, based on cutting-edge research:

| System | Research Paper | Purpose | Status |
|--------|---------------|---------|--------|
| **RLP** | [NVIDIA 2024](https://arxiv.org/abs/2510.01265) | Think before acting (internal reasoning) | ✅ Active when society enabled |
| **SAO** | [Hugging Face 2024](https://arxiv.org/abs/2510.06652) | Self-generate training data | ✅ Active when society enabled |

---

## 🧠 RLP: Reinforcement Learning on Policy

**Paper**: [Reinforcement as a Pretraining Objective](https://arxiv.org/abs/2510.01265) (NVIDIA, October 2024)

**File**: `rlp.py`

### Core Idea

Reward agents for generating thoughts that improve prediction accuracy:
- **Before** making a decision: Generate internal reasoning
- **Reward** = Information gain (does the thought help predict outcomes?)
- **Learn** from every decision (dense reward signal)
- **No external verifier needed** - self-supervised!

### How It Works in Convergence

```python
# During optimization:
1. RLP generates internal reasoning before selecting configs
2. System tests configs and measures results
3. RLP receives reward based on how much reasoning helped
4. Policy improves over time → smarter config selection
```

### Example

```
Generation 1:
🧠 RLP: Internal reasoning active...
   "Based on previous runs, focusing on temperature range 0.6-0.8
    appears most promising for quality metrics..."
   
📊 Config tested: temperature=0.7 → Score: 0.85
🧠 RLP: Training policy from results...
   📊 RLP Stats: Mean reward=0.12, Episodes=5

Generation 2:
🧠 RLP: Internal reasoning active...
   "Temperature 0.7 worked well. Let's explore nearby values
    while also testing different model combinations..."
```

### Features Implemented

- ✅ Internal reasoning generation via LLM
- ✅ Information gain reward calculation
- ✅ Experience replay buffer (10,000 experiences)
- ✅ Reward normalization for stable learning
- ✅ GAE (Generalized Advantage Estimation)
- ✅ Log-probability extraction (when available)
- ✅ Weave integration for tracking

### Configuration

```yaml
society:
  enabled: true
  learning:
    rlp_enabled: true  # Enable RLP reasoning
```

### Research Highlights

**From the paper:**
- "Reward agents for generating thoughts that improve next-token prediction"
- "Dense reward signal at every position"
- "No need for external verifier or reward model"
- "Self-supervised learning from reasoning"

### Metrics

RLP tracks:
- Mean reward per episode
- Reward trend (improving/stable/declining)
- Buffer size (experiences stored)
- Policy update statistics

---

## 🔄 SAO: Self-Alignment Optimization

**Paper**: [Aligning LLMs via Fully Self-Synthetic Data](https://arxiv.org/abs/2510.06652) (Hugging Face, October 2024)

**File**: `sao.py`

### Core Idea

LLMs generate their own training data - **no external labels needed**:
1. **Generate prompts** via persona role-play (PersonaHub)
2. **Generate response pairs** for each prompt
3. **Self-judge** to create preference labels
4. **Train** on self-generated data

**No GPT-4, no human labeling, fully self-synthetic!**

### How It Works in Convergence

```python
# During optimization (every 3 generations):
1. SAO analyzes optimization history
2. Generates synthetic preference pairs:
   - Prompt: "How should I optimize this API?"
   - Response A: "Try lowering temperature"
   - Response B: "Test multiple models first"
   - Self-judgment: "B is better - more systematic"
3. Stores in dataset for future policy improvements
```

### Example

```
Generation 3:
🔄 SAO: Self-improvement active...
   📊 Generating synthetic preference pairs...
   ✅ Generated 3 preference pairs
   📈 SAO Dataset: 12 total samples
   
Sample generated:
  Prompt: "For API optimization with quality focus, what strategy works best?"
  Chosen: "Systematically test multiple models, then fine-tune the winner"
  Rejected: "Just increase temperature until quality improves"
```

### Features Implemented

- ✅ Persona-based prompt generation (100+ persona templates)
- ✅ Response pair generation with temperature variation
- ✅ Self-judgment for preference labeling
- ✅ Quality filtering (length, diversity, duplicates)
- ✅ Diversity metrics and duplicate detection
- ✅ Iterative refinement (multi-round SAO)
- ✅ Dataset export/import (JSONL, JSON)
- ✅ Batch processing for efficiency
- ✅ Weave integration for tracking

### Configuration

```yaml
society:
  enabled: true
  learning:
    sao_enabled: true  # Enable SAO self-improvement
```

### Persona Templates

SAO uses diverse personas to generate varied prompts:
- Software engineers, teachers, researchers
- Different age groups, locations, interests
- Various expertise levels and backgrounds

**Example persona:**
```
"A 35 year old software engineer from San Francisco who is passionate
 about technology, seeking advice on API optimization strategies"
```

### Research Highlights

**From the paper:**
- "Model's self-judgment is MORE effective than GPT-4 within SAO framework"
- "Persona role-play acts as compress-and-decompress for world knowledge"
- "Fully self-synthetic - no external reward model needed"
- "Outperforms traditional RLHF on alignment benchmarks"

### Metrics

SAO tracks:
- Dataset size (total preference pairs)
- Unique prompts generated
- Diversity score (1.0 = very diverse)
- Quality filtering statistics
- Generation rounds completed

---

## 🔧 How They Work Together

### Integration in Optimization Loop

```python
for generation in range(total_generations):
    # 1. RLP: Think before selecting configs
    if rlp_enabled:
        reasoning = await rlp.generate_internal_reasoning(state)
        # Use reasoning to guide config selection
    
    # 2. Test configs (existing optimization)
    results = await test_configs(population)
    
    # 3. RLP: Learn from results
    if rlp_enabled:
        reward = calculate_information_gain(reasoning, results)
        rlp.update_policy(reasoning, reward)
    
    # 4. SAO: Generate training data (every 3 generations)
    if sao_enabled and generation % 3 == 0:
        synthetic_data = await sao.generate_synthetic_dataset(n=3)
        # Store for future policy improvements
    
    # 5. Evolve population for next generation
    population = evolve(population, results)
```

### Console Output

When agent society is enabled, you'll see:

```
🚀 STARTING API OPTIMIZATION
...
🤖 AGENT SOCIETY: 2 agents active
   • RLP (Reasoning-based Learning)
   • SAO (Self-Alignment Optimization)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧬 GENERATION 1/3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧠 RLP: Internal reasoning active...
...

🧠 RLP: Training policy from generation results...
   📊 RLP Stats: Mean reward=0.12, Episodes=5

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧬 GENERATION 3/3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔄 SAO: Self-improvement active...
   📊 Generating synthetic preference pairs...
   ✅ Generated 3 preference pairs
   📈 SAO Dataset: 12 total samples

🤖 AGENT SOCIETY CONTRIBUTIONS:
   • RLP (Reasoning): Active
   • SAO (Self-Improvement): Active
```

---

## 📊 Current Status

### v0.1.0 (Beta) - Experimental Integration

**What Works:**
- ✅ RLP generates internal reasoning before config selection
- ✅ RLP trains policy based on optimization results
- ✅ SAO generates synthetic preference pairs from history
- ✅ Both track metrics and integrate with Weave
- ✅ Console output shows their contributions

**Experimental Limitations:**
- ⚠️  Requires LiteLLM-compatible LLM (set via env vars)
- ⚠️  RLP reasoning is not yet used for direct config selection (guidance only)
- ⚠️  SAO data is generated but not yet used for retraining
- ⚠️  Performance impact: +10-20% runtime for reasoning/generation
- ⚠️  No persistence between runs (in-memory only)

**Why Include Now?**
1. **Active Research**: Based on papers from Oct 2024 (bleeding edge!)
2. **Infrastructure Ready**: Full implementations with tracking
3. **Experimental Value**: Users can see agent society in action
4. **Foundation**: Enables rapid improvements in v0.2.0+

---

## 🚀 Roadmap

### v0.2.0 (Planned)

- [ ] Use RLP reasoning to directly influence config selection
- [ ] Train policies on SAO-generated preference data
- [ ] Persist RLP experience buffer between runs
- [ ] Export/import SAO datasets for sharing
- [ ] Multi-agent collaboration (agents share learnings)
- [ ] Adaptive learning rates based on performance

### v0.3.0+ (Future)

- [ ] Full DPO (Direct Preference Optimization) training loop
- [ ] Cross-session policy transfer
- [ ] Agent specialization (different agents for different APIs)
- [ ] Federated learning across optimization runs
- [ ] Advanced persona generation (domain-specific)

---

## 🔬 Research Background

### RLP (NVIDIA Research, Oct 2024)

**Key Innovation**: Reward models for generating useful thoughts, not just final answers.

**Insight**: Traditional RL rewards final outcomes. RLP rewards intermediate reasoning steps. This creates a dense learning signal and enables "thinking before acting."

**Results**: Models trained with RLP show improved reasoning, better generalization, and don't require external verifiers.

### SAO (Hugging Face Research, Oct 2024)

**Key Innovation**: Models generate their own training data - prompts, responses, AND preference labels.

**Insight**: Instead of human labeling or GPT-4 judging, the model judges its own responses. Surprisingly, this works BETTER than external judges when done within the SAO framework.

**Results**: Models trained with SAO achieve strong alignment without any external supervision. Fully self-synthetic!

---

## 💡 Usage Examples

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

## ⚙️ Configuration Options

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

## 🤝 Contributing

Want to enhance the learning systems? Check `CONTRIBUTING.md` and:

1. Review the research papers
2. Test experimental features
3. Propose improvements
4. Submit PR with Weave tracking
5. Update documentation

---

## 📚 Further Reading

- **RLP Paper**: https://arxiv.org/abs/2510.01265
- **SAO Paper**: https://arxiv.org/abs/2510.06652

---

**Status**: ✅ Experimental - Active when agent society enabled  
**Research**: October 2024 (NVIDIA + Hugging Face)  
**Last Updated**: October 15, 2025