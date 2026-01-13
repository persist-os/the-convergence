# The Convergence

**TIER:** 2 (Production-grade)
**STACK:** Python 3.11+, asyncio, Pydantic, LiteLLM
**DOMAIN:** Framework/Library (Self-Evolving Agent Systems)

---

## Project Identity

The Convergence is a **self-evolving agent framework** built on reinforcement learning principles. The core insight: systems that improve themselves through experience outperform static configurations.

### Core Narrative (Use This Language)

- **NOT** "API optimization" - that's a use case, not identity
- **YES** "Self-evolving agents" / "Agentic RL" / "Systems that learn"
- Lead with the RL story: Thompson Sampling, policy learning, dense rewards
- RLP and SAO are **core differentiators**, not optional extras

### Key Differentiators

1. **RL-First Design** - Every decision is a learning opportunity
2. **Evolution as Principle** - Genetic algorithms for configuration space
3. **Self-Improvement** - RLP (think before acting) + SAO (self-generated training)
4. **Production-Ready** - Not just research, actually works

---

## Coding Rules

### Architecture Invariants

```
OPTIMIZATION LOOP IS SACRED
MAB → Evolution → RL Meta → Storage → Repeat
Never break this cycle. All features integrate into it.

PLUGINS EXTEND, DON'T REPLACE
Core loop is fixed. Plugins add strategies, storage backends, adapters.
New features = new plugins, not core modifications.

ASYNC THROUGHOUT
All I/O is async. No blocking calls in hot paths.
Use asyncio patterns consistently.
```

### Testing Requirements

```
- [ ] Unit tests for all new components
- [ ] Integration tests for optimization loop changes
- [ ] Property-based tests for evolutionary operators (Hypothesis)
- [ ] Benchmark tests for performance-critical paths
- [ ] Type checking passes (mypy --strict)
```

### Validation Protocol

```
Before any PR:
- [ ] All tests pass
- [ ] Type checking clean
- [ ] Ruff linting clean
- [ ] Documentation updated for public API changes
- [ ] CHANGELOG.md updated
```

---

## Messaging Guidelines

When writing docs, README, or user-facing content:

### Lead With

- "Self-evolving" / "Self-improving"
- "Reinforcement learning" / "RL-first"
- "Dense reward signals" / "Policy learning"
- "Thompson Sampling" / "Bayesian optimization"
- "Evolution" / "Genetic algorithms"
- "Agent society" / "Multi-agent coordination"

### Avoid Leading With

- "API optimization" (makes it sound like a utility)
- "Parameter tuning" (undersells the intelligence)
- "Configuration" (too static)
- "Optional features" for RLP/SAO (they're core)

### Competitive Positioning

- vs manual tuning: "Systems that learn vs. systems you tune"
- vs grid search: "Intelligent exploration vs. brute force"
- vs hyperopt: "Continuous learning vs. one-shot optimization"
