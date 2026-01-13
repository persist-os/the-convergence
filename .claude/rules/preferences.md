---
paths: "**/*"
---

# Project Preferences

## Language & Narrative

When writing user-facing content (README, docs, CLI output):
- Lead with "self-evolving", "reinforcement learning", "agent"
- Avoid "API optimization", "parameter tuning" as primary framing
- Emphasize the learning loop, not the configuration

## Code Style

- Async by default for all I/O
- Pydantic for all data models
- Type hints everywhere (mypy --strict compatible)
- Weave integration for observability

## RL Terminology

Use correct RL terms:
- "arm" not "option" (for MAB)
- "reward" not "score" (for feedback signals)
- "policy" not "strategy" (for decision rules)
- "episode" not "run" (for learning iterations)
