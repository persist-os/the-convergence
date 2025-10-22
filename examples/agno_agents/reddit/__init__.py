"""
Agno Reddit Agent Optimization with Azure OpenAI.

Test and optimize Agno agents using Reddit's social toolkit across
different Azure model deployments with MAB-based model selection.

Key Components:
- reddit_agent_optimization.yaml: Main configuration with Azure integration
- reddit_test_cases.json: 4 comprehensive test cases
- reddit_evaluator.py: Custom evaluation (accuracy, completeness, latency, token efficiency)
- reddit_agent_runner.py: Agno agent wrapper with Azure support
"""

from .reddit_agent_runner import RedditAgentRunner, test_reddit_connection

__all__ = ['RedditAgentRunner', 'test_reddit_connection']

