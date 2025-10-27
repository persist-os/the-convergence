"""
Agno Gmail Agent Optimization with Azure OpenAI.

Test and optimize Agno agents using Gmail toolkit across
different Azure model deployments with MAB-based model selection.

Key Components:
- gmail_agent_optimization.yaml: Main configuration with environment-based token
- gmail_test_cases.json: Comprehensive test cases for email management
- gmail_evaluator.py: Custom evaluation (accuracy, completeness, latency, token efficiency)
- gmail_agent_runner.py: Agno agent wrapper with Azure support
"""

from .gmail_agent_runner import GmailAgentRunner, test_gmail_connection

__all__ = ['GmailAgentRunner', 'test_gmail_connection']

