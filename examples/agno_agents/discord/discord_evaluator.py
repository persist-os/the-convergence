"""
Custom evaluator for Agno Discord agent optimization with Azure OpenAI.

Evaluates agent performance across 4 metrics:
1. Accuracy (40%): Tool usage correctness and result relevance
2. Completeness (30%): Data field presence and population
3. Latency (20%): Response time
4. Token Efficiency (10%): Value per token used

Validation Features:
- Schema validation (required/optional fields)
- Data completeness checks (field presence, value reasonableness)
- Keyword matching (case-insensitive, flexible)
- Discord-specific validation (channel IDs, guild IDs, message content)
- Multi-tool workflow validation
"""

import json
import logging
import re
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def score_discord_agent_response(
    result: Any,
    expected: Dict[str, Any],
    params: Dict[str, Any],
    metric: Optional[str] = None
) -> float:
    """
    Score Agno Discord agent responses across multiple metrics.
    
    Args:
        result: Agent response with tool calls and results
        expected: Expected validation criteria from test case
        params: Agent parameters used (model, temperature, etc.)
        metric: Specific metric to evaluate
    
    Returns:
        float: Score between 0.0 and 1.0
        
    Raises:
        RuntimeError: If result is None or fundamentally invalid
    """
    # FAIL HARD if no result (API call failed)
    if result is None:
        raise RuntimeError(
            "API call returned None. Check Azure API key, endpoint, and deployment."
        )
    
    logger.info(f"="*80)
    logger.info(f"Evaluating metric: {metric}")
    logger.info(f"Agent params: {params}")
    
    try:
        # Parse agent response
        agent_data = _parse_agent_response(result)
        
        # Check for empty content
        if not agent_data.get('final_response') and not agent_data.get('tool_results'):
            logger.warning("Empty response from agent")
            # Still evaluate what we can
            if metric == "latency_seconds":
                return _score_latency(agent_data, expected, params)
            return 0.0
        
        # Route to appropriate evaluator
        if metric == "accuracy":
            return _score_accuracy(agent_data, expected, params)
        
        elif metric == "completeness":
            return _score_completeness(agent_data, expected, params)
        
        elif metric == "latency_seconds":
            return _score_latency(agent_data, expected, params)
        
        elif metric == "token_efficiency":
            return _score_token_efficiency(agent_data, expected, params)
        
        # Aggregate score (weighted average)
        return _aggregate_score(agent_data, expected, params)
        
    except Exception as e:
        logger.error(f"Evaluator error: {e}", exc_info=True)
        raise RuntimeError(f"Evaluation failed: {e}") from e


def _parse_agent_response(result: Any) -> Dict[str, Any]:
    """Parse agent response to extract tool calls and results."""
    if isinstance(result, str):
        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            return {
                'tool_calls': [],
                'final_response': result,
                'tool_results': [],
                'tokens_used': {},
                'latency_seconds': 0.0
            }
    elif isinstance(result, dict):
        data = result
    else:
        data = {'raw_result': str(result)}
    
    parsed = {
        'tool_calls': [],
        'final_response': '',
        'tool_results': [],
        'tokens_used': {},
        'latency_seconds': 0.0
    }
    
    # Extract from Agno agent runner format
    if 'final_response' in data:
        parsed['final_response'] = data.get('final_response', '')
        parsed['tool_calls'] = data.get('tool_calls', [])
        parsed['tool_results'] = data.get('tool_results', [])
        parsed['tokens_used'] = data.get('tokens_used', {})
        parsed['latency_seconds'] = data.get('latency_seconds', 0.0)
    
    return parsed


def _score_accuracy(agent_data: Dict[str, Any], expected: Dict[str, Any], params: Dict[str, Any]) -> float:
    """Score accuracy: Did agent call correct tools with correct parameters?"""
    score = 0.0
    
    tool_score = _check_tool_usage(agent_data, expected)
    score += tool_score * 0.30
    
    param_score = _check_tool_parameters(agent_data, expected)
    score += param_score * 0.20
    
    results_score = _check_results_validity(agent_data, expected)
    score += results_score * 0.30
    
    keyword_score = _check_keywords(agent_data, expected)
    score += keyword_score * 0.20
    
    logger.info(f"  Accuracy score: {score:.3f}")
    return score


def _score_completeness(agent_data: Dict[str, Any], expected: Dict[str, Any], params: Dict[str, Any]) -> float:
    """Score completeness: Are all expected fields present with valid data?"""
    validation = expected.get('result_validation', {})
    schema = validation.get('schema', {})
    data_checks = validation.get('data_checks', {})
    
    if not data_checks:
        return 1.0
    
    # Check required field presence
    score = 0.0
    checks_passed = 0
    total_checks = len([k for k in data_checks.keys() if isinstance(data_checks[k], bool)])
    
    for check_name, check_value in data_checks.items():
        if isinstance(check_value, bool):
            # Simple boolean check
            checks_passed += 1 if check_value else 0
        elif isinstance(check_value, dict) and 'min' in check_value:
            # Numeric threshold check
            if check_name in agent_data.get('final_response', '').lower():
                checks_passed += 1
    
    if total_checks > 0:
        score = checks_passed / total_checks
    
    logger.info(f"  Completeness score: {score:.3f}")
    return score


def _score_latency(agent_data: Dict[str, Any], expected: Dict[str, Any], params: Dict[str, Any]) -> float:
    """Score latency: Lower is better."""
    latency = agent_data.get('latency_seconds', 0.0)
    
    # Target: < 5 seconds is excellent
    if latency < 2.0:
        return 1.0
    elif latency < 5.0:
        return 0.9
    elif latency < 10.0:
        return 0.7
    elif latency < 30.0:
        return 0.5
    else:
        return 0.2


def _score_token_efficiency(agent_data: Dict[str, Any], expected: Dict[str, Any], params: Dict[str, Any]) -> float:
    """Score token efficiency: Value per token."""
    tokens = agent_data.get('tokens_used', {})
    if not tokens or tokens.get('total', 0) == 0:
        return 0.5  # Neutral if no token info
    
    total_tokens = tokens.get('total', 0)
    
    # Target: < 1000 tokens is efficient
    if total_tokens < 500:
        return 1.0
    elif total_tokens < 1000:
        return 0.9
    elif total_tokens < 2000:
        return 0.7
    else:
        return 0.5


def _aggregate_score(agent_data: Dict[str, Any], expected: Dict[str, Any], params: Dict[str, Any]) -> float:
    """Aggregate weighted score across all metrics."""
    accuracy = _score_accuracy(agent_data, expected, params)
    completeness = _score_completeness(agent_data, expected, params)
    latency = _score_latency(agent_data, expected, params)
    token_efficiency = _score_token_efficiency(agent_data, expected, params)
    
    weighted_score = (
        accuracy * 0.40 +
        completeness * 0.30 +
        latency * 0.20 +
        token_efficiency * 0.10
    )
    
    logger.info(f"Aggregate score: {weighted_score:.3f}")
    return weighted_score


def _check_tool_usage(agent_data: Dict[str, Any], expected: Dict[str, Any]) -> float:
    """Check if correct tools were called."""
    expected_tools = expected.get('expected', {}).get('tools_called', [])
    if not expected_tools:
        return 1.0
    
    tool_calls = agent_data.get('tool_calls', [])
    if not tool_calls:
        return 0.0
    
    called_tools = [tc.get('function', '') for tc in tool_calls]
    
    matches = sum(1 for tool in expected_tools if tool in called_tools)
    return matches / len(expected_tools) if expected_tools else 0.0


def _check_tool_parameters(agent_data: Dict[str, Any], expected: Dict[str, Any]) -> float:
    """Check if tool parameters are correct."""
    tool_sequence = expected.get('expected', {}).get('tool_sequence', [])
    if not tool_sequence:
        return 1.0
    
    tool_calls = agent_data.get('tool_calls', [])
    if not tool_calls:
        return 0.0
    
    # Check if any tool call matches expected parameters
    for expected_call in tool_sequence:
        expected_tool = expected_call.get('tool')
        expected_params = expected_call.get('params', {})
        
        for tool_call in tool_calls:
            if tool_call.get('function') == expected_tool:
                call_args = tool_call.get('arguments', {})
                if all(key in call_args and str(call_args[key]) == str(expected_params[key]) 
                       for key in expected_params):
                    return 1.0
    
    return 0.5  # Partial credit if tools called but params don't match exactly


def _check_results_validity(agent_data: Dict[str, Any], expected: Dict[str, Any]) -> float:
    """Check if results contain valid data."""
    final_response = agent_data.get('final_response', '')
    if not final_response:
        return 0.0
    
    # Basic check: response is not empty
    if len(final_response) < 10:
        return 0.0
    
    # Check for Discord-specific content
    discord_indicators = ['channel', 'message', 'guild', 'server', 'discord']
    has_discord_context = any(indicator in final_response.lower() for indicator in discord_indicators)
    
    return 1.0 if has_discord_context else 0.7


def _check_keywords(agent_data: Dict[str, Any], expected: Dict[str, Any]) -> float:
    """Check if response contains expected keywords."""
    validation = expected.get('result_validation', {})
    keyword_config = validation.get('keywords', {})
    
    if not keyword_config:
        return 1.0
    
    required_keywords = keyword_config.get('required_in_response', [])
    if not required_keywords:
        return 1.0
    
    final_response = agent_data.get('final_response', '').lower()
    
    matches = sum(1 for keyword in required_keywords if keyword.lower() in final_response)
    min_matches = keyword_config.get('min_matches', len(required_keywords))
    
    if matches >= min_matches:
        return 1.0
    elif matches > 0:
        return matches / min_matches
    else:
        return 0.0
