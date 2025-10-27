"""
Custom evaluator for Agno Gmail agent optimization with Azure OpenAI.

Evaluates agent performance across 4 metrics:
1. Accuracy (40%): Tool usage correctness and result relevance
2. Completeness (30%): Data field presence and population
3. Latency (20%): Response time
4. Token Efficiency (10%): Value per token used

Validation Features:
- Schema validation (required/optional fields)
- Data completeness checks (field presence, value reasonableness)
- Keyword matching (case-insensitive, flexible)
- Email-specific validation (from, to, subject, body, thread_id)
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


def score_gmail_agent_response(
    result: Any,
    expected: Dict[str, Any],
    params: Dict[str, Any],
    metric: Optional[str] = None
) -> float:
    """
    Score Agno Gmail agent responses across multiple metrics.
    
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
    
    # Extract structured data
    parsed = {
        'tool_calls': [],
        'final_response': '',
        'tool_results': [],
        'tokens_used': {},
        'latency_seconds': 0.0
    }
    
    # Extract from Gmail agent runner format (direct fields)
    if 'final_response' in data:
        parsed['final_response'] = data.get('final_response', '')
        parsed['tool_calls'] = data.get('tool_calls', [])
        parsed['tool_results'] = data.get('tool_results', [])
        parsed['tokens_used'] = data.get('tokens_used', {})
        parsed['latency_seconds'] = data.get('latency_seconds', 0.0)
    
    # Extract from Azure OpenAI format (fallback)
    elif 'choices' in data and len(data['choices']) > 0:
        choice = data['choices'][0]
        message = choice.get('message', {})
        parsed['final_response'] = message.get('content', '')
        if 'tool_calls' in message:
            parsed['tool_calls'] = message['tool_calls']
    
    # Extract usage
    if 'usage' in data:
        parsed['tokens_used'] = data['usage']
    
    if 'tool_results' in data:
        parsed['tool_results'] = data['tool_results']
    
    if 'latency_seconds' in data:
        parsed['latency_seconds'] = data['latency_seconds']
    elif 'latency_ms' in data:
        parsed['latency_seconds'] = data['latency_ms'] / 1000.0
    
    return parsed


def _score_accuracy(agent_data: Dict[str, Any], expected: Dict[str, Any], params: Dict[str, Any]) -> float:
    """Score accuracy: Did agent call correct tools with correct parameters? Components: 30% tool, 20% params, 30% results, 20% keywords."""
    score = 0.0
    
    tool_score = _check_tool_usage(agent_data, expected)
    score += tool_score * 0.30
    logger.info(f"  Tool usage score: {tool_score:.3f}")
    
    param_score = _check_tool_parameters(agent_data, expected)
    score += param_score * 0.20
    logger.info(f"  Tool params score: {param_score:.3f}")
    
    results_score = _check_results_validity(agent_data, expected)
    score += results_score * 0.30
    logger.info(f"  Results validity score: {results_score:.3f}")
    
    keyword_score = _check_keywords(agent_data, expected)
    score += keyword_score * 0.20
    logger.info(f"  Keyword matching score: {keyword_score:.3f}")
    
    final_score = min(1.0, max(0.0, score))
    logger.info(f"📊 ACCURACY final score: {final_score:.3f}")
    return final_score


def _check_tool_usage(agent_data: Dict[str, Any], expected: Dict[str, Any]) -> float:
    """Check if correct tool(s) were called."""
    tool_calls = agent_data.get('tool_calls', [])
    
    actual_tools = set()
    for call in tool_calls:
        if isinstance(call, dict):
            if 'function' in call:
                actual_tools.add(call['function'].get('name', ''))
            elif 'name' in call:
                actual_tools.add(call['name'])
    
    if 'tool_called' in expected:
        expected_tool = expected['tool_called']
        if expected_tool in actual_tools:
            return 1.0
        return 0.0
    elif 'tools_called' in expected:
        expected_tools = set(expected['tools_called'])
        if expected_tools == actual_tools:
            return 1.0
        if not expected_tools or not actual_tools:
            return 0.0
        overlap = len(expected_tools & actual_tools)
        total = len(expected_tools)
        return overlap / total
    
    return 0.5


def _check_tool_parameters(agent_data: Dict[str, Any], expected: Dict[str, Any]) -> float:
    """Check if tool parameters are correct."""
    tool_calls = agent_data.get('tool_calls', [])
    if not tool_calls:
        return 0.0
    
    if 'tool_params' in expected:
        expected_params = expected['tool_params']
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            
            actual_params = {}
            if 'function' in call:
                args = call['function'].get('arguments', {})
                if isinstance(args, str):
                    try:
                        actual_params = json.loads(args)
                    except:
                        pass
                else:
                    actual_params = args
            elif 'arguments' in call:
                actual_params = call['arguments']
            
            if not expected_params:
                return 1.0
            
            matches = 0
            total = len(expected_params)
            for key, value in expected_params.items():
                if key in actual_params:
                    if value == "extracted_from_previous_call":
                        matches += 1
                    elif str(value).lower() in str(actual_params[key]).lower():
                        matches += 1
            
            return matches / total if total > 0 else 1.0
    
    return 1.0


def _check_results_validity(agent_data: Dict[str, Any], expected: Dict[str, Any]) -> float:
    """Check if results match expected validation criteria."""
    results = agent_data.get('tool_results', [])
    validation = expected.get('result_validation', {})
    
    if not validation:
        return 1.0
    
    score = 0.0
    checks = 0
    
    if 'schema' in validation:
        schema_score = _validate_schema(results, validation['schema'])
        score += schema_score
        checks += 1
        logger.debug(f"    Schema validation: {schema_score:.3f}")
    
    if 'data_checks' in validation:
        data_score = _validate_data_checks(results, validation['data_checks'])
        score += data_score
        checks += 1
        logger.debug(f"    Data checks: {data_score:.3f}")
    
    return score / checks if checks > 0 else 1.0


def _validate_schema(results: List[Any], schema: Dict[str, Any]) -> float:
    """Validate response schema (required fields present)."""
    if not results:
        return 0.0
    
    result = results[0] if isinstance(results, list) and results else results
    
    if schema.get('type') == 'array':
        if not isinstance(result, list):
            return 0.0
        if len(result) == 0:
            return 0.5
        item = result[0]
        required_fields = schema.get('items', {}).get('required_fields', [])
        if not required_fields:
            return 1.0
        present = sum(1 for field in required_fields if field in item and item[field])
        return present / len(required_fields)
    elif schema.get('type') == 'object':
        if not isinstance(result, dict):
            return 0.0
        required_fields = schema.get('required_fields', [])
        if not required_fields:
            return 1.0
        present = sum(1 for field in required_fields if field in result and result[field])
        return present / len(required_fields)
    
    return 1.0


def _validate_data_checks(results: List[Any], checks: Dict[str, Any]) -> float:
    """Validate data completeness and reasonableness."""
    if not results:
        return 0.0
    
    result = results[0] if isinstance(results, list) and results else results
    score = 0.0
    total_checks = 0
    
    # Email-specific validations
    if 'has_from_field' in checks:
        has_from = False
        if isinstance(result, dict):
            has_from = 'from' in result
        elif isinstance(result, list):
            has_from = any('from' in item for item in result if isinstance(item, dict))
        if has_from:
            score += 1.0
        total_checks += 1
    
    if 'has_subject_field' in checks:
        has_subject = False
        if isinstance(result, dict):
            has_subject = 'subject' in result
        elif isinstance(result, list):
            has_subject = any('subject' in item for item in result if isinstance(item, dict))
        if has_subject:
            score += 1.0
        total_checks += 1
    
    if 'has_body_field' in checks:
        has_body = False
        if isinstance(result, dict):
            has_body = 'body' in result or 'content' in result
        elif isinstance(result, list):
            has_body = any('body' in item or 'content' in item for item in result if isinstance(item, dict))
        if has_body:
            score += 1.0
        total_checks += 1
    
    if 'has_date_field' in checks:
        has_date = False
        if isinstance(result, dict):
            has_date = 'date' in result
        elif isinstance(result, list):
            has_date = any('date' in item for item in result if isinstance(item, dict))
        if has_date:
            score += 1.0
        total_checks += 1
    
    if 'min_results' in checks:
        if isinstance(result, list) and len(result) >= checks['min_results']:
            score += 1.0
        total_checks += 1
    
    if 'has_thread_id' in checks:
        has_thread = False
        if isinstance(result, dict):
            has_thread = 'thread_id' in result
        elif isinstance(result, list):
            has_thread = any('thread_id' in item for item in result if isinstance(item, dict))
        if has_thread:
            score += 1.0
        total_checks += 1
    
    return score / total_checks if total_checks > 0 else 1.0


def _check_keywords(agent_data: Dict[str, Any], expected: Dict[str, Any]) -> float:
    """Check if expected keywords appear in results."""
    validation = expected.get('result_validation', {})
    keyword_config = validation.get('keywords', {})
    
    if not keyword_config:
        return 1.0
    
    required_keywords = keyword_config.get('required_in_results', [])
    if not required_keywords:
        required_keywords = keyword_config.get('required_in_subject', [])
    if not required_keywords:
        required_keywords = keyword_config.get('required_in_body', [])
    
    if not required_keywords:
        return 1.0
    
    results = agent_data.get('tool_results', [])
    results_text = json.dumps(results).lower()
    final_response = agent_data.get('final_response', '')
    combined_text = results_text + ' ' + str(final_response).lower()
    
    matches = sum(1 for kw in required_keywords if kw.lower() in combined_text)
    min_matches = keyword_config.get('min_matches', len(required_keywords))
    score = min(1.0, matches / min_matches) if min_matches > 0 else 1.0
    
    return score


def _score_completeness(agent_data: Dict[str, Any], expected: Dict[str, Any], params: Dict[str, Any]) -> float:
    """Score completeness: Are all expected data fields present and populated? 70% required fields, 30% optional fields."""
    results = agent_data.get('tool_results', [])
    if not results:
        logger.info("📊 COMPLETENESS score: 0.000 (no results)")
        return 0.0
    
    validation = expected.get('result_validation', {})
    schema = validation.get('schema', {})
    
    if not schema:
        return 1.0
    
    score = 0.0
    
    if schema.get('type') == 'array':
        if not isinstance(results, list) or len(results) == 0:
            score = 0.0
        else:
            item = results[0] if isinstance(results[0], list) and results[0] else results
            if isinstance(item, list) and len(item) > 0:
                item = item[0]
            score = _calculate_field_completeness(item, schema.get('items', {}))
    elif schema.get('type') == 'object':
        item = results[0] if isinstance(results, list) else results
        score = _calculate_field_completeness(item, schema)
    else:
        score = 0.5
    
    logger.info(f"📊 COMPLETENESS score: {score:.3f}")
    return score


def _calculate_field_completeness(item: Any, schema: Dict[str, Any]) -> float:
    """Calculate field completeness score."""
    if not isinstance(item, dict):
        return 0.0
    
    required_fields = schema.get('required_fields', [])
    optional_fields = schema.get('optional_fields', [])
    
    if not required_fields:
        return 1.0
    
    required_present = sum(1 for field in required_fields if field in item and item[field])
    required_score = (required_present / len(required_fields)) * 0.70
    
    optional_present = sum(1 for field in optional_fields if field in item and item[field])
    optional_score = (optional_present / len(optional_fields)) * 0.30 if optional_fields else 0.0
    
    return required_score + optional_score


def _score_latency(agent_data: Dict[str, Any], expected: Dict[str, Any], params: Dict[str, Any]) -> float:
    """Score latency: How fast did the agent respond? <5s=1.0, <10s=0.8, <20s=0.6, <30s=0.4, >=30s=0.2"""
    latency = agent_data.get('latency_seconds', 0.0)
    
    if latency < 5:
        score = 1.0
    elif latency < 10:
        score = 0.8
    elif latency < 20:
        score = 0.6
    elif latency < 30:
        score = 0.4
    else:
        score = 0.2
    
    logger.info(f"⏱️  LATENCY: {latency:.2f}s, score: {score:.3f}")
    return score


def _score_token_efficiency(agent_data: Dict[str, Any], expected: Dict[str, Any], params: Dict[str, Any]) -> float:
    """Score token efficiency: Value delivered per token used."""
    tokens_used = agent_data.get('tokens_used', {})
    total_tokens = tokens_used.get('total_tokens', 0)
    
    if total_tokens == 0:
        logger.info("💰 TOKEN EFFICIENCY: unknown, score: 0.500")
        return 0.5
    
    estimated_tokens = expected.get('metadata', {}).get('estimated_tokens', 500)
    
    if total_tokens <= estimated_tokens * 0.7:
        score = 1.0
    elif total_tokens <= estimated_tokens:
        score = 0.9
    elif total_tokens <= estimated_tokens * 1.3:
        score = 0.7
    elif total_tokens <= estimated_tokens * 1.5:
        score = 0.5
    else:
        score = 0.3
    
    logger.info(f"💰 TOKEN EFFICIENCY: {total_tokens} tokens (est: {estimated_tokens}), score: {score:.3f}")
    return score


def _aggregate_score(agent_data: Dict[str, Any], expected: Dict[str, Any], params: Dict[str, Any]) -> float:
    """Calculate weighted aggregate score across all metrics."""
    accuracy = _score_accuracy(agent_data, expected, params)
    completeness = _score_completeness(agent_data, expected, params)
    latency = _score_latency(agent_data, expected, params)
    token_eff = _score_token_efficiency(agent_data, expected, params)
    
    aggregate = (
        accuracy * 0.40 +
        completeness * 0.30 +
        latency * 0.20 +
        token_eff * 0.10
    )
    
    logger.info(f"📊 AGGREGATE SCORE: {aggregate:.3f}")
    logger.info(f"   Accuracy: {accuracy:.3f} (40%)")
    logger.info(f"   Completeness: {completeness:.3f} (30%)")
    logger.info(f"   Latency: {latency:.3f} (20%)")
    logger.info(f"   Token Eff: {token_eff:.3f} (10%)")
    
    return aggregate

