"""
Agno Discord Agent Optimization - Programmatic Example

This demonstrates the new programmatic Convergence SDK interface for optimizing
Discord agent workflows without YAML configuration files.
"""

import asyncio
from convergence.types import (
    ConvergenceConfig,
    ApiConfig,
    SearchSpaceConfig,
    RunnerConfig,
    EvaluationConfig,
    AgentConfig,
)
from convergence.sdk import run_optimization


async def discord_optimization_example():
    """Example of optimizing Discord agent with programmatic interface."""
    
    # Load test cases (simplified for demo)
    test_cases = [
        {
            "id": "list_server_channels",
            "function": "list_channels",
            "description": "List all channels in a Discord server",
            "input": {
                "query": "List all channels in my Discord server",
                "guild_id": "1342350434794213386"
            },
            "expected": {
                "tools_called": ["list_channels"],
                "tool_sequence": [
                    {
                        "tool": "list_channels",
                        "params": {
                            "guild_id": "1342350434794213386"
                        }
                    }
                ],
                "result_validation": {
                    "schema": {
                        "type": "array",
                        "items": {
                            "required_fields": ["id", "name", "type"],
                            "optional_fields": ["topic", "position", "nsfw"]
                        }
                    },
                    "data_checks": {
                        "has_channels": True,
                        "min_channels": 1,
                        "has_channel_ids": True,
                        "has_channel_names": True
                    },
                    "keywords": {
                        "required_in_response": ["channels", "server"],
                        "match_type": "case_insensitive",
                        "min_matches": 1
                    }
                },
                "success_criteria": {
                    "tools_called_correctly": True,
                    "tool_sequence_logical": True,
                    "valid_response_format": True,
                    "data_completeness_percent": 80,
                    "comprehensive_output": True
                }
            },
            "metadata": {
                "category": "integration_test",
                "difficulty": "easy",
                "estimated_tokens": 400,
                "weight": 1.0,
                "note": "Agent must use list_channels tool with correct guild_id"
            }
        }
    ]
    
    # Define configuration as Python objects
    config = ConvergenceConfig(
        api=ApiConfig(
            name="agno_discord_agent",
            kind="http",
            # Note: endpoint is a placeholder; actual models are in agent.models registry
            endpoint="https://placeholder-see-agent-models-registry",
            request_timeout=120.0,
            # Azure OpenAI authentication
            auth_type="api_key",
            auth_token_env="AZURE_API_KEY",
            auth_header_name="api-key"
        ),
        search_space=SearchSpaceConfig(
            parameters={
                "model": {
                    "type": "categorical",
                    "choices": ["gpt-4.1"]  # Removed o4-mini due to API version issues
                },
                "temperature": {
                    "type": "categorical",
                    "choices": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
                },
                "max_completion_tokens": {
                    "type": "int",
                    "min": 500,
                    "max": 4000,
                    "step": 500
                },
                "instruction_style": {
                    "type": "categorical",
                    "choices": ["minimal", "detailed", "structured"]
                },
                "tool_strategy": {
                    "type": "categorical",
                    "choices": ["include_all", "include_specific"]
                }
            }
        ),
        runner=RunnerConfig(
            generations=3,
            population=4,
            seed=42,
            early_stopping={"enabled": False}
        ),
        evaluation=EvaluationConfig(
            required_metrics=["accuracy", "completeness", "latency_seconds", "token_efficiency"],
            weights={
                "accuracy": 0.40,
                "completeness": 0.30,
                "latency_seconds": 0.20,
                "token_efficiency": 0.10
            }
        ),
        agent=AgentConfig(
            models={
                "gpt-4.1": {
                    "endpoint": "https://heycontext-resource.cognitiveservices.azure.com/openai/deployments/gpt-4.1/chat/completions?api-version=2025-01-01-preview",
                    "description": "GPT-4 for high-quality responses"
                },
                "o4-mini": {
                    "endpoint": "https://heycontext-resource.cognitiveservices.azure.com/openai/deployments/o4-mini/chat/completions?api-version=2025-01-01-preview",
                    "description": "o4-mini model for fast, cost-effective testing"
                }
            },
            discord_auth={
                "bot_token_env": "DISCORD_BOT_TOKEN"
            }
        )
    )
    
    # Define evaluator function
    def discord_evaluator(prediction: dict, expected: dict, *, context: dict = None) -> dict:
        """
        Score Discord agent responses across multiple metrics.
        
        This wraps the custom Discord evaluator logic for programmatic use.
        """
        from discord_evaluator import score_discord_agent_response
        
        # Extract parameters from context for evaluation
        params = context.get('params', {}) if context else {}
        result = prediction.get('result')
        
        # Score each metric
        scores = {}
        
        # Accuracy (40% weight)
        accuracy = score_discord_agent_response(
            result, expected, params, metric="accuracy"
        )
        scores['accuracy'] = accuracy
        
        # Completeness (30% weight)
        completeness = score_discord_agent_response(
            result, expected, params, metric="completeness"
        )
        scores['completeness'] = completeness
        
        # Latency (20% weight)
        latency = score_discord_agent_response(
            result, expected, params, metric="latency_seconds"
        )
        scores['latency_seconds'] = latency
        
        # Token efficiency (10% weight)
        token_efficiency = score_discord_agent_response(
            result, expected, params, metric="token_efficiency"
        )
        scores['token_efficiency'] = token_efficiency
        
        # Aggregate weighted score
        aggregate = (
            accuracy * 0.40 +
            completeness * 0.30 +
            latency * 0.20 +
            token_efficiency * 0.10
        )
        scores['score'] = aggregate
        
        return scores
    
    # Run optimization
    print("Starting Discord agent optimization with programmatic interface...")
    print("=" * 80)
    
    try:
        result = await run_optimization(
            config=config,
            evaluator=discord_evaluator,
            test_cases=test_cases,
            logging_mode="summary"
        )
        
        # Print results
        print("\n" + "=" * 80)
        print("OPTIMIZATION COMPLETE!")
        print("=" * 80)
        print(f"Best Config: {result.best_config}")
        print(f"Best Score: {result.best_score:.3f}")
        print(f"Configs Generated: {result.configs_generated}")
        print(f"Generations Run: {result.generations_run}")
        print(f"Run ID: {result.optimization_run_id}")
        print(f"Timestamp: {result.timestamp}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        raise


def load_full_test_cases():
    """Load full test cases from JSON file."""
    import json
    from pathlib import Path
    
    test_cases_path = Path(__file__).parent / "discord_test_cases.json"
    
    if not test_cases_path.exists():
        print(f"Warning: Test cases file not found: {test_cases_path}")
        return []
    
    with open(test_cases_path, 'r') as f:
        data = json.load(f)
    
    # Return just the test cases list
    return data.get('test_cases', [])


async def discord_optimization_full():
    """Example with full test cases loaded from JSON file."""
    
    # Load all test cases from JSON
    test_cases = load_full_test_cases()
    
    if not test_cases:
        print("No test cases loaded. Exiting.")
        return
    
    print(f"Loaded {len(test_cases)} test cases")
    
    # Define configuration
    config = ConvergenceConfig(
        api=ApiConfig(
            name="agno_discord_agent",
            kind="http",
            endpoint="https://placeholder-see-agent-models-registry",
            request_timeout=120.0,
            # Azure OpenAI authentication
            auth_type="api_key",
            auth_token_env="AZURE_API_KEY",
            auth_header_name="api-key"
        ),
        search_space=SearchSpaceConfig(
            parameters={
                "model": {
                    "type": "categorical",
                    "choices": ["gpt-4.1"]
                },
                "temperature": {
                    "type": "categorical",
                    "choices": [0.0, 0.4, 0.8]
                },
                "instruction_style": {
                    "type": "categorical",
                    "choices": ["minimal", "detailed"]
                },
                "tool_strategy": {
                    "type": "categorical",
                    "choices": ["include_all"]
                }
            }
        ),
        runner=RunnerConfig(
            generations=2,
            population=2,
            seed=42
        ),
        evaluation=EvaluationConfig(
            required_metrics=["accuracy", "completeness", "latency_seconds", "token_efficiency"],
            weights={
                "accuracy": 0.40,
                "completeness": 0.30,
                "latency_seconds": 0.20,
                "token_efficiency": 0.10
            }
        ),
        agent=AgentConfig(
            models={
                "gpt-4.1": {
                    "endpoint": "https://heycontext-resource.cognitiveservices.azure.com/openai/deployments/gpt-4.1/chat/completions?api-version=2025-01-01-preview",
                    "description": "GPT-4 for high-quality responses"
                }
            },
            discord_auth={
                "bot_token_env": "DISCORD_BOT_TOKEN"
            }
        )
    )
    
    # Use evaluator module directly
    from discord_evaluator import score_discord_agent_response
    
    def evaluator_wrapper(prediction: dict, expected: dict, *, context: dict = None) -> dict:
        """Wrapper that calls the Discord evaluator."""
        params = context.get('params', {}) if context else {}
        result = prediction.get('result')
        
        return {
            'score': score_discord_agent_response(result, expected, params)
        }
    
    # Run optimization
    print("Starting Discord agent optimization...")
    print("=" * 80)
    
    result = await run_optimization(
        config=config,
        evaluator=evaluator_wrapper,
        test_cases=test_cases,
        logging_mode="verbose"
    )
    
    print("\n" + "=" * 80)
    print("RESULTS:")
    print(f"Best Config: {result.best_config}")
    print(f"Best Score: {result.best_score:.3f}")
    print(f"Configs: {result.configs_generated}")
    print(f"Generations: {result.generations_run}")
    
    return result


if __name__ == "__main__":
    # Run simple example
    asyncio.run(discord_optimization_example())
    
    # Or run full example with all test cases
    # asyncio.run(discord_optimization_full())

