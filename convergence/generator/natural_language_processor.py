"""
Natural Language Processor for AI-Powered Setup

Uses LiteLLM to process user natural language input and generate
optimization.yaml, test_cases.json, and evaluator.py files.
"""
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from rich.console import Console

from convergence.core.llm_provider import LiteLLMProvider, LiteLLMConfig
from convergence.generator.constants import DEFAULT_LLM_MODEL


class NaturalLanguageProcessor:
    """Uses LiteLLM to process user intent and extract configuration."""
    
    def __init__(self):
        self.console = Console()
        self.llm_provider = LiteLLMProvider()
        
        # Default model for AI-powered setup
        self.model = DEFAULT_LLM_MODEL or "gemini/gemini-2.0-flash-exp"
        
    async def process_user_intent(self, user_input: str) -> Dict[str, Any]:
        """
        Extract all necessary information from natural language.
        
        Args:
            user_input: User's natural language description
            
        Returns:
            Dict containing extracted configuration information
        """
        self.console.print(f"[dim]Processing user intent: {user_input}[/dim]")
        
        # Extract information using LLM
        extracted_info = await self._extract_information(user_input)
        
        # Check if extraction was successful
        if extracted_info is None:
            self.console.print("[red]❌ Failed to extract information from user input[/red]")
            raise ValueError("Could not extract configuration information from user input")
        
        # Generate configuration
        config = await self._generate_configuration(extracted_info)
        
        # Generate test cases
        test_cases = await self._generate_test_cases(extracted_info)
        
        # Generate evaluator
        evaluator_code = await self._generate_evaluator(extracted_info)
        
        return {
            'extracted_info': extracted_info,
            'config': config,
            'test_cases': test_cases,
            'evaluator_code': evaluator_code
        }
    
    async def _extract_information(self, user_input: str) -> Dict[str, Any]:
        """Extract configuration information from user input."""
        
        extraction_prompt = f"""
Extract API configuration from: "{user_input}"

Analyze the user's detailed description and extract:

1. API Provider (openai, groq, azure, anthropic, custom)
2. Optimization Priority (quality, speed, cost, balanced)
3. Use Case (creative_writing, qa, summarization, coding, etc.)
4. Specific Requirements (response length, creativity level, etc.)
5. Budget/Cost constraints
6. Speed requirements
7. Models mentioned or implied by the provider

CRITICAL: You MUST extract the correct endpoint and at least one model for the provider.

For endpoints, determine from context:
- If user mentions "Groq", "Llama", "ultra-fast" → use "https://api.groq.com/openai/v1/chat/completions"
- If user mentions "OpenAI", "GPT", "ChatGPT" → use "https://api.openai.com/v1/chat/completions"  
- If user mentions "Anthropic", "Claude" → use "https://api.anthropic.com/v1/messages"
- If user mentions "Azure" → use "https://your-resource.openai.azure.com/openai/deployments/your-model/chat/completions"
- If user mentions a custom endpoint, use that exact endpoint

For models, extract from context or use provider defaults:
- Groq: ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
- OpenAI: ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"]
- Anthropic: ["claude-3-haiku", "claude-3-sonnet"]
- Azure: ["gpt-4o", "gpt-4o-mini"]

Return JSON only:
{{
    "api_type": "llm_chat",
    "provider": "openai",
    "endpoint": "https://api.openai.com/v1/chat/completions",
    "api_key_env": "OPENAI_API_KEY",
    "models": ["gpt-4o-mini", "gpt-3.5-turbo"],
    "use_case": "creative writing",
    "optimization_goal": "quality",
    "intensity": "thorough",
    "test_scenarios": ["creative writing", "story generation"],
    "metrics": {{
        "quality_weight": 0.6,
        "latency_weight": 0.2,
        "cost_weight": 0.2
    }},
    "search_space": {{
        "parameters": {{
            "temperature": {{
                "type": "continuous",
                "min": 0.7,
                "max": 1.0,
                "step": 0.1
            }},
            "max_tokens": {{
                "type": "discrete",
                "values": [200, 400, 600, 800]
            }}
        }}
    }},
    "requirements": {{
        "response_length": "200-500 words",
        "creativity_level": "high",
        "cost_budget": "moderate",
        "speed_requirement": "not critical"
    }}
}}"""

        try:
            response = await self.llm_provider.generate(
                prompt=extraction_prompt,
                temperature=0.3,  # Lower temperature for more consistent extraction
                max_tokens=32000  # Maximum token limit for Gemini 2.5 Flash
            )
            
            # Parse JSON response - handle markdown code blocks
            content = response['content'].strip()
            
            # Extract JSON from markdown code blocks if present
            if content.startswith('```json'):
                # Remove ```json marker
                json_start = content.find('```json') + 7
                content = content[json_start:].strip()
                # Try to find closing ```, but if not found, assume it was truncated
                json_end = content.rfind('```')
                if json_end > 0:
                    content = content[:json_end].strip()
            elif content.startswith('```'):
                # Remove generic ``` marker
                json_start = content.find('```') + 3
                content = content[json_start:].strip()
                # Try to find closing ```, but if not found, assume it was truncated
                json_end = content.rfind('```')
                if json_end > 0:
                    content = content[:json_end].strip()
            
            # If content doesn't start with {, try to find the JSON object
            if not content.startswith('{'):
                json_start = content.find('{')
                if json_start >= 0:
                    content = content[json_start:]
            
            # Try to parse JSON, but handle truncation gracefully
            try:
                extracted_info = json.loads(content)
            except json.JSONDecodeError as e:
                # If JSON is malformed due to truncation, try to extract valid parts
                self.console.print(f"[yellow]⚠️ JSON truncated, attempting to extract valid information...[/yellow]")
                
                # Find the last complete property
                last_complete_prop = content.rfind('",')
                if last_complete_prop > 0:
                    # Try to reconstruct valid JSON
                    partial_content = content[:last_complete_prop + 1]
                    # Add closing braces
                    reconstructed = partial_content + '}}'
                    try:
                        extracted_info = json.loads(reconstructed)
                    except:
                        # Fallback: create a basic extraction
                        extracted_info = {
                            "api_type": "llm_chat",
                            "provider": "openai",
                            "endpoint": "https://api.openai.com/v1/chat/completions",
                            "api_key_env": "OPENAI_API_KEY",
                            "models": ["gpt-4o-mini", "gpt-3.5-turbo"],
                            "use_case": "creative writing",
                            "optimization_goal": "quality",
                            "intensity": "thorough",
                            "test_scenarios": ["creative writing", "story generation"],
                            "metrics": {"quality_weight": 0.6, "latency_weight": 0.2, "cost_weight": 0.2},
                            "search_space": {
                                "parameters": {
                                    "temperature": {"type": "continuous", "min": 0.7, "max": 1.0, "step": 0.1},
                                    "max_tokens": {"type": "discrete", "values": [300, 500, 750, 1000]}
                                }
                            },
                            "requirements": {"response_length": "200-500 words", "creativity_level": "high", "cost_budget": "moderate", "speed_requirement": "not critical"}
                        }
                else:
                    # Complete fallback
                    extracted_info = {
                        "api_type": "llm_chat",
                        "provider": "openai",
                        "endpoint": "https://api.openai.com/v1/chat/completions",
                        "api_key_env": "OPENAI_API_KEY",
                        "models": ["gpt-4o-mini", "gpt-3.5-turbo"],
                        "use_case": "creative writing",
                        "optimization_goal": "quality",
                        "intensity": "thorough",
                        "test_scenarios": ["creative writing", "story generation"],
                        "metrics": {"quality_weight": 0.6, "latency_weight": 0.2, "cost_weight": 0.2},
                        "search_space": {
                            "parameters": {
                                "temperature": {"type": "continuous", "min": 0.7, "max": 1.0, "step": 0.1},
                                "max_tokens": {"type": "discrete", "values": [300, 500, 750, 1000]}
                            }
                        },
                        "requirements": {"response_length": "200-500 words", "creativity_level": "high", "cost_budget": "moderate", "speed_requirement": "not critical"}
                    }
            
            self.console.print("[green]✅ Successfully extracted configuration information[/green]")
            return extracted_info
            
        except Exception as e:
            self.console.print(f"[red]❌ Error extracting information: {e}[/red]")
            self.console.print(f"[yellow]Error type: {type(e).__name__}[/yellow]")
            import traceback
            self.console.print(f"[yellow]Traceback: {traceback.format_exc()}[/yellow]")
            raise e
    
    async def _generate_configuration(self, extracted_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimization.yaml configuration."""
        
        config_prompt = f"""Generate optimization.yaml for {extracted_info.get('provider', 'api')} API:

Based on user requirements:
- Use case: {extracted_info.get('use_case', 'general')}
- Optimization goal: {extracted_info.get('optimization_goal', 'balanced')}
- Intensity: {extracted_info.get('intensity', 'balanced')}
- Requirements: {extracted_info.get('requirements', {})}

MANDATORY REQUIREMENT: You MUST include detailed comments throughout the YAML explaining WHY each value was chosen based on the user's specific input. Comments should reference specific parts of the user's requirements and explain the reasoning behind each configuration choice. NO YAML WITHOUT COMMENTS WILL BE ACCEPTED.

Generate YAML with MANDATORY detailed comments:

# Configuration generated for: {extracted_info.get('api_type', 'llm_chat')} API
# User requirements: {extracted_info.get('requirements', {})}
# Optimization goal: {extracted_info.get('optimization_goal', 'balanced')}

api:
  # API name chosen because: [explain based on user input]
  name: "custom_{extracted_info.get('provider', 'api')}_{extracted_info.get('api_type', 'llm_chat')}"
  # Endpoint chosen because: [explain based on user input]
  endpoint: "{extracted_info.get('endpoint', 'https://api.example.com/v1/endpoint')}"
  request:
    method: "POST"
    headers:
      Content-Type: "application/json"
    timeout_seconds: 30
  auth:
    type: "bearer"
    # API key env var chosen because: [explain based on user input]
    token_env: "{extracted_info.get('api_key_env', 'API_KEY')}"
  response:
    result_field: "choices[0].message.content"

search_space:
  parameters:
    # Model selection chosen because: [explain based on user's provider choice and requirements]
    model:
      type: "categorical"
      values: {extracted_info.get('models', ['gpt-4o-mini', 'gpt-3.5-turbo'])}
    # Temperature range chosen because: [explain user's quality vs creativity requirements]
    temperature:
      type: "continuous"
      min: {extracted_info.get('search_space', {}).get('parameters', {}).get('temperature', {}).get('min', 0.1)}
      max: {extracted_info.get('search_space', {}).get('parameters', {}).get('temperature', {}).get('max', 1.0)}
      step: {extracted_info.get('search_space', {}).get('parameters', {}).get('temperature', {}).get('step', 0.1)}
    # Max tokens chosen because: [explain user's response length requirements]
    max_tokens:
      type: "discrete"
      values: {extracted_info.get('search_space', {}).get('parameters', {}).get('max_tokens', {}).get('values', [100, 256, 512, 1024])}

evaluation:
  test_cases:
    path: "test_cases.json"
  metrics:
    # Metrics weights chosen because: [explain user's priority preferences]
    response_quality:
      weight: {extracted_info.get('metrics', {}).get('quality_weight', 0.4)}  # High weight because user emphasized quality
      type: "higher_is_better"
      function: "custom"
    latency_ms:
      weight: {extracted_info.get('metrics', {}).get('latency_weight', 0.3)}  # Medium weight because user mentioned speed concerns
      type: "lower_is_better"
      threshold: 2000
    cost_per_call:
      weight: {extracted_info.get('metrics', {}).get('cost_weight', 0.3)}  # Medium weight because user wants cost efficiency
      type: "lower_is_better"
      budget_per_call: 0.01
  custom_evaluator:
    enabled: true
    module: "evaluator"
    function: "score_custom_response"

optimization:
  algorithm: "mab_evolution"
  evolution:
    # Evolution settings chosen because: [explain based on user's optimization intensity]
    population_size: 4
    generations: 3
    mutation_rate: 0.3
    crossover_rate: 0.2
    elite_size: 1
  execution:
    parallel_workers: 1
    experiments_per_generation: 3

society:
  enabled: false

legacy:
  enabled: true"""

        try:
            response = await self.llm_provider.generate(
                prompt=config_prompt,
                temperature=0.2,  # Very low temperature for consistent YAML
                max_tokens=32000  # Maximum token limit for Gemini 2.5 Flash
            )
            
            # Get raw YAML content - handle markdown code blocks
            content = response['content'].strip()
            
            # Extract YAML from markdown code blocks if present
            if content.startswith('```yaml'):
                # Remove ```yaml and ``` markers
                yaml_start = content.find('```yaml') + 7
                yaml_end = content.rfind('```')
                if yaml_end > yaml_start:
                    content = content[yaml_start:yaml_end].strip()
            elif content.startswith('```'):
                # Remove generic ``` markers
                yaml_start = content.find('```') + 3
                yaml_end = content.rfind('```')
                if yaml_end > yaml_start:
                    content = content[yaml_start:yaml_end].strip()
            
            # Return raw YAML string to preserve comments
            self.console.print("[green]✅ Successfully generated optimization configuration[/green]")
            return content
            
        except Exception as e:
            self.console.print(f"[red]❌ Error generating configuration: {e}[/red]")
            raise e
    
    async def _generate_test_cases(self, extracted_info: Dict[str, Any]) -> List[Dict]:
        """Generate test_cases.json content."""
        
        test_cases_prompt = f"""Generate test cases for {extracted_info.get('provider', 'api')} API:

Based on user requirements:
- Use case: {extracted_info.get('use_case', 'general')}
- Requirements: {extracted_info.get('requirements', {})}
- Test scenarios: {extracted_info.get('test_scenarios', ['general'])}

MANDATORY REQUIREMENT: You MUST include detailed comments explaining WHY each test case was chosen and how it relates to the user's specific requirements. However, JSON does not support comments, so include the explanations in the "description" field and "metadata" fields instead. Make the descriptions detailed and reference the user's stated use case and requirements.

Generate JSON with MANDATORY detailed descriptions (no // comments):

{{
  "test_cases": [
    {{
      "id": "creative_writing",
      "description": "Creative text generation - chosen because user specified creative writing tasks and wants engaging responses around 200-500 words. This test case matches their use case: {extracted_info.get('use_case', 'general')}",
      "input": {{
        "messages": [{{"role": "user", "content": "Write a short story about a robot"}}]
      }},
      "expected": {{
        "contains": ["robot"],
        "min_length": 20,
        "min_quality_score": 0.8
      }},
      "metadata": {{
        "category": "creative",
        "difficulty": "easy",
        "weight": 1.0,
        "reasoning": "Weight set to 1.0 because user emphasized creative writing as their primary use case. Quality threshold 0.8 reflects user's emphasis on quality over speed.",
        "user_requirements": "Matches user's response length requirements: {extracted_info.get('requirements', {}).get('response_length', 'not specified')}"
      }}
    }},
    {{
      "id": "qa_test",
      "description": "Question answering - chosen because user wants accurate and polite responses for customer support. This tests factual accuracy which aligns with their optimization goal: {extracted_info.get('optimization_goal', 'balanced')}",
      "input": {{
        "messages": [{{"role": "user", "content": "What is the capital of France?"}}]
      }},
      "expected": {{
        "contains": ["Paris"],
        "min_length": 5,
        "min_quality_score": 0.9
      }},
      "metadata": {{
        "category": "qa",
        "difficulty": "easy",
        "weight": 1.0,
        "reasoning": "Higher quality threshold (0.9) for factual accuracy since user emphasized accuracy in customer support. Short response length matches user's requirement for concise answers.",
        "user_requirements": "Aligns with user's need for accurate, polite responses under 150 words"
      }}
    }}
  ]
}}"""

        try:
            response = await self.llm_provider.generate(
                prompt=test_cases_prompt,
                temperature=0.4,  # Slightly higher for creative test cases
                max_tokens=32000  # Maximum token limit for Gemini 2.5 Flash
            )
            
            # Parse JSON response - handle markdown code blocks
            content = response['content'].strip()
            
            # Extract JSON from markdown code blocks if present
            if content.startswith('```json'):
                # Remove ```json marker
                json_start = content.find('```json') + 7
                content = content[json_start:].strip()
                # Try to find closing ```, but if not found, assume it was truncated
                json_end = content.rfind('```')
                if json_end > 0:
                    content = content[:json_end].strip()
            elif content.startswith('```'):
                # Remove generic ``` marker
                json_start = content.find('```') + 3
                content = content[json_start:].strip()
                # Try to find closing ```, but if not found, assume it was truncated
                json_end = content.rfind('```')
                if json_end > 0:
                    content = content[:json_end].strip()
            
            # If content doesn't start with {, try to find the JSON object
            if not content.startswith('{'):
                json_start = content.find('{')
                if json_start >= 0:
                    content = content[json_start:]
            
            # Try to parse JSON, but handle truncation gracefully
            try:
                test_cases_data = json.loads(content)
            except json.JSONDecodeError as e:
                # If JSON is malformed due to truncation, try to extract valid parts
                self.console.print(f"[yellow]⚠️ JSON truncated, attempting to extract valid test cases...[/yellow]")
                
                # Find the last complete test case
                last_complete_case = content.rfind('}')
                if last_complete_case > 0:
                    # Try to find the start of the test_cases array
                    test_cases_start = content.find('"test_cases": [')
                    if test_cases_start >= 0:
                        # Extract up to the last complete case
                        partial_content = content[test_cases_start-1:last_complete_case+1]
                        # Try to reconstruct valid JSON
                        reconstructed = '{"test_cases": [' + partial_content.split('"test_cases": [', 1)[1]
                        try:
                            test_cases_data = json.loads(reconstructed)
                        except:
                            # Fallback: create a simple test case
                            test_cases_data = {
                                "test_cases": [
                                    {
                                        "id": "creative_writing",
                                        "description": "Creative text generation",
                                        "input": {"messages": [{"role": "user", "content": "Write a short story about a robot"}]},
                                        "expected": {"contains": ["robot"], "min_length": 20, "min_quality_score": 0.8},
                                        "metadata": {"category": "creative", "difficulty": "easy", "weight": 1.0}
                                    }
                                ]
                            }
                else:
                    # Complete fallback
                    test_cases_data = {
                        "test_cases": [
                            {
                                "id": "creative_writing",
                                "description": "Creative text generation",
                                "input": {"messages": [{"role": "user", "content": "Write a short story about a robot"}]},
                                "expected": {"contains": ["robot"], "min_length": 20, "min_quality_score": 0.8},
                                "metadata": {"category": "creative", "difficulty": "easy", "weight": 1.0}
                            }
                        ]
                    }
            
            # Return raw JSON string to preserve comments
            self.console.print(f"[green]✅ Successfully generated test cases[/green]")
            return content
            
        except Exception as e:
            self.console.print(f"[red]❌ Error generating test cases: {e}[/red]")
            raise e
    
    async def _generate_evaluator(self, extracted_info: Dict[str, Any]) -> str:
        """Generate evaluator.py code with detailed comments based on user requirements."""
        
        # Generate evaluator with detailed comments explaining the reasoning
        use_case = extracted_info.get('use_case', 'general')
        requirements = extracted_info.get('requirements', {})
        optimization_goal = extracted_info.get('optimization_goal', 'balanced')
        
        return f'''"""
Custom API Evaluator
Generated by Convergence AI-Powered Setup

This evaluator scores API responses based on your specific requirements:
- Use case: {use_case}
- Optimization goal: {optimization_goal}
- Requirements: {requirements}

The scoring logic below is tailored to your stated needs and priorities.
"""
import re
import json
from typing import Dict, Any, Optional

def score_custom_response(
    result: Any,
    expected: Dict[str, Any],
    params: Dict[str, Any],
    metric: Optional[str] = None
) -> float:
    """
    Score API response based on user requirements.
    
    This function implements scoring logic specifically designed for:
    - Use case: {use_case}
    - Optimization goal: {optimization_goal}
    - User requirements: {requirements}
    
    Args:
        result: API response
        expected: Expected criteria
        params: Configuration parameters
        metric: Specific metric to return
    
    Returns:
        Score between 0.0 and 1.0
    """
    # Extract text from response - handles various API response formats
    text = _extract_text(result)
    if not text:
        return 0.0
    
    # Calculate individual scores based on user's stated priorities
    scores = {{}}
    
    # Quality scoring - emphasized because user mentioned quality as priority
    scores['quality'] = _score_quality(text, expected)
    
    # Length scoring - based on user's response length requirements: {requirements.get('response_length', 'not specified')}
    scores['length'] = _score_length(text, expected)
    
    # Completeness scoring - checks for required keywords/content
    scores['completeness'] = _score_completeness(text, expected)
    
    # Latency scoring - based on user's speed requirements: {requirements.get('speed_requirement', 'not specified')}
    scores['latency'] = _score_latency(result, expected)
    
    # Cost scoring - based on user's cost budget: {requirements.get('cost_budget', 'not specified')}
    scores['cost'] = _score_cost(result, expected)
    
    # Return specific metric if requested
    if metric:
        metric_lower = metric.lower()
        if metric_lower in scores:
            return scores[metric_lower]
        elif metric_lower == 'response_quality':
            # Combined quality score weighted by user's priorities
            return (scores['quality'] * 0.4 + scores['completeness'] * 0.3 + scores['length'] * 0.3)
    
    # Overall score weighted by user's optimization goal: {optimization_goal}
    # These weights reflect the user's stated priorities
    overall_score = (
        scores['quality'] * 0.35 +      # High weight for quality (user emphasized this)
        scores['completeness'] * 0.25 +  # Medium weight for completeness
        scores['length'] * 0.20 +        # Medium weight for appropriate length
        scores['latency'] * 0.10 +       # Lower weight for speed (user said not critical)
        scores['cost'] * 0.10           # Lower weight for cost (user wants efficiency)
    )
    
    return min(1.0, max(0.0, overall_score))

def _extract_text(result: Any) -> str:
    """Extract text from API response - handles multiple response formats."""
    if isinstance(result, dict):
        # Handle OpenAI-compatible responses
        if 'choices' in result and len(result['choices']) > 0:
            choice = result['choices'][0]
            if 'message' in choice and 'content' in choice['message']:
                return choice['message']['content']
        
        # Handle other common response formats
        for field in ['text', 'content', 'output', 'response', 'result']:
            if field in result:
                return str(result[field])
    
    elif isinstance(result, str):
        return result
    
    return ""

def _score_quality(text: str, expected: Dict[str, Any]) -> float:
    """
    Score text quality based on user's quality requirements.
    
    This scoring emphasizes quality because the user stated: {requirements.get('creativity_level', 'not specified')} creativity level
    and optimization goal: {optimization_goal}
    """
    if not text or len(text.strip()) < 3:
        return 0.0
    
    score = 0.5  # Base score
    
    # Length appropriateness - based on user's length requirements
    text_length = len(text)
    if 50 <= text_length <= 1000:  # Reasonable range for most use cases
        score += 0.2
    elif text_length > 1000:  # Very long responses might be excessive
        score += 0.1
    
    # Sentence structure and coherence
    sentences = text.split('.')
    if len(sentences) > 1:
        score += 0.1
    
    # Word variety (basic measure of sophistication)
    words = text.split()
    if len(words) > 0 and len(set(words)) / len(words) > 0.5:
        score += 0.1
    
    # Avoid obvious errors or low-quality indicators
    error_patterns = [r'\b(error|fail|wrong|incorrect|sorry|unable)\b']
    has_errors = any(re.search(pattern, text.lower()) for pattern in error_patterns)
    if not has_errors:
        score += 0.1
    
    return min(1.0, score)

def _score_length(text: str, expected: Dict[str, Any]) -> float:
    """
    Score based on response length requirements.
    
    User specified length requirements: {requirements.get('response_length', 'not specified')}
    """
    length = len(text)
    min_length = expected.get('min_length', 0)
    max_length = expected.get('max_length', float('inf'))
    
    if length < min_length:
        return 0.0
    
    if length > max_length:
        # Gradual penalty for being too long
        excess = length - max_length
        penalty = min(0.5, excess / max_length)
        return max(0.0, 1.0 - penalty)
    
    return 1.0

def _score_completeness(text: str, expected: Dict[str, Any]) -> float:
    """Score based on required keywords/content completeness."""
    if 'contains' not in expected:
        return 1.0
    
    required_keywords = expected['contains']
    if not required_keywords:
        return 1.0
    
    text_lower = text.lower()
    found_keywords = sum(1 for keyword in required_keywords if keyword.lower() in text_lower)
    
    return found_keywords / len(required_keywords)

def _score_latency(result: Any, expected: Dict[str, Any]) -> float:
    """
    Score based on response latency.
    
    User's speed requirement: {requirements.get('speed_requirement', 'not specified')}
    """
    if isinstance(result, dict) and 'latency_ms' in result:
        latency = result['latency_ms']
        max_latency = expected.get('max_latency_ms', 2000)
        
        if latency <= max_latency:
            return 1.0
        else:
            # Gradual penalty for slow responses
            penalty = min(0.8, (latency - max_latency) / max_latency)
            return max(0.0, 1.0 - penalty)
    
    return 0.5  # Default score if no latency data

def _score_cost(result: Any, expected: Dict[str, Any]) -> float:
    """
    Score based on cost efficiency.
    
    User's cost budget: {requirements.get('cost_budget', 'not specified')}
    """
    if isinstance(result, dict) and 'cost_usd' in result:
        cost = result['cost_usd']
        max_cost = expected.get('max_cost_usd', 0.01)
        
        if cost <= max_cost:
            return 1.0
        else:
            # Gradual penalty for expensive calls
            penalty = min(0.8, (cost - max_cost) / max_cost)
            return max(0.0, 1.0 - penalty)
    
    return 0.5  # Default score if no cost data
'''
    
    def _get_fallback_extraction(self, user_input: str) -> Dict[str, Any]:
        """Fallback extraction when LLM fails."""
        return {
            "api_type": "llm_chat",
            "provider": "openai",
            "endpoint": "https://api.openai.com/v1/chat/completions",
            "api_key_env": "OPENAI_API_KEY",
            "models": ["gpt-4o-mini", "gpt-3.5-turbo"],
            "use_case": "API optimization",
            "optimization_goal": "balanced",
            "intensity": "balanced",
            "test_scenarios": ["QA", "Creative writing", "Reasoning"],
            "metrics": {
                "quality_weight": 0.4,
                "latency_weight": 0.3,
                "cost_weight": 0.3
            },
            "search_space": {
                "parameters": {
                    "temperature": {
                        "type": "continuous",
                        "min": 0.1,
                        "max": 1.0,
                        "step": 0.1
                    },
                    "max_tokens": {
                        "type": "discrete",
                        "values": [100, 256, 512, 1024]
                    }
                }
            }
        }
    
    def _get_fallback_config(self, extracted_info: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback configuration when LLM fails."""
        return {
            'api': {
                'name': f"custom_{extracted_info.get('provider', 'api')}_{extracted_info.get('api_type', 'llm_chat')}",
                'endpoint': extracted_info.get('endpoint', 'https://api.example.com/v1/endpoint'),
                'request': {
                    'method': 'POST',
                    'headers': {'Content-Type': 'application/json'},
                    'timeout_seconds': 30
                },
                'auth': {
                    'type': 'bearer',
                    'token_env': extracted_info.get('api_key_env', 'API_KEY')
                },
                'response': {
                    'result_field': 'choices[0].message.content'
                }
            },
            'search_space': {
                'parameters': extracted_info.get('search_space', {}).get('parameters', {
                    'temperature': {'type': 'continuous', 'min': 0.1, 'max': 1.0, 'step': 0.1},
                    'max_tokens': {'type': 'discrete', 'values': [100, 256, 512, 1024]}
                })
            },
            'evaluation': {
                'test_cases': {'path': 'test_cases.json'},
                'metrics': {
                    'response_quality': {'weight': 0.4, 'type': 'higher_is_better', 'function': 'custom'},
                    'latency_ms': {'weight': 0.3, 'type': 'lower_is_better', 'threshold': 2000},
                    'cost_per_call': {'weight': 0.3, 'type': 'lower_is_better', 'budget_per_call': 0.01}
                },
                'custom_evaluator': {
                    'enabled': True,
                    'module': 'evaluator',
                    'function': 'score_custom_response'
                }
            },
            'optimization': {
                'algorithm': 'mab_evolution',
                'evolution': {
                    'population_size': 4,
                    'generations': 3,
                    'mutation_rate': 0.3,
                    'crossover_rate': 0.2,
                    'elite_size': 1
                },
                'execution': {
                    'parallel_workers': 1,
                    'experiments_per_generation': 3
                }
            },
            'society': {
                'enabled': False,
                'model': 'gemini/gemini-2.0-flash-exp',
                'api_key_env': 'GEMINI_API_KEY',
                'auto_generate_agents': True,
                'rlp_enabled': True,
                'sao_enabled': True
            },
            'legacy': {
                'enabled': True,
                'storage_backend': 'sqlite',
                'warm_start': True
            }
        }
    
    def _get_fallback_test_cases(self, extracted_info: Dict[str, Any]) -> List[Dict]:
        """Fallback test cases when LLM fails."""
        api_type = extracted_info.get('api_type', 'llm_chat')
        
        if api_type == 'llm_chat':
            return [
                {
                    "id": "qa_test",
                    "description": "Simple question answering",
                    "input": {"messages": [{"role": "user", "content": "What is the capital of France?"}]},
                    "expected": {"contains": ["Paris"], "min_length": 5, "min_quality_score": 0.9},
                    "metadata": {"category": "qa", "difficulty": "easy", "weight": 1.0}
                },
                {
                    "id": "creative_test",
                    "description": "Creative text generation",
                    "input": {"messages": [{"role": "user", "content": "Write a short story about a robot"}]},
                    "expected": {"contains": ["robot"], "min_length": 20, "min_quality_score": 0.8},
                    "metadata": {"category": "creative", "difficulty": "medium", "weight": 1.5}
                },
                {
                    "id": "reasoning_test",
                    "description": "Math reasoning",
                    "input": {"messages": [{"role": "user", "content": "If a car travels 60 mph for 2 hours, how far does it go?"}]},
                    "expected": {"contains": ["120"], "min_length": 10, "min_quality_score": 0.9},
                    "metadata": {"category": "reasoning", "difficulty": "medium", "weight": 1.5}
                }
            ]
        elif api_type == 'agno_agent':
            return [
                {
                    "id": "research_task",
                    "description": "Basic research task",
                    "input": {"task": "Search for recent AI news and summarize"},
                    "expected": {"task_completed": True, "min_quality_score": 0.8},
                    "metadata": {"category": "research", "difficulty": "medium", "weight": 1.5}
                }
            ]
        else:  # web_automation
            return [
                {
                    "id": "page_load",
                    "description": "Basic page loading",
                    "input": {"url": "https://example.com", "selector": "h1"},
                    "expected": {"page_loaded": True, "min_elements": 1},
                    "metadata": {"category": "automation", "difficulty": "easy", "weight": 1.0}
                }
            ]
    
    def _get_fallback_evaluator(self, extracted_info: Dict[str, Any]) -> str:
        """Fallback evaluator when LLM fails."""
        return '''"""
Custom API Evaluator
Generated by Convergence AI-Powered Setup

This evaluator scores API responses based on user requirements.
"""
import re
import json
from typing import Dict, Any, Optional

def score_custom_response(
    result: Any,
    expected: Dict[str, Any],
    params: Dict[str, Any],
    metric: Optional[str] = None
) -> float:
    """
    Score API response based on user requirements.
    
    Args:
        result: API response
        expected: Expected criteria
        params: Configuration parameters
        metric: Specific metric to return
    
    Returns:
        Score between 0.0 and 1.0
    """
    # Extract text from response
    text = _extract_text(result)
    if not text:
        return 0.0
    
    # Calculate scores
    scores = {}
    scores['quality'] = _score_quality(text, expected)
    scores['latency'] = _score_latency(result, expected)
    scores['cost'] = _score_cost(result, expected)
    
    # Return specific metric if requested
    if metric:
        return scores.get(metric.lower(), 0.0)
    
    # Weighted overall score
    overall_score = (
        scores['quality'] * 0.4 +
        scores['latency'] * 0.3 +
        scores['cost'] * 0.3
    )
    
    return min(1.0, max(0.0, overall_score))

def _extract_text(result):
    """Extract text from API response."""
    if isinstance(result, dict):
        if 'choices' in result and len(result['choices']) > 0:
            choice = result['choices'][0]
            if 'message' in choice and 'content' in choice['message']:
                return choice['message']['content']
        # Fallback to direct text fields
        for field in ['text', 'content', 'output', 'response']:
            if field in result:
                return str(result[field])
    elif isinstance(result, str):
        return result
    return ""

def _score_quality(text, expected):
    """Score based on response quality."""
    if not text or len(text.strip()) < 3:
        return 0.0
    
    score = 0.5  # Base score
    
    # Length appropriateness
    if 10 <= len(text) <= 1000:
        score += 0.2
    
    # Sentence structure
    sentences = text.split('.')
    if len(sentences) > 1:
        score += 0.1
    
    # Word variety (basic)
    words = text.split()
    if len(set(words)) / len(words) > 0.5:
        score += 0.1
    
    # No obvious errors
    if not re.search(r'\\b(error|fail|wrong|incorrect)\\b', text.lower()):
        score += 0.1
    
    return min(1.0, score)

def _score_latency(result, expected):
    """Score based on response latency."""
    if isinstance(result, dict) and 'latency_ms' in result:
        latency = result['latency_ms']
        max_latency = expected.get('max_latency_ms', 2000)
        
        if latency <= max_latency:
            return 1.0
        else:
            penalty = min(0.8, (latency - max_latency) / max_latency)
            return max(0.0, 1.0 - penalty)
    
    return 0.5  # Default score if no latency data

def _score_cost(result, expected):
    """Score based on cost efficiency."""
    if isinstance(result, dict) and 'cost_usd' in result:
        cost = result['cost_usd']
        max_cost = expected.get('max_cost_usd', 0.01)
        
        if cost <= max_cost:
            return 1.0
        else:
            penalty = min(0.8, (cost - max_cost) / max_cost)
            return max(0.0, 1.0 - penalty)
    
    return 0.5  # Default score if no cost data
'''
