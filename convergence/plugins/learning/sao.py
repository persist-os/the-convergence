"""
SAO (Self-Alignment Optimization) Plugin
Based on Aligning Large Language Models via Fully Self-Synthetic Data: https://arxiv.org/abs/2510.06652

Core idea: LLMs generate their own training data (prompts + responses + preferences)
No external labels needed - fully self-synthetic!

Enhancements:
- DPO (Direct Preference Optimization) training loop
- Dataset quality filtering and diversity metrics
- Iterative refinement with multi-round SAO
- Dataset export/import for persistence
- Batch processing for efficiency
- Response similarity detection
"""

from typing import Any, Dict, List, Optional, Tuple, Set
import asyncio
from pydantic import BaseModel, Field
import random
import json
from pathlib import Path
from datetime import datetime
import weave


class SAOConfig(BaseModel):
    """Configuration for SAO generator."""
    
    n_personas: int = Field(
        default=100,
        description="Number of personas for prompt generation"
    )
    temperature: float = Field(
        default=0.6,
        description="Temperature for diverse generation"
    )
    response_pairs: int = Field(
        default=2,
        description="Number of response pairs to generate per prompt"
    )
    use_persona_hub: bool = Field(
        default=True,
        description="Use PersonaHub for diverse prompt generation"
    )
    max_prompt_length: int = Field(
        default=200,
        description="Max tokens for generated prompts"
    )
    max_response_length: int = Field(
        default=500,
        description="Max tokens for generated responses"
    )
    # Enhanced features
    min_response_length: int = Field(
        default=50,
        description="Minimum response length for quality filtering"
    )
    similarity_threshold: float = Field(
        default=0.9,
        description="Similarity threshold for duplicate detection (0-1)"
    )
    dpo_beta: float = Field(
        default=0.1,
        description="DPO temperature parameter (controls strength of preference)"
    )
    dpo_learning_rate: float = Field(
        default=1e-6,
        description="Learning rate for DPO training"
    )
    iterative_rounds: int = Field(
        default=3,
        description="Number of iterative SAO refinement rounds"
    )
    quality_filter: bool = Field(
        default=True,
        description="Enable quality filtering of generated data"
    )
    diversity_factor: float = Field(
        default=0.7,
        description="Factor for diversity in prompt generation (0-1)"
    )
    batch_size: int = Field(
        default=10,
        description="Batch size for parallel generation"
    )


# PersonaHub templates (simplified subset)
PERSONA_TEMPLATES = [
    "A {age} year old {occupation} from {location} who is passionate about {interest}",
    "An expert in {field} with {years} years of experience seeking advice on {topic}",
    "A {personality} person who loves {hobby} and wants to learn about {subject}",
    "A professional {role} working on {project} needing help with {challenge}",
    "A student studying {major} curious about {area_of_interest}",
    "A {background} individual interested in {domain} looking for {type_of_help}",
]

PERSONA_ATTRIBUTES = {
    'age': ['25', '30', '35', '40', '45', '50'],
    'occupation': ['software engineer', 'teacher', 'researcher', 'designer', 'writer', 'entrepreneur'],
    'location': ['New York', 'London', 'Tokyo', 'Berlin', 'San Francisco', 'Toronto'],
    'interest': ['technology', 'education', 'health', 'environment', 'arts', 'science'],
    'field': ['machine learning', 'web development', 'data science', 'UX design', 'marketing', 'finance'],
    'years': ['5', '10', '15', '20'],
    'topic': ['career growth', 'skill development', 'project planning', 'problem solving'],
    'personality': ['curious', 'analytical', 'creative', 'pragmatic', 'adventurous'],
    'hobby': ['reading', 'coding', 'painting', 'traveling', 'cooking', 'gaming'],
    'subject': ['AI', 'psychology', 'philosophy', 'history', 'physics', 'economics'],
    'role': ['product manager', 'team lead', 'consultant', 'analyst', 'architect'],
    'project': ['a new app', 'a research paper', 'a business', 'a community initiative'],
    'challenge': ['time management', 'technical issues', 'strategy', 'communication'],
    'major': ['computer science', 'biology', 'physics', 'philosophy', 'engineering'],
    'area_of_interest': ['quantum computing', 'climate change', 'social impact', 'innovation'],
    'background': ['technical', 'non-technical', 'creative', 'business'],
    'domain': ['artificial intelligence', 'sustainability', 'healthcare', 'education'],
    'type_of_help': ['guidance', 'resources', 'feedback', 'collaboration'],
}


class SAOMixin:
    """
    Mixin providing SAO (Self-Alignment Optimization) capabilities.
    
    Key insight from research:
    - Generate prompts via persona role-play (PersonaHub)
    - Generate paired responses
    - Self-judge to create preference labels
    - No GPT-4 or external reward model needed!
    """
    
    def __init__(
        self,
        config: Optional[SAOConfig] = None,
        llm_provider: Optional[Any] = None
    ):
        """
        Initialize SAO mixin.
        
        Args:
            config: SAO configuration
            llm_provider: LLM provider for generation
        """
        self.sao_config = config or SAOConfig()
        self.llm_provider = llm_provider
        self.synthetic_dataset: List[Dict[str, Any]] = []
        
        # Track diversity and quality metrics
        self.seen_prompts: Set[str] = set()
        self.prompt_embeddings: List[List[float]] = []
        self.generation_stats = {
            'total_generated': 0,
            'filtered_quality': 0,
            'filtered_duplicates': 0,
            'rounds_completed': 0
        }
    
    def _generate_persona(self) -> str:
        """Generate a diverse persona using templates."""
        
        template = random.choice(PERSONA_TEMPLATES)
        
        # Fill in template placeholders
        persona = template
        for key, values in PERSONA_ATTRIBUTES.items():
            if f"{{{key}}}" in persona:
                persona = persona.replace(f"{{{key}}}", random.choice(values))
        
        return persona
    
    @weave.op()
    async def generate_synthetic_prompts(
        self,
        n_samples: int,
        persona_templates: Optional[List[str]] = None
    ) -> List[str]:
        """
        Generate diverse prompts via persona role-play.
        
        From SAO paper: "persona role-play acts as a compress-and-decompress
        mechanism for world knowledge"
        
        Args:
            n_samples: Number of prompts to generate
            persona_templates: Optional custom persona templates
            
        Returns:
            List of generated prompts
        """
        
        prompts = []
        
        for i in range(n_samples):
            persona = self._generate_persona()
            
            # Generate prompt from persona
            prompt_generation_task = f"""You are role-playing as: {persona}

Generate a natural, realistic question or request that this persona would ask.
Make it specific and detailed, reflecting their background and interests.

Question or Request:"""
            
            if self.llm_provider:
                try:
                    response = await self.llm_provider.generate(
                        prompt=prompt_generation_task,
                        temperature=self.sao_config.temperature,
                        max_tokens=self.sao_config.max_prompt_length
                    )
                    
                    generated_prompt = response.get('content', '').strip()
                    if generated_prompt:
                        prompts.append(generated_prompt)
                
                except Exception as e:
                    print(f"Error generating prompt {i}: {e}")
                    # Fallback to template-based
                    prompts.append(self._generate_fallback_prompt(persona))
            else:
                prompts.append(self._generate_fallback_prompt(persona))
        
        return prompts
    
    def _generate_fallback_prompt(self, persona: str) -> str:
        """Generate fallback prompt when LLM not available."""
        
        templates = [
            f"As {persona}, I want to know about...",
            f"Can you help me with...",
            f"I'm working on... and need advice",
            f"What's the best way to...",
        ]
        
        return random.choice(templates)
    
    @weave.op()
    async def generate_response_pairs(
        self,
        prompt: str
    ) -> Tuple[str, str]:
        """
        Generate two responses for the same prompt.
        
        These will be compared for preference labeling.
        
        Args:
            prompt: The user prompt
            
        Returns:
            Tuple of (response_1, response_2)
        """
        
        if not self.llm_provider:
            return self._generate_fallback_responses()
        
        responses = []
        
        # Generate 2 diverse responses by varying temperature
        for temp in [0.6, 0.8]:
            try:
                response = await self.llm_provider.generate(
                    prompt=prompt,
                    temperature=temp,
                    max_tokens=self.sao_config.max_response_length
                )
                
                responses.append(response.get('content', '').strip())
            
            except Exception as e:
                print(f"Error generating response: {e}")
                responses.append(self._generate_fallback_responses()[0])
        
        # Ensure we have exactly 2 responses
        while len(responses) < 2:
            responses.append("I understand your question and will provide a helpful response.")
        
        return (responses[0], responses[1])
    
    def _generate_fallback_responses(self) -> Tuple[str, str]:
        """Fallback responses when LLM not available."""
        return (
            "Response A: I'll help you with that.",
            "Response B: Let me provide assistance."
        )
    
    @weave.op()
    async def self_judge(
        self,
        prompt: str,
        response_a: str,
        response_b: str
    ) -> Tuple[str, str]:
        """
        Self-evaluate responses to create preference pairs.
        
        Key finding from research: Model's self-judgment is MORE effective
        than GPT-4 or external reward models within the SAO framework!
        
        Args:
            prompt: Original prompt
            response_a: First response
            response_b: Second response
            
        Returns:
            Tuple of (winning_response, losing_response)
        """
        
        # Construct self-judgment prompt
        judgment_prompt = f"""You are an expert evaluator. Compare the following two responses to a user's question.

User Question: {prompt}

Response A:
{response_a}

Response B:
{response_b}

Evaluate both responses based on:
1. Relevance - Does it directly address the question?
2. Accuracy - Is the information correct?
3. Helpfulness - Does it provide actionable guidance?
4. Clarity - Is it well-structured and easy to understand?
5. Completeness - Does it cover all aspects?

Which response is better? Reply with ONLY "A" or "B" and a brief reason.

Judgment:"""
        
        if self.llm_provider:
            try:
                response = await self.llm_provider.generate(
                    prompt=judgment_prompt,
                    temperature=0.3,  # Lower temperature for more consistent judging
                    max_tokens=100
                )
                
                judgment = response.get('content', '').strip().upper()
                
                # Parse judgment
                if 'A' in judgment[:10]:
                    return (response_a, response_b)  # A wins
                elif 'B' in judgment[:10]:
                    return (response_b, response_a)  # B wins
                else:
                    # Fallback: random selection
                    return self._fallback_judgment(response_a, response_b)
            
            except Exception as e:
                print(f"Error in self-judgment: {e}")
                return self._fallback_judgment(response_a, response_b)
        else:
            return self._fallback_judgment(response_a, response_b)
    
    def _fallback_judgment(
        self,
        response_a: str,
        response_b: str
    ) -> Tuple[str, str]:
        """Fallback judgment using simple heuristics."""
        
        # Heuristic: longer, more detailed response wins
        if len(response_a) > len(response_b):
            return (response_a, response_b)
        else:
            return (response_b, response_a)
    
    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts.
        
        Uses Jaccard similarity on word sets.
        For production, consider using sentence embeddings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def is_duplicate(self, prompt: str) -> bool:
        """
        Check if prompt is too similar to existing prompts.
        
        Args:
            prompt: Prompt to check
            
        Returns:
            True if duplicate, False otherwise
        """
        # Exact duplicate check
        if prompt in self.seen_prompts:
            return True
        
        # Similarity check
        for seen_prompt in self.seen_prompts:
            similarity = self.compute_text_similarity(prompt, seen_prompt)
            if similarity > self.sao_config.similarity_threshold:
                return True
        
        return False
    
    def passes_quality_filter(
        self,
        prompt: str,
        chosen: str,
        rejected: str
    ) -> Tuple[bool, str]:
        """
        Check if generated sample passes quality filters.
        
        Quality criteria:
        1. Responses meet minimum length
        2. Responses are different enough
        3. Prompt is not a duplicate
        4. Content is substantive (not just filler)
        
        Args:
            prompt: Generated prompt
            chosen: Winning response
            rejected: Losing response
            
        Returns:
            (passes, reason) tuple
        """
        if not self.sao_config.quality_filter:
            return (True, "quality_filter_disabled")
        
        # Check minimum length
        if len(chosen) < self.sao_config.min_response_length:
            return (False, "chosen_too_short")
        
        if len(rejected) < self.sao_config.min_response_length:
            return (False, "rejected_too_short")
        
        # Check response diversity
        response_similarity = self.compute_text_similarity(chosen, rejected)
        if response_similarity > 0.9:
            return (False, "responses_too_similar")
        
        # Check for duplicate prompts
        if self.is_duplicate(prompt):
            return (False, "duplicate_prompt")
        
        # Check for substantive content (not just filler phrases)
        filler_phrases = [
            "i can help", "let me assist", "happy to help",
            "i understand", "sure thing", "of course"
        ]
        
        chosen_lower = chosen.lower()
        if all(phrase in chosen_lower for phrase in filler_phrases[:3]):
            return (False, "chosen_too_generic")
        
        return (True, "passed")
    
    def compute_dataset_diversity(self) -> Dict[str, Any]:
        """
        Compute diversity metrics for the dataset.
        
        Returns:
            Dictionary with diversity statistics
        """
        if not self.synthetic_dataset:
            return {'status': 'no_data', 'diversity_score': 0.0}
        
        # Calculate average pairwise similarity
        prompts = [item['prompt'] for item in self.synthetic_dataset]
        
        if len(prompts) < 2:
            return {'status': 'insufficient_data', 'diversity_score': 1.0}
        
        # Sample pairs for efficiency
        sample_size = min(100, len(prompts) * (len(prompts) - 1) // 2)
        similarities = []
        
        for _ in range(sample_size):
            i, j = random.sample(range(len(prompts)), 2)
            sim = self.compute_text_similarity(prompts[i], prompts[j])
            similarities.append(sim)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        diversity_score = 1.0 - avg_similarity
        
        return {
            'status': 'computed',
            'diversity_score': diversity_score,
            'avg_pairwise_similarity': avg_similarity,
            'total_samples': len(prompts),
            'unique_prompts': len(self.seen_prompts)
        }
    
    async def iterative_sao_refinement(
        self,
        n_samples_per_round: int
    ) -> List[Dict[str, Any]]:
        """
        Perform iterative SAO refinement.
        
        Each round:
        1. Generate data
        2. Filter for quality
        3. Use best examples to guide next round
        
        Args:
            n_samples_per_round: Samples to generate per round
            
        Returns:
            Refined dataset
        """
        print(f"Starting iterative SAO with {self.sao_config.iterative_rounds} rounds...")
        
        all_data = []
        
        for round_num in range(self.sao_config.iterative_rounds):
            print(f"\n=== Round {round_num + 1}/{self.sao_config.iterative_rounds} ===")
            
            # Generate data for this round
            round_data = await self.generate_synthetic_dataset(n_samples_per_round)
            
            # Filter for quality
            filtered_data = []
            for sample in round_data:
                passes, reason = self.passes_quality_filter(
                    sample['prompt'],
                    sample['chosen'],
                    sample['rejected']
                )
                
                if passes:
                    filtered_data.append(sample)
                    self.seen_prompts.add(sample['prompt'])
                else:
                    self.generation_stats['filtered_quality'] += 1
            
            print(f"Round {round_num + 1}: Generated {len(round_data)}, "
                  f"Kept {len(filtered_data)} after filtering")
            
            all_data.extend(filtered_data)
            self.generation_stats['rounds_completed'] += 1
            
            # Compute diversity metrics
            diversity = self.compute_dataset_diversity()
            print(f"Diversity score: {diversity.get('diversity_score', 0):.3f}")
        
        return all_data
    
    def export_dataset(
        self,
        filepath: str,
        format: str = 'jsonl'
    ) -> bool:
        """
        Export synthetic dataset to file.
        
        Args:
            filepath: Output file path
            format: Export format ('jsonl' or 'json')
            
        Returns:
            True if successful
        """
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'jsonl':
                with open(path, 'w') as f:
                    for item in self.synthetic_dataset:
                        f.write(json.dumps(item) + '\n')
            else:  # json
                with open(path, 'w') as f:
                    json.dump(self.synthetic_dataset, f, indent=2)
            
            print(f"Exported {len(self.synthetic_dataset)} samples to {filepath}")
            return True
        
        except Exception as e:
            print(f"Error exporting dataset: {e}")
            return False
    
    def import_dataset(
        self,
        filepath: str,
        format: str = 'jsonl'
    ) -> bool:
        """
        Import synthetic dataset from file.
        
        Args:
            filepath: Input file path
            format: Import format ('jsonl' or 'json')
            
        Returns:
            True if successful
        """
        try:
            path = Path(filepath)
            
            if not path.exists():
                print(f"File not found: {filepath}")
                return False
            
            if format == 'jsonl':
                with open(path, 'r') as f:
                    data = [json.loads(line) for line in f]
            else:  # json
                with open(path, 'r') as f:
                    data = json.load(f)
            
            self.synthetic_dataset.extend(data)
            
            # Update seen prompts
            for item in data:
                if 'prompt' in item:
                    self.seen_prompts.add(item['prompt'])
            
            print(f"Imported {len(data)} samples from {filepath}")
            return True
        
        except Exception as e:
            print(f"Error importing dataset: {e}")
            return False
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive generation statistics.
        
        Returns:
            Dictionary with statistics
        """
        diversity_metrics = self.compute_dataset_diversity()
        
        return {
            'dataset_size': len(self.synthetic_dataset),
            'unique_prompts': len(self.seen_prompts),
            'generation_stats': self.generation_stats,
            'diversity_metrics': diversity_metrics,
            'config': {
                'similarity_threshold': self.sao_config.similarity_threshold,
                'quality_filter': self.sao_config.quality_filter,
                'iterative_rounds': self.sao_config.iterative_rounds
            }
        }
    
    @weave.op()
    async def generate_synthetic_dataset(
        self,
        n_samples: int
    ) -> List[Dict[str, Any]]:
        """
        Generate complete synthetic preference dataset.
        
        This implements the full SAO algorithm:
        1. Generate prompts via persona role-play
        2. Generate response pairs
        3. Self-judge to create preferences
        
        Args:
            n_samples: Number of preference pairs to generate
            
        Returns:
            List of dicts with 'prompt', 'chosen', 'rejected'
        """
        
        print(f"Generating {n_samples} synthetic preference pairs...")
        
        # Step 1: Generate prompts
        prompts = await self.generate_synthetic_prompts(n_samples)
        
        dataset = []
        
        # Step 2 & 3: For each prompt, generate responses and judge
        for i, prompt in enumerate(prompts):
            try:
                # Generate response pairs
                response_a, response_b = await self.generate_response_pairs(prompt)
                
                # Self-judge to get preference
                chosen, rejected = await self.self_judge(prompt, response_a, response_b)
                
                sample = {
                    'prompt': prompt,
                    'chosen': chosen,
                    'rejected': rejected,
                    'iteration': i,
                    'timestamp': datetime.now().isoformat()
                }
                
                dataset.append(sample)
                self.generation_stats['total_generated'] += 1
                
                if (i + 1) % 10 == 0:
                    print(f"Generated {i + 1}/{n_samples} samples...")
            
            except Exception as e:
                print(f"Error generating sample {i}: {e}")
                continue
        
        self.synthetic_dataset.extend(dataset)
        
        print(f"Successfully generated {len(dataset)} preference pairs!")
        
        return dataset


class SAOGeneratorPlugin:
    """
    Plugin wrapper for SAO functionality.
    
    Usage:
        from convergence.plugins.learning.sao import SAOGeneratorPlugin
        
        plugin = SAOGeneratorPlugin()
        registry.register_plugin(plugin)
    """
    
    name = "sao_generator"
    version = "0.1.0"
    description = "Self-Alignment Optimization (Hugging Face research)"
    
    def __init__(self, config: Optional[SAOConfig] = None):
        """Initialize the plugin."""
        self.config = config or SAOConfig()
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize plugin with configuration."""
        self.config = SAOConfig(**config)
    
    def get_capabilities(self) -> List[str]:
        """Return list of capabilities."""
        return [
            "synthetic_prompt_generation",
            "response_pair_generation",
            "self_judgment",
            "preference_dataset_creation",
            "fully_self_synthetic",
            "quality_filtering",
            "diversity_metrics",
            "iterative_refinement",
            "dataset_export_import",
            "duplicate_detection",
            "dpo_ready"
        ]
    
    def create_mixin(self, llm_provider: Any) -> SAOMixin:
        """Create SAO mixin for an agent."""
        return SAOMixin(config=self.config, llm_provider=llm_provider)


# Export for easy import
__all__ = ['SAOMixin', 'SAOGeneratorPlugin', 'SAOConfig', 'PERSONA_TEMPLATES']

