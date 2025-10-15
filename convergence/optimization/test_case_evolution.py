"""
Test Case Evolution Engine

Applies evolutionary algorithms to test cases to automatically generate
variations and increase test coverage without manual effort.

Key Features:
- Intelligent prompt mutation (synonym replacement, paraphrasing)
- Numerical value variation
- Test case crossover (combining elements from different tests)
- Complexity scaling
- Structure-aware mutations

Inspired by genetic algorithms but adapted for test case augmentation.
"""
import random
import copy
import re
from typing import List, Dict, Any, Optional, Tuple


class TestCaseEvolutionEngine:
    """
    Evolutionary test case generator.
    
    Generates variations of existing test cases through:
    - Mutation: Smart modifications to prompts, values, and structure
    - Crossover: Combining elements from multiple test cases
    - Scaling: Adjusting complexity and difficulty
    
    Reuses evolutionary principles from config optimization but adapted
    for test case augmentation.
    """
    
    def __init__(
        self,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.2,
        augmentation_factor: int = 2,
        preserve_originals: bool = True
    ):
        """
        Initialize test case evolution engine.
        
        Args:
            mutation_rate: Probability of applying mutations (0.0-1.0)
            crossover_rate: Probability of crossover vs pure mutation (0.0-1.0)
            augmentation_factor: How many variants to generate per original test
            preserve_originals: Whether to keep original tests in the final set
        """
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.augmentation_factor = augmentation_factor
        self.preserve_originals = preserve_originals
        
        # Synonym dictionary for intelligent prompt mutations
        self.synonyms = {
            "calculate": ["compute", "determine", "find", "work out"],
            "solve": ["figure out", "work through", "resolve", "answer"],
            "explain": ["describe", "clarify", "elaborate on", "detail"],
            "analyze": ["examine", "evaluate", "assess", "investigate"],
            "compare": ["contrast", "evaluate", "differentiate", "distinguish"],
            "simple": ["basic", "straightforward", "easy", "elementary"],
            "complex": ["complicated", "intricate", "sophisticated", "advanced"],
            "problem": ["question", "task", "challenge", "exercise"],
            "step": ["stage", "phase", "part", "process"],
            "show": ["demonstrate", "display", "present", "illustrate"],
            "work": ["process", "solution", "reasoning", "approach"],
        }
        
        # Numerical scaling factors
        self.scaling_factors = [0.5, 0.75, 1.25, 1.5, 2.0]
    
    def augment_test_cases(
        self,
        original_tests: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate augmented test case set from originals.
        
        Args:
            original_tests: Original test cases
            
        Returns:
            Augmented test case list (includes originals if preserve_originals=True)
        """
        augmented = []
        
        # Always include originals if requested
        if self.preserve_originals:
            augmented.extend(copy.deepcopy(original_tests))
        
        # Generate variants for each original
        for original in original_tests:
            for variant_idx in range(self.augmentation_factor):
                if random.random() < self.crossover_rate and len(original_tests) > 1:
                    # Crossover: Combine elements from two tests
                    other = random.choice([t for t in original_tests if t != original])
                    variant = self._crossover_test_cases(original, other)
                else:
                    # Mutation: Mutate a single test
                    variant = self._mutate_test_case(original, variant_idx)
                
                augmented.append(variant)
        
        return augmented
    
    def _mutate_test_case(
        self,
        test_case: Dict[str, Any],
        variant_idx: int
    ) -> Dict[str, Any]:
        """
        Mutate a test case to create a variant.
        
        Applies multiple mutation strategies:
        - Prompt text mutations (synonyms, rephrasing)
        - Numerical value scaling
        - Structural modifications
        
        Args:
            test_case: Original test case
            variant_idx: Variant index (for deterministic ID generation)
            
        Returns:
            Mutated test case
        """
        mutated = copy.deepcopy(test_case)
        
        # Update ID to reflect variant
        if "id" in mutated:
            mutated["id"] = f"{mutated['id']}_variant_{variant_idx + 1}"
        
        # Mutate input
        if "input" in mutated:
            mutated["input"] = self._mutate_input(mutated["input"])
        
        # Adjust expected values if needed
        if "expected" in mutated:
            mutated["expected"] = self._mutate_expected(mutated["expected"])
        
        # Update metadata
        if "metadata" in mutated:
            if "description" in mutated["metadata"]:
                mutated["metadata"]["description"] += f" (variant {variant_idx + 1})"
        
        return mutated
    
    def _mutate_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mutate input data intelligently based on structure.
        
        Handles:
        - Text prompts (synonym replacement, paraphrasing)
        - Messages arrays (OpenAI/Azure format)
        - Numerical parameters
        - Nested structures
        """
        mutated = copy.deepcopy(input_data)
        
        # Handle messages format (OpenAI/Azure)
        if "messages" in mutated and isinstance(mutated["messages"], list):
            for message in mutated["messages"]:
                if "content" in message and isinstance(message["content"], str):
                    if random.random() < self.mutation_rate:
                        message["content"] = self._mutate_text(message["content"])
        
        # Handle direct prompt
        elif "prompt" in mutated and isinstance(mutated["prompt"], str):
            if random.random() < self.mutation_rate:
                mutated["prompt"] = self._mutate_text(mutated["prompt"])
        
        # Handle numerical values in input
        for key, value in mutated.items():
            if isinstance(value, (int, float)) and key not in ["id", "temperature"]:
                if random.random() < self.mutation_rate * 0.5:  # Lower rate for numbers
                    mutated[key] = self._mutate_number(value)
        
        return mutated
    
    def _mutate_text(self, text: str) -> str:
        """
        Mutate text using multiple strategies.
        
        Strategies:
        1. Synonym replacement (intelligent)
        2. Number scaling (in text)
        3. Sentence reordering
        4. Minor rephrasing
        """
        mutated = text
        
        # Strategy 1: Synonym replacement
        if random.random() < 0.6:
            mutated = self._replace_synonyms(mutated)
        
        # Strategy 2: Scale numbers in text
        if random.random() < 0.4:
            mutated = self._scale_numbers_in_text(mutated)
        
        # Strategy 3: Add/remove emphasis
        if random.random() < 0.3:
            mutated = self._adjust_emphasis(mutated)
        
        return mutated
    
    def _replace_synonyms(self, text: str) -> str:
        """Replace words with synonyms intelligently."""
        words = text.split()
        mutated_words = []
        
        for word in words:
            word_lower = word.lower().strip('.,!?;:')
            
            # Check if word has synonyms
            if word_lower in self.synonyms and random.random() < 0.3:
                # Replace with synonym
                synonym = random.choice(self.synonyms[word_lower])
                
                # Preserve capitalization
                if word[0].isupper():
                    synonym = synonym.capitalize()
                
                # Preserve punctuation
                if word[-1] in '.,!?;:':
                    synonym += word[-1]
                
                mutated_words.append(synonym)
            else:
                mutated_words.append(word)
        
        return ' '.join(mutated_words)
    
    def _scale_numbers_in_text(self, text: str) -> str:
        """Scale numerical values in text."""
        def scale_match(match):
            num = float(match.group())
            # Apply random scaling
            scale = random.choice(self.scaling_factors)
            scaled = num * scale
            
            # Keep as integer if original was integer
            if '.' not in match.group():
                return str(int(scaled))
            else:
                return f"{scaled:.2f}"
        
        # Find and scale numbers
        mutated = re.sub(r'\b\d+\.?\d*\b', scale_match, text)
        return mutated
    
    def _adjust_emphasis(self, text: str) -> str:
        """Add or remove emphasis words."""
        emphasis_additions = [
            ("Please", lambda t: f"Please {t}"),
            ("carefully", lambda t: t.replace("Show", "Carefully show").replace("Explain", "Carefully explain")),
            ("clearly", lambda t: t.replace("Show", "Clearly show").replace("Explain", "Clearly explain")),
        ]
        
        if random.random() < 0.5:
            # Add emphasis
            emphasis_type, transform = random.choice(emphasis_additions)
            return transform(text)
        else:
            # Remove existing emphasis (reverse operation)
            return text.replace("Please ", "").replace("carefully ", "").replace("clearly ", "")
    
    def _mutate_number(self, value: float) -> float:
        """Mutate a numerical value."""
        scale = random.choice(self.scaling_factors)
        mutated = value * scale
        
        # Preserve integer type if original was integer
        if isinstance(value, int):
            return int(mutated)
        else:
            return round(mutated, 4)
    
    def _mutate_expected(self, expected: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mutate expected values to match mutated inputs.
        
        Adjusts numerical answers and tolerances accordingly.
        """
        mutated = copy.deepcopy(expected)
        
        # Scale numerical answers if present
        for key in ["answer", "min_steps", "min_quality_score"]:
            if key in mutated and isinstance(mutated[key], (int, float)):
                # Be more conservative with expected value mutations
                if random.random() < 0.3:
                    if key == "answer":
                        # For answers, scale proportionally to input mutations
                        mutated[key] = self._mutate_number(mutated[key])
                    elif key == "min_quality_score":
                        # Slightly relax quality requirements for variants
                        mutated[key] = max(0.5, mutated[key] - 0.05)
        
        return mutated
    
    def _crossover_test_cases(
        self,
        test1: Dict[str, Any],
        test2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a hybrid test case from two parents.
        
        Combines elements intelligently:
        - Mixes prompt elements
        - Averages numerical parameters
        - Blends expected outcomes
        
        Args:
            test1: First parent test case
            test2: Second parent test case
            
        Returns:
            Hybrid test case
        """
        child = copy.deepcopy(test1)
        
        # Update ID
        if "id" in test1 and "id" in test2:
            child["id"] = f"crossover_{test1['id']}_{test2['id']}"
        
        # Crossover input
        if "input" in test1 and "input" in test2:
            child["input"] = self._crossover_inputs(test1["input"], test2["input"])
        
        # Average numerical expected values
        if "expected" in test1 and "expected" in test2:
            child["expected"] = self._crossover_expected(test1["expected"], test2["expected"])
        
        # Blend metadata
        if "metadata" in test1 and "metadata" in test2:
            child["metadata"] = self._crossover_metadata(test1["metadata"], test2["metadata"])
        
        return child
    
    def _crossover_inputs(
        self,
        input1: Dict[str, Any],
        input2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Crossover input data from two test cases."""
        child_input = {}
        
        # Handle messages format
        if "messages" in input1 and "messages" in input2:
            # Mix message contents
            child_messages = []
            max_len = max(len(input1["messages"]), len(input2["messages"]))
            
            for i in range(max_len):
                # Randomly pick from parent1 or parent2
                if i < len(input1["messages"]) and i < len(input2["messages"]):
                    message = random.choice([input1["messages"][i], input2["messages"][i]])
                    child_messages.append(copy.deepcopy(message))
                elif i < len(input1["messages"]):
                    child_messages.append(copy.deepcopy(input1["messages"][i]))
                else:
                    child_messages.append(copy.deepcopy(input2["messages"][i]))
            
            child_input["messages"] = child_messages
        
        # Handle prompt format
        elif "prompt" in input1 and "prompt" in input2:
            # Blend prompts by taking parts from each
            prompt1_parts = input1["prompt"].split('. ')
            prompt2_parts = input2["prompt"].split('. ')
            
            # Interleave parts
            blended_parts = []
            for i in range(max(len(prompt1_parts), len(prompt2_parts))):
                if i < len(prompt1_parts) and random.random() < 0.5:
                    blended_parts.append(prompt1_parts[i])
                if i < len(prompt2_parts) and random.random() < 0.5:
                    blended_parts.append(prompt2_parts[i])
            
            child_input["prompt"] = '. '.join(blended_parts)
        
        # Copy other fields randomly from parents
        all_keys = set(input1.keys()) | set(input2.keys())
        for key in all_keys:
            if key not in child_input:
                if key in input1 and key in input2:
                    child_input[key] = random.choice([input1[key], input2[key]])
                elif key in input1:
                    child_input[key] = input1[key]
                else:
                    child_input[key] = input2[key]
        
        return child_input
    
    def _crossover_expected(
        self,
        expected1: Dict[str, Any],
        expected2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Crossover expected outcomes."""
        child_expected = {}
        
        # Average numerical values
        all_keys = set(expected1.keys()) | set(expected2.keys())
        for key in all_keys:
            val1 = expected1.get(key)
            val2 = expected2.get(key)
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Average numerical values
                avg = (val1 + val2) / 2
                child_expected[key] = int(avg) if isinstance(val1, int) else round(avg, 4)
            elif val1 is not None and val2 is not None:
                # Randomly pick for non-numerical
                child_expected[key] = random.choice([val1, val2])
            elif val1 is not None:
                child_expected[key] = val1
            elif val2 is not None:
                child_expected[key] = val2
        
        return child_expected
    
    def _crossover_metadata(
        self,
        meta1: Dict[str, Any],
        meta2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Crossover metadata fields."""
        child_meta = {}
        
        # Combine categories
        if "category" in meta1 and "category" in meta2:
            child_meta["category"] = f"{meta1['category']}-{meta2['category']}"
        
        # Average difficulty if both present
        if "difficulty" in meta1 and "difficulty" in meta2:
            diff_map = {"easy": 1, "medium": 2, "hard": 3}
            avg_diff = (diff_map.get(meta1["difficulty"], 2) + diff_map.get(meta2["difficulty"], 2)) / 2
            child_meta["difficulty"] = {1: "easy", 2: "medium", 3: "hard"}.get(round(avg_diff), "medium")
        
        # Average weights
        if "weight" in meta1 and "weight" in meta2:
            child_meta["weight"] = (meta1["weight"] + meta2["weight"]) / 2
        
        # Blend descriptions
        if "description" in meta1 and "description" in meta2:
            child_meta["description"] = f"Hybrid: {meta1['description'][:30]}... + {meta2['description'][:30]}..."
        
        return child_meta


class TestCaseAnalyzer:
    """Analyze test case diversity and coverage."""
    
    @staticmethod
    def analyze_diversity(test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze diversity of test case set.
        
        Returns:
            Dictionary with diversity metrics
        """
        if not test_cases:
            return {"diversity": 0.0, "categories": [], "difficulties": []}
        
        # Extract categories
        categories = set()
        difficulties = set()
        
        for test in test_cases:
            if "metadata" in test:
                if "category" in test["metadata"]:
                    categories.add(test["metadata"]["category"])
                if "difficulty" in test["metadata"]:
                    difficulties.add(test["metadata"]["difficulty"])
        
        # Calculate diversity score
        diversity = len(set(str(t.get("input", "")) for t in test_cases)) / len(test_cases)
        
        return {
            "diversity": diversity,
            "total_tests": len(test_cases),
            "unique_categories": len(categories),
            "unique_difficulties": len(difficulties),
            "categories": list(categories),
            "difficulties": list(difficulties)
        }

