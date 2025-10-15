"""
Built-in evaluators for The Convergence.

Custom evaluators can be placed in this directory or loaded from your project directory.
See evaluators/README.md for detailed guide on creating custom evaluators.

Available Built-in Evaluators:
    - gemini_evaluator.score_task_decomposition - Task breakdown quality
    - text_quality.score_text_quality - Text quality assessment
    - json_structure.score_json_structure - JSON validation and structure
    - json_structure.score_json_validity - Simple JSON validity check
    - code_quality.score_code_quality - Code quality evaluation
    - code_quality.score_python_syntax - Python syntax validation
"""

from .base import BaseEvaluator, score_wrapper
from .gemini_evaluator import score_task_decomposition
from .text_quality import score_text_quality
from .json_structure import score_json_structure, score_json_validity
from .code_quality import score_code_quality, score_python_syntax
from .openai_responses import score_openai_response, score_reasoning_response

__all__ = [
    # Base classes
    'BaseEvaluator',
    'score_wrapper',
    
    # LLM-specific evaluators
    'score_task_decomposition',
    'score_openai_response',
    'score_reasoning_response',
    
    # General purpose evaluators
    'score_text_quality',
    'score_json_structure',
    'score_json_validity',
    'score_code_quality',
    'score_python_syntax',
]

