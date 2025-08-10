"""
PyRefine Framework

A Python implementation of the SELF-REFINE iterative refinement engine enhanced 
with Change-of-Thought (CoT) capture, built using LangGraph for orchestration 
and supporting both OpenAI ChatGPT and Google Gemini models.
"""

__version__ = "1.0.0"
__author__ = "Siva Sandeep Reddy"
__email__ = "sivasandeep@example.com"

# Core imports for easy access
from .graph.self_refine_graph import SelfRefineGraph, run_self_refine
from .core.llm_clients import LLMClientManager
from .core.cot_capture import ChangeOfThoughtAnalyzer
from .core.critic import Critic
from .core.adaptive_stopping import AdaptiveStoppingManager
from .config.settings import get_config, set_config, Config

# Main classes for public API
__all__ = [
    "SelfRefineGraph",
    "run_self_refine", 
    "LLMClientManager",
    "ChangeOfThoughtAnalyzer",
    "Critic",
    "AdaptiveStoppingManager",
    "get_config",
    "set_config",
    "Config"
]