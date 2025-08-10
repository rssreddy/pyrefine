"""
Core modules for SELF-REFINE with Change-of-Thought framework.
"""

from .llm_clients import (
    LLMClientManager,
    OpenAIClient,
    GeminiClient,
    BaseLLMClient,
    create_system_message,
    create_human_message,
    create_ai_message,
    create_conversation
)

from .cot_capture import (
    ChangeOfThoughtAnalyzer,
    CoTCapture,
    ReasoningStep,
    create_cot_prompt
)

from .critic import (
    Critic,
    CriticFeedback,
    create_refinement_prompt
)

from .adaptive_stopping import (
    AdaptiveStoppingManager,
    StoppingDecision,
    BaseStoppingCriterion,
    ConfidenceStoppingCriterion,
    ConsistencyStoppingCriterion,
    ImprovementStoppingCriterion,
    CriticBasedStoppingCriterion,
    CompositeStoppingCriterion
)

__all__ = [
    # LLM Clients
    "LLMClientManager",
    "OpenAIClient", 
    "GeminiClient",
    "BaseLLMClient",
    "create_system_message",
    "create_human_message",
    "create_ai_message",
    "create_conversation",
    
    # CoT Capture
    "ChangeOfThoughtAnalyzer",
    "CoTCapture",
    "ReasoningStep", 
    "create_cot_prompt",
    
    # Critic
    "Critic",
    "CriticFeedback",
    "create_refinement_prompt",
    
    # Adaptive Stopping
    "AdaptiveStoppingManager",
    "StoppingDecision",
    "BaseStoppingCriterion",
    "ConfidenceStoppingCriterion",
    "ConsistencyStoppingCriterion", 
    "ImprovementStoppingCriterion",
    "CriticBasedStoppingCriterion",
    "CompositeStoppingCriterion"
]