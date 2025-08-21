"""
Critic Module for SELF-REFINE with Change-of-Thought

This module provides the critic functionality as specified in the project requirements,
generating structured feedback in JSON format.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from pyrefine.config.settings import get_config
from pyrefine.core.llm_clients import LLMClientManager


logger = logging.getLogger(__name__)


@dataclass
class CriticFeedback:
    """Structured feedback from the critic."""
    feedback: str
    category: str
    severity: str  # 'low', 'medium', 'high'
    suggestion: str
    confidence: float


class Critic:
    """
    Critic module that analyzes responses and provides structured feedback.

    As specified in the requirements, this outputs feedback in JSON format:
    {"feedback": "..."}
    """

    def __init__(self, llm_manager: LLMClientManager, config: Optional[Dict[str, Any]] = None):
        """Initialize the critic."""
        self.llm_manager = llm_manager
        if config is not None:
            self.config = config
        else:
            self.config = get_config().critic.model_dump()

        # Critic prompt template as specified in requirements
        self.critic_prompt = self._build_critic_prompt()

        logger.info("Critic module initialized")

    def _build_critic_prompt(self) -> str:
        """Build the critic prompt as specified in requirements."""
        # Use triple quotes and escape braces properly
        prompt = '''ROLE: Critic
Input:
  - Original_User_Prompt: {user_prompt}
  - Previous_Answer: {prev_answer}

Task:
  1. Point out factual errors, logic gaps, style issues.
  2. Suggest concrete edits.

You are a meticulous critic analyzing responses for improvement. Focus on:

FACTUAL ERRORS:
- Identify any incorrect facts, figures, or statements
- Check for consistency with known information
- Verify logical reasoning chains

LOGIC GAPS:
- Look for missing steps in reasoning
- Identify assumptions that need support
- Check for contradictions or inconsistencies

STYLE ISSUES:
- Assess clarity and readability
- Check for appropriate tone and structure
- Identify areas where explanation could be improved

COMPLETENESS:
- Determine if the response fully addresses the original prompt
- Identify missing information or perspectives
- Check if examples or evidence are needed

For each issue found, provide:
1. Specific description of the problem
2. Concrete suggestion for improvement
3. Assessment of severity (low/medium/high)

Output your analysis in JSON format:
{{"feedback": "detailed feedback with specific issues and suggestions"}}

Be constructive and specific in your criticism. Focus on actionable improvements.
'''
        return prompt

    async def analyze_response(
        self, 
        original_prompt: str, 
        response: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a response and provide structured feedback.

        Args:
            original_prompt: The original user prompt
            response: The response to analyze
            context: Additional context for analysis

        Returns:
            Dict containing the JSON feedback as specified in requirements
        """
        # Format the critic prompt
        formatted_prompt = self.critic_prompt.format(
            user_prompt=original_prompt,
            prev_answer=response
        )

        # Create messages for the LLM
        messages = [
            SystemMessage(content="You are an expert critic providing detailed analysis."),
            HumanMessage(content=formatted_prompt)
        ]

        try:
            # Get feedback from LLM
            feedback_text, model_used = await self.llm_manager.generate_response(
                messages=messages,
                temperature=0.3  # Lower temperature for more consistent criticism
            )

            # Try to parse as JSON
            try:
                feedback_json = json.loads(feedback_text)
                if "feedback" not in feedback_json:
                    # If not in correct format, wrap it
                    feedback_json = {"feedback": feedback_text}
            except json.JSONDecodeError:
                # If not valid JSON, wrap the text
                feedback_json = {"feedback": feedback_text}

            # Add metadata
            feedback_json["meta"] = {
                "critic_model": model_used,
                "analysis_timestamp": datetime.now().isoformat(),
                "original_prompt_length": len(original_prompt),
                "response_length": len(response),
                "context": context or {}
            }

            logger.info(f"Generated critic feedback using {model_used}")
            return feedback_json

        except Exception as e:
            logger.error(f"Error in critic analysis: {e}")
            return {
                "feedback": f"Error in analysis: {str(e)}",
                "meta": {
                    "error": True,
                    "error_message": str(e)
                }
            }

    def parse_structured_feedback(self, feedback_json: Dict[str, Any]) -> List[CriticFeedback]:
        """
        Parse structured feedback from JSON into CriticFeedback objects.

        Args:
            feedback_json: JSON feedback from the critic

        Returns:
            List of CriticFeedback objects
        """
        feedback_text = feedback_json.get("feedback", "")

        # Simple parsing - in practice, this could be more sophisticated
        # Extract different types of issues
        feedback_items = []

        # Categories from config
        categories = self.config.get("feedback_categories", [
            "factual_errors", "logic_gaps", "style_issues", "completeness"
        ])

        # Simple heuristic parsing
        for category in categories:
            if category.replace("_", " ") in feedback_text.lower():
                # Extract relevant section
                category_feedback = self._extract_category_feedback(feedback_text, category)

                if category_feedback:
                    feedback_item = CriticFeedback(
                        feedback=category_feedback,
                        category=category,
                        severity=self._assess_severity(category_feedback),
                        suggestion=self._extract_suggestion(category_feedback),
                        confidence=0.8  # Default confidence
                    )
                    feedback_items.append(feedback_item)

        # If no specific categories found, create general feedback
        if not feedback_items:
            feedback_items.append(CriticFeedback(
                feedback=feedback_text,
                category="general",
                severity="medium",
                suggestion="Review and refine the response based on the feedback.",
                confidence=0.7
            ))

        return feedback_items

    def _extract_category_feedback(self, text: str, category: str) -> str:
        """Extract feedback for a specific category."""
        # Simple extraction based on category keywords
        lines = text.split('\n')
        category_lines = []

        category_keywords = {
            "factual_errors": ["fact", "error", "incorrect", "wrong", "inaccurate"],
            "logic_gaps": ["logic", "reasoning", "gap", "inconsistent", "contradiction"],
            "style_issues": ["style", "clarity", "tone", "readability", "structure"],
            "completeness": ["complete", "missing", "address", "cover", "include"]
        }

        keywords = category_keywords.get(category, [category.replace("_", " ")])

        for line in lines:
            for keyword in keywords:
                if keyword in line.lower():
                    category_lines.append(line.strip())
                    break

        return " ".join(category_lines) if category_lines else ""

    def _assess_severity(self, feedback: str) -> str:
        """Assess the severity of feedback."""
        high_severity_words = ["major", "critical", "serious", "significant", "wrong", "error"]
        medium_severity_words = ["improve", "unclear", "confusing", "inconsistent"]

        feedback_lower = feedback.lower()

        if any(word in feedback_lower for word in high_severity_words):
            return "high"
        elif any(word in feedback_lower for word in medium_severity_words):
            return "medium"
        else:
            return "low"

    def _extract_suggestion(self, feedback: str) -> str:
        """Extract concrete suggestions from feedback."""
        # Look for suggestion keywords
        suggestion_keywords = ["suggest", "recommend", "should", "could", "try", "consider"]

        sentences = feedback.split('.')
        suggestions = []

        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in suggestion_keywords):
                suggestions.append(sentence.strip())

        return " ".join(suggestions) if suggestions else "Consider revising based on the feedback."

    def format_feedback_for_refinement(self, feedback_items: List[CriticFeedback]) -> str:
        """
        Format critic feedback for use in the refinement process.

        Args:
            feedback_items: List of CriticFeedback objects

        Returns:
            Formatted feedback string for refinement
        """
        if not feedback_items:
            return "No specific issues identified. Consider general improvements."

        formatted_parts = []

        # Group by severity
        high_severity = [f for f in feedback_items if f.severity == "high"]
        medium_severity = [f for f in feedback_items if f.severity == "medium"]  
        low_severity = [f for f in feedback_items if f.severity == "low"]

        if high_severity:
            formatted_parts.append("CRITICAL ISSUES:")
            for item in high_severity:
                formatted_parts.append(f"- {item.feedback}")
                formatted_parts.append(f"  Suggestion: {item.suggestion}")

        if medium_severity:
            formatted_parts.append("\nIMPROVEMENT AREAS:")
            for item in medium_severity:
                formatted_parts.append(f"- {item.feedback}")
                formatted_parts.append(f"  Suggestion: {item.suggestion}")

        if low_severity:
            formatted_parts.append("\nMINOR SUGGESTIONS:")
            for item in low_severity:
                formatted_parts.append(f"- {item.feedback}")

        return "\n".join(formatted_parts)

    def should_continue_refinement(self, feedback_items: List[CriticFeedback]) -> bool:
        """
        Determine if refinement should continue based on critic feedback.

        Args:
            feedback_items: List of CriticFeedback objects

        Returns:
            bool: True if refinement should continue
        """
        if not feedback_items:
            return False

        # Continue if there are high severity issues
        high_severity_count = sum(1 for item in feedback_items if item.severity == "high")
        if high_severity_count > 0:
            logger.info(f"Continuing refinement: {high_severity_count} high severity issues")
            return True

        # Continue if there are multiple medium severity issues
        medium_severity_count = sum(1 for item in feedback_items if item.severity == "medium")
        if medium_severity_count > 2:
            logger.info(f"Continuing refinement: {medium_severity_count} medium severity issues")
            return True

        # Stop if only low severity issues
        logger.info("Stopping refinement: Only minor issues remaining")
        return False


def create_refinement_prompt(original_prompt: str, previous_response: str, feedback: str) -> str:
    """
    Create a prompt for refinement based on critic feedback.

    Args:
        original_prompt: Original user prompt
        previous_response: Previous response to refine
        feedback: Critic feedback

    Returns:
        Formatted refinement prompt
    """
    return f"""Please refine the following response based on the provided feedback.

ORIGINAL PROMPT:
{original_prompt}

PREVIOUS RESPONSE:
{previous_response}

CRITIC FEEDBACK:
{feedback}

TASK:
Improve the response by addressing the feedback. Maintain the core content while fixing issues and incorporating suggestions. Be thorough and ensure your refined response fully addresses the original prompt.

REFINED RESPONSE:"""