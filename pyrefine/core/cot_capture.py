"""
Change-of-Thought (CoT) Capture Module with Enhanced Logging

This module replaces the original "Performance Monitoring" from SELF-REFINE
with a Change-of-Thought mechanism that tracks reasoning evolution.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import re

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from pyrefine.config.settings import get_config


logger = logging.getLogger(__name__)


@dataclass
class ReasoningStep:
    """Represents a single reasoning step in the CoT."""
    step_number: int
    content: str
    confidence: float
    timestamp: datetime
    reasoning_type: str  # 'analysis', 'synthesis', 'evaluation', 'generation'


@dataclass
class CoTCapture:
    """Captures the change of thought throughout iterations."""
    iteration: int
    reasoning_steps: List[ReasoningStep]
    overall_confidence: float
    reasoning_coherence: float
    thought_changes: List[str]
    metadata: Dict[str, Any]


class ChangeOfThoughtAnalyzer:
    """
    Analyzes and captures changes in reasoning between iterations.

    This is the core replacement for Performance Monitoring in the original
    SELF-REFINE paper, focusing on thought evolution rather than just metrics.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize CoT analyzer."""
        if config is not None:
            self.config = config
        else:
            self.config = get_config().self_refine.cot_capture.model_dump()
        self.previous_cot: Optional[CoTCapture] = None
        self.cot_history: List[CoTCapture] = []

        # CoT extraction patterns
        self.reasoning_patterns = {
            'step_indicators': [
                r'(?:step|stage|phase)\s*(\d+)',
                r'(?:first|second|third|fourth|fifth|next|then|finally)',
                r'(?:\d+\.|•|-)\s*(.+?)(?=\n|\d+\.|•|-|$)'
            ],
            'confidence_indicators': [
                r'(?:confident|certain|sure|likely|probably|possibly|uncertain)',
                r'(?:high|medium|low)\s*confidence',
                r'(?:\d+)%\s*(?:confident|certain|sure)'
            ],
            'reasoning_types': {
                'analysis': r'(?:analyze|examine|investigate|study|consider)',
                'synthesis': r'(?:combine|integrate|synthesize|merge|unify)',
                'evaluation': r'(?:evaluate|assess|judge|critique|review)',
                'generation': r'(?:create|generate|produce|develop|build)'
            }
        }


    def extract_reasoning_steps(self, text: str, iteration: int) -> List[ReasoningStep]:
        """Extract individual reasoning steps from text."""
        steps = []

        # Split text into potential steps
        lines = text.split('\n')
        step_num = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this line contains reasoning
            if self._is_reasoning_line(line):
                step_num += 1

                # Extract confidence
                confidence = self._extract_confidence(line)

                # Determine reasoning type
                reasoning_type = self._determine_reasoning_type(line)

                step = ReasoningStep(
                    step_number=step_num,
                    content=line,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    reasoning_type=reasoning_type
                )
                steps.append(step)

        return steps

    def _is_reasoning_line(self, line: str) -> bool:
        """Check if a line contains reasoning content."""
        # Skip very short lines
        if len(line) < 10:
            return False

        # Check for reasoning indicators
        for pattern in self.reasoning_patterns['step_indicators']:
            if re.search(pattern, line, re.IGNORECASE):
                return True

        # Check for logical connectors
        logical_connectors = [
            'because', 'therefore', 'thus', 'hence', 'consequently',
            'however', 'although', 'while', 'whereas', 'given that',
            'since', 'as a result', 'in conclusion', 'furthermore'
        ]

        for connector in logical_connectors:
            if connector in line.lower():
                return True

        # If the line contains keywords associated with reasoning types,
        # it should still be considered a reasoning line even if it lacks
        # explicit step indicators or connectors. Previously these lines
        # were ignored which meant valid reasoning such as "I will analyze"
        # or "a synthesis approach" was not captured, leading to missing
        # thought change detection between iterations.
        for pattern in self.reasoning_patterns['reasoning_types'].values():
            if re.search(pattern, line, re.IGNORECASE):
                return True

        return False

    def _extract_confidence(self, text: str) -> float:
        """Extract confidence level from text."""
        # Look for explicit confidence indicators
        for pattern in self.reasoning_patterns['confidence_indicators']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Map confidence words to scores
                confidence_map = {
                    'certain': 0.95, 'confident': 0.9, 'sure': 0.85,
                    'likely': 0.7, 'probably': 0.65, 'possibly': 0.4,
                    'uncertain': 0.3, 'high': 0.9, 'medium': 0.6, 'low': 0.3
                }

                matched_text = match.group().lower()
                for word, score in confidence_map.items():
                    if word in matched_text:
                        return score

        # Default confidence based on text characteristics
        if '?' in text:
            return 0.4  # Questions indicate uncertainty
        elif '!' in text:
            return 0.8  # Exclamations indicate confidence
        else:
            return 0.6  # Neutral confidence

    def _determine_reasoning_type(self, text: str) -> str:
        """Determine the type of reasoning in the text."""
        for reasoning_type, pattern in self.reasoning_patterns['reasoning_types'].items():
            if re.search(pattern, text, re.IGNORECASE):
                return reasoning_type
        return 'analysis'  # Default

    def _calculate_coherence(self, steps: List[ReasoningStep]) -> float:
        """Calculate coherence score for reasoning steps."""
        if len(steps) < 2:
            return 1.0

        # Simple coherence metric based on step transitions
        coherence_score = 0.0

        for i in range(len(steps) - 1):
            current_step = steps[i]
            next_step = steps[i + 1]

            # Check for logical flow
            if self._steps_are_coherent(current_step, next_step):
                coherence_score += 1.0
            else:
                coherence_score += 0.5

        return coherence_score / (len(steps) - 1)

    def _steps_are_coherent(self, step1: ReasoningStep, step2: ReasoningStep) -> bool:
        """Check if two reasoning steps are coherent."""
        # Simple heuristic: check if step types follow logical order
        type_order = ['analysis', 'synthesis', 'evaluation', 'generation']

        try:
            idx1 = type_order.index(step1.reasoning_type)
            idx2 = type_order.index(step2.reasoning_type)
            return idx2 >= idx1  # Allow same or forward progression
        except ValueError:
            return True  # Unknown types, assume coherent

    def capture_cot(self, response_text: str, iteration: int) -> CoTCapture:
        """
        Capture change of thought from a response.

        Args:
            response_text: The LLM's response text
            iteration: Current iteration number

        Returns:
            CoTCapture object with extracted reasoning information
        """
        # Extract reasoning steps
        reasoning_steps = self.extract_reasoning_steps(response_text, iteration)

        # Calculate overall confidence
        if reasoning_steps:
            overall_confidence = sum(step.confidence for step in reasoning_steps) / len(reasoning_steps)
        else:
            overall_confidence = 0.5

        # Calculate coherence
        coherence = self._calculate_coherence(reasoning_steps)

        # Identify thought changes from previous iteration
        thought_changes = self._identify_thought_changes(reasoning_steps)

        # Create CoT capture
        cot_capture = CoTCapture(
            iteration=iteration,
            reasoning_steps=reasoning_steps,
            overall_confidence=overall_confidence,
            reasoning_coherence=coherence,
            thought_changes=thought_changes,
            metadata={
                'response_length': len(response_text),
                'num_reasoning_steps': len(reasoning_steps),
                'capture_timestamp': datetime.now().isoformat(),
                'reasoning_types': [step.reasoning_type for step in reasoning_steps]
            }
        )

        # Store for comparison
        self.cot_history.append(cot_capture)
        self.previous_cot = cot_capture

        # Enhanced logging for CoT details
        logger.info(
            f"🧠 CoT CAPTURED for iteration {iteration}: "
            f"{len(reasoning_steps)} reasoning steps, "
            f"confidence={overall_confidence:.3f}, "
            f"coherence={coherence:.3f}"
        )
        
        # Log individual reasoning steps
        if reasoning_steps:
            logger.info(f"📝 REASONING STEPS (Iteration {iteration}):")
            for i, step in enumerate(reasoning_steps[:5]):  # Show first 5 steps
                logger.info(
                    f"   Step {step.step_number}: [{step.reasoning_type}] "
                    f"(conf: {step.confidence:.2f}) {step.content[:100]}..."
                )
            if len(reasoning_steps) > 5:
                logger.info(f"   ... and {len(reasoning_steps) - 5} more steps")
        
        # Log thought changes
        if thought_changes:
            logger.info(f"🔄 THOUGHT CHANGES (Iteration {iteration}):")
            for change in thought_changes:
                logger.info(f"   • {change}")
        else:
            logger.info(f"🔄 No significant thought changes detected (Iteration {iteration})")
        
        # Log reasoning type distribution
        if reasoning_steps:
            type_counts = {}
            for step in reasoning_steps:
                type_counts[step.reasoning_type] = type_counts.get(step.reasoning_type, 0) + 1
            
            type_summary = ", ".join([f"{rtype}: {count}" for rtype, count in type_counts.items()])
            logger.info(f"🎯 REASONING TYPES (Iteration {iteration}): {type_summary}")

        return cot_capture

    def _identify_thought_changes(self, current_steps: List[ReasoningStep]) -> List[str]:
        """Identify changes in thought from previous iteration."""
        if not self.previous_cot:
            return []

        changes = []
        prev_steps = self.previous_cot.reasoning_steps

        # Compare reasoning types
        current_types = [step.reasoning_type for step in current_steps]
        previous_types = [step.reasoning_type for step in prev_steps]

        if current_types != previous_types:
            changes.append(f"Reasoning approach changed from {previous_types} to {current_types}")

        # Compare confidence levels
        current_avg_confidence = sum(step.confidence for step in current_steps) / max(len(current_steps), 1)
        previous_avg_confidence = sum(step.confidence for step in prev_steps) / max(len(prev_steps), 1)

        confidence_diff = current_avg_confidence - previous_avg_confidence
        if abs(confidence_diff) > 0.1:
            direction = "increased" if confidence_diff > 0 else "decreased"
            changes.append(f"Overall confidence {direction} by {abs(confidence_diff):.2f}")

        # Compare number of steps
        if len(current_steps) != len(prev_steps):
            changes.append(f"Reasoning complexity changed: {len(prev_steps)} → {len(current_steps)} steps")

        return changes

    def get_cot_analysis(self) -> Dict[str, Any]:
        """Get comprehensive CoT analysis across all iterations."""
        if not self.cot_history:
            return {}

        analysis = {
            'total_iterations': len(self.cot_history),
            'confidence_progression': [cot.overall_confidence for cot in self.cot_history],
            'coherence_progression': [cot.reasoning_coherence for cot in self.cot_history],
            'reasoning_evolution': [],
            'thought_changes_summary': []
        }

        # Analyze reasoning evolution
        for i, cot in enumerate(self.cot_history):
            analysis['reasoning_evolution'].append({
                'iteration': i,
                'num_steps': len(cot.reasoning_steps),
                'dominant_reasoning_type': self._get_dominant_reasoning_type(cot.reasoning_steps),
                'confidence': cot.overall_confidence,
                'coherence': cot.reasoning_coherence
            })

            if cot.thought_changes:
                analysis['thought_changes_summary'].extend(cot.thought_changes)

        return analysis

    def _get_dominant_reasoning_type(self, steps: List[ReasoningStep]) -> str:
        """Get the most common reasoning type in the steps."""
        if not steps:
            return 'none'

        type_counts = {}
        for step in steps:
            type_counts[step.reasoning_type] = type_counts.get(step.reasoning_type, 0) + 1

        return max(type_counts.items(), key=lambda x: x[1])[0]

    def should_continue_based_on_cot(self) -> bool:
        """
        Determine if refinement should continue based on CoT analysis.

        Returns:
            bool: True if refinement should continue, False otherwise
        """
        if not self.cot_history:
            return True

        current_cot = self.cot_history[-1]

        # Continue if confidence is low
        if current_cot.overall_confidence < self.config['confidence_threshold']:
            logger.info(f"🔄 CoT Decision: Continuing - Low confidence ({current_cot.overall_confidence:.2f})")
            return True

        # Continue if coherence is low
        if current_cot.reasoning_coherence < 0.7:
            logger.info(f"🔄 CoT Decision: Continuing - Low coherence ({current_cot.reasoning_coherence:.2f})")
            return True

        # Continue if there are significant thought changes (indicates active refinement)
        if len(current_cot.thought_changes) > 0:
            logger.info(f"🔄 CoT Decision: Continuing - Active thought changes detected")
            return True

        # Stop if we have consistent high-quality reasoning
        logger.info("🛑 CoT Decision: Stopping - High confidence and coherence achieved")
        return False

    def reset(self):
        """Reset the CoT analyzer for a new refinement session."""
        self.previous_cot = None
        self.cot_history = []
        logger.info("🔄 CoT analyzer reset for new session")


def create_cot_prompt() -> str:
    """Create a prompt that encourages change-of-thought reasoning."""
    return """Please think through this step-by-step and explain your reasoning process. 
As you work through the problem:
1. Break down your thinking into clear steps
2. Indicate your confidence level in each step
3. Explain how each step connects to the next
4. Note any changes in your reasoning approach

Focus on making your thought process transparent and traceable."""