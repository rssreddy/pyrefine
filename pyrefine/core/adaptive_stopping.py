"""
Adaptive Stopping Criteria Module

This module implements various stopping criteria for the SELF-REFINE process,
determining when the iterative refinement should halt.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from datetime import datetime

from pyrefine.config.settings import get_config
from pyrefine.core.cot_capture import CoTCapture, ChangeOfThoughtAnalyzer
from pyrefine.core.critic import CriticFeedback


logger = logging.getLogger(__name__)


@dataclass
class StoppingDecision:
    """Decision about whether to stop the refinement process."""
    should_stop: bool
    reason: str
    confidence: float
    metadata: Dict[str, Any]


class BaseStoppingCriterion(ABC):
    """Abstract base class for stopping criteria."""

    @abstractmethod
    def should_stop(
        self, 
        iteration: int,
        responses: List[str],
        cot_captures: List[CoTCapture],
        critic_feedback: List[List[CriticFeedback]],
        **kwargs
    ) -> StoppingDecision:
        """Determine if the process should stop."""
        pass


class ConfidenceStoppingCriterion(BaseStoppingCriterion):
    """Stop based on confidence threshold."""

    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold

    def should_stop(
        self, 
        iteration: int,
        responses: List[str],
        cot_captures: List[CoTCapture],
        critic_feedback: List[List[CriticFeedback]],
        **kwargs
    ) -> StoppingDecision:
        """Stop if confidence exceeds threshold."""
        if not cot_captures:
            return StoppingDecision(
                should_stop=False,
                reason="No CoT data available",
                confidence=0.0,
                metadata={}
            )

        current_confidence = cot_captures[-1].overall_confidence

        should_stop = current_confidence >= self.threshold

        return StoppingDecision(
            should_stop=should_stop,
            reason=f"Confidence {current_confidence:.3f} {'exceeds' if should_stop else 'below'} threshold {self.threshold}",
            confidence=current_confidence,
            metadata={
                "threshold": self.threshold,
                "current_confidence": current_confidence,
                "iteration": iteration
            }
        )


class ConsistencyStoppingCriterion(BaseStoppingCriterion):
    """Stop based on response consistency."""

    def __init__(self, threshold: float = 0.85, window_size: int = 2):
        self.threshold = threshold
        self.window_size = window_size

    def should_stop(
        self, 
        iteration: int,
        responses: List[str],
        cot_captures: List[CoTCapture],
        critic_feedback: List[List[CriticFeedback]],
        **kwargs
    ) -> StoppingDecision:
        """Stop if recent responses are consistent."""
        if len(responses) < self.window_size:
            return StoppingDecision(
                should_stop=False,
                reason=f"Need at least {self.window_size} responses",
                confidence=0.0,
                metadata={}
            )

        # Calculate consistency of recent responses
        recent_responses = responses[-self.window_size:]
        consistency = self._calculate_consistency(recent_responses)

        should_stop = consistency >= self.threshold

        return StoppingDecision(
            should_stop=should_stop,
            reason=f"Consistency {consistency:.3f} {'exceeds' if should_stop else 'below'} threshold {self.threshold}",
            confidence=consistency,
            metadata={
                "threshold": self.threshold,
                "consistency": consistency,
                "window_size": self.window_size,
                "responses_analyzed": len(recent_responses)
            }
        )

    def _calculate_consistency(self, responses: List[str]) -> float:
        """Calculate consistency between responses."""
        if len(responses) < 2:
            return 1.0

        # Simple consistency metric based on length and word overlap
        consistencies = []

        for i in range(len(responses) - 1):
            response1 = responses[i]
            response2 = responses[i + 1]

            # Length similarity
            len1, len2 = len(response1), len(response2)
            length_sim = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 1.0

            # Word overlap similarity
            words1 = set(response1.lower().split())
            words2 = set(response2.lower().split())

            if len(words1) == 0 and len(words2) == 0:
                word_sim = 1.0
            else:
                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))
                word_sim = intersection / union if union > 0 else 0.0

            # Combined similarity
            consistency = (length_sim + word_sim) / 2
            consistencies.append(consistency)

        return sum(consistencies) / len(consistencies)


class ImprovementStoppingCriterion(BaseStoppingCriterion):
    """Stop when improvement rate falls below threshold."""

    def __init__(self, threshold: float = 0.05, window_size: int = 2):
        self.threshold = threshold
        self.window_size = window_size

    def should_stop(
        self, 
        iteration: int,
        responses: List[str],
        cot_captures: List[CoTCapture],
        critic_feedback: List[List[CriticFeedback]],
        **kwargs
    ) -> StoppingDecision:
        """Stop if improvement rate is below threshold."""
        if len(cot_captures) < self.window_size:
            return StoppingDecision(
                should_stop=False,
                reason=f"Need at least {self.window_size} iterations",
                confidence=0.0,
                metadata={}
            )

        # Calculate improvement rate
        improvement = self._calculate_improvement_rate(cot_captures)

        should_stop = improvement < self.threshold

        return StoppingDecision(
            should_stop=should_stop,
            reason=f"Improvement rate {improvement:.3f} {'below' if should_stop else 'above'} threshold {self.threshold}",
            confidence=1.0 - improvement,  # Higher confidence when improvement is low
            metadata={
                "threshold": self.threshold,
                "improvement_rate": improvement,
                "window_size": self.window_size
            }
        )

    def _calculate_improvement_rate(self, cot_captures: List[CoTCapture]) -> float:
        """Calculate the rate of improvement."""
        if len(cot_captures) < 2:
            return 1.0  # Assume high improvement if we can't calculate

        # Use confidence as improvement metric
        recent_confidences = [cot.overall_confidence for cot in cot_captures[-self.window_size:]]

        if len(recent_confidences) < 2:
            return 1.0

        # Calculate average improvement
        improvements = []
        for i in range(1, len(recent_confidences)):
            improvement = recent_confidences[i] - recent_confidences[i-1]
            improvements.append(max(0, improvement))  # Only positive improvements

        return sum(improvements) / len(improvements) if improvements else 0.0


class CriticBasedStoppingCriterion(BaseStoppingCriterion):
    """Stop based on critic feedback analysis."""

    def __init__(self, max_high_severity: int = 0, max_medium_severity: int = 1):
        self.max_high_severity = max_high_severity
        self.max_medium_severity = max_medium_severity

    def should_stop(
        self, 
        iteration: int,
        responses: List[str],
        cot_captures: List[CoTCapture],
        critic_feedback: List[List[CriticFeedback]],
        **kwargs
    ) -> StoppingDecision:
        """Stop based on critic feedback severity."""
        if not critic_feedback or not critic_feedback[-1]:
            return StoppingDecision(
                should_stop=False,
                reason="No critic feedback available",
                confidence=0.0,
                metadata={}
            )

        current_feedback = critic_feedback[-1]

        # Count severity levels
        high_count = sum(1 for f in current_feedback if f.severity == "high")
        medium_count = sum(1 for f in current_feedback if f.severity == "medium")
        low_count = sum(1 for f in current_feedback if f.severity == "low")

        # Decide based on severity thresholds
        should_stop = (
            high_count <= self.max_high_severity and 
            medium_count <= self.max_medium_severity
        )

        return StoppingDecision(
            should_stop=should_stop,
            reason=f"Severity levels: high={high_count}, medium={medium_count}, low={low_count}",
            confidence=0.9 if should_stop else 0.3,
            metadata={
                "high_count": high_count,
                "medium_count": medium_count,
                "low_count": low_count,
                "max_high_severity": self.max_high_severity,
                "max_medium_severity": self.max_medium_severity
            }
        )


class CompositeStoppingCriterion(BaseStoppingCriterion):
    """Composite stopping criterion that combines multiple criteria."""

    def __init__(self, criteria: List[Tuple[BaseStoppingCriterion, float]]):
        """
        Initialize with weighted criteria.

        Args:
            criteria: List of (criterion, weight) tuples
        """
        self.criteria = criteria
        self.total_weight = sum(weight for _, weight in criteria)

    def should_stop(
        self, 
        iteration: int,
        responses: List[str],
        cot_captures: List[CoTCapture],
        critic_feedback: List[List[CriticFeedback]],
        **kwargs
    ) -> StoppingDecision:
        """Stop based on weighted combination of criteria."""
        decisions = []
        weighted_confidence = 0.0
        reasons = []

        for criterion, weight in self.criteria:
            decision = criterion.should_stop(
                iteration, responses, cot_captures, critic_feedback, **kwargs
            )
            decisions.append(decision)

            # Weight the confidence
            weighted_confidence += decision.confidence * weight

            # Collect reasons
            reasons.append(f"{criterion.__class__.__name__}: {decision.reason}")

        # Normalize confidence
        final_confidence = weighted_confidence / self.total_weight

        # Decision based on majority voting or confidence threshold
        stop_votes = sum(1 for d in decisions if d.should_stop)
        should_stop = stop_votes > len(decisions) / 2 or final_confidence > 0.8

        return StoppingDecision(
            should_stop=should_stop,
            reason=f"Composite decision (confidence={final_confidence:.3f}): " + "; ".join(reasons),
            confidence=final_confidence,
            metadata={
                "individual_decisions": [
                    {
                        "criterion": criterion.__class__.__name__,
                        "should_stop": decision.should_stop,
                        "confidence": decision.confidence,
                        "weight": weight
                    }
                    for (criterion, weight), decision in zip(self.criteria, decisions)
                ],
                "stop_votes": stop_votes,
                "total_criteria": len(decisions)
            }
        )


class AdaptiveStoppingManager:
    """Manager for adaptive stopping criteria."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize adaptive stopping manager."""
        self.config = config or get_config().adaptive_stopping.model_dump()
        self.stopping_criterion = self._create_stopping_criterion()
        self.decision_history: List[StoppingDecision] = []

        logger.info(f"Adaptive stopping manager initialized with {self.config['criteria_type']} criterion")

    def _create_stopping_criterion(self) -> BaseStoppingCriterion:
        """Create stopping criterion based on configuration."""
        criteria_type = self.config.get("criteria_type", "confidence")

        if criteria_type == "confidence":
            return ConfidenceStoppingCriterion(
                threshold=self.config.get("confidence_threshold", 0.9)
            )
        elif criteria_type == "consistency":
            return ConsistencyStoppingCriterion(
                threshold=self.config.get("consistency_threshold", 0.85)
            )
        elif criteria_type == "improvement":
            return ImprovementStoppingCriterion(
                threshold=self.config.get("improvement_threshold", 0.05)
            )
        elif criteria_type == "critic":
            return CriticBasedStoppingCriterion()
        elif criteria_type == "composite":
            # Create composite criterion with multiple criteria
            criteria = [
                (ConfidenceStoppingCriterion(self.config.get("confidence_threshold", 0.9)), 0.4),
                (ConsistencyStoppingCriterion(self.config.get("consistency_threshold", 0.85)), 0.3),
                (ImprovementStoppingCriterion(self.config.get("improvement_threshold", 0.05)), 0.2),
                (CriticBasedStoppingCriterion(), 0.1)
            ]
            return CompositeStoppingCriterion(criteria)
        else:
            logger.warning(f"Unknown criteria type: {criteria_type}, using confidence")
            return ConfidenceStoppingCriterion()

    def should_stop(
        self,
        iteration: int,
        responses: List[str],
        cot_captures: List[CoTCapture],
        critic_feedback: List[List[CriticFeedback]],
        max_iterations: int = 4,
        min_iterations: int = 1,
        **kwargs
    ) -> StoppingDecision:
        """
        Determine if the refinement process should stop.

        Args:
            iteration: Current iteration number
            responses: List of responses so far
            cot_captures: List of CoT captures
            critic_feedback: List of critic feedback per iteration
            max_iterations: Maximum number of iterations
            min_iterations: Minimum number of iterations

        Returns:
            StoppingDecision object
        """
        # Check minimum iterations
        if iteration < min_iterations:
            decision = StoppingDecision(
                should_stop=False,
                reason=f"Below minimum iterations ({iteration} < {min_iterations})",
                confidence=0.0,
                metadata={"iteration": iteration, "min_iterations": min_iterations}
            )
            self.decision_history.append(decision)
            return decision

        # Check maximum iterations
        if iteration >= max_iterations:
            decision = StoppingDecision(
                should_stop=True,
                reason=f"Reached maximum iterations ({iteration} >= {max_iterations})",
                confidence=1.0,
                metadata={"iteration": iteration, "max_iterations": max_iterations}
            )
            self.decision_history.append(decision)
            return decision

        # Use configured stopping criterion
        decision = self.stopping_criterion.should_stop(
            iteration, responses, cot_captures, critic_feedback, **kwargs
        )

        # Add iteration info to metadata
        decision.metadata.update({
            "iteration": iteration,
            "max_iterations": max_iterations,
            "min_iterations": min_iterations
        })

        self.decision_history.append(decision)

        logger.info(f"Stopping decision for iteration {iteration}: {decision.reason}")

        return decision

    def get_stopping_history(self) -> List[StoppingDecision]:
        """Get the history of stopping decisions."""
        return self.decision_history.copy()

    def reset(self):
        """Reset the stopping manager for a new refinement session."""
        self.decision_history = []
        logger.info("Adaptive stopping manager reset")
