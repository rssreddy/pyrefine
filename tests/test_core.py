"""
Tests for core modules of SELF-REFINE with Change-of-Thought framework.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from pyrefine.core.llm_clients import LLMClientManager, OpenAIClient, GeminiClient
from pyrefine.core.cot_capture import ChangeOfThoughtAnalyzer, ReasoningStep
from pyrefine.core.critic import Critic, CriticFeedback
from pyrefine.core.adaptive_stopping import (
    AdaptiveStoppingManager, 
    ConfidenceStoppingCriterion,
    ConsistencyStoppingCriterion
)
from pyrefine.config.settings import Config


class TestLLMClientManager:
    """Test LLM Client Manager functionality."""
    
    def test_initialization(self):
        """Test client manager initialization."""
        manager = LLMClientManager()
        assert manager.primary_client_name in ['openai', 'gemini']
        assert manager.fallback_client_name in ['openai', 'gemini']
        assert manager.primary_client_name != manager.fallback_client_name
    
    def test_client_switching(self):
        """Test switching between clients."""
        manager = LLMClientManager()
        original_primary = manager.primary_client_name
        
        # Switch to the other client
        new_primary = 'gemini' if original_primary == 'openai' else 'openai'
        manager.switch_primary_client(new_primary)
        
        assert manager.primary_client_name == new_primary
    
    def test_invalid_client_switch(self):
        """Test switching to invalid client."""
        manager = LLMClientManager()
        
        with pytest.raises(ValueError):
            manager.switch_primary_client('invalid_client')
    
    @pytest.mark.asyncio
    async def test_generate_response_mock(self):
        """Test response generation with mocked clients."""
        with patch('pyrefine.core.llm_clients.OpenAIClient') as mock_openai:
            # Mock the client
            mock_client = Mock()
            mock_client.generate_response = AsyncMock(return_value="Test response")
            mock_client.get_model_name.return_value = "gpt-4o"
            mock_openai.return_value = mock_client
            
            manager = LLMClientManager()
            manager.clients['openai'] = mock_client
            
            from langchain_core.messages import HumanMessage
            messages = [HumanMessage(content="Test prompt")]
            
            response, model = await manager.generate_response(messages)
            
            assert response == "Test response"
            assert model == "gpt-4o"
            mock_client.generate_response.assert_called_once()


class TestChangeOfThoughtAnalyzer:
    """Test Change-of-Thought analyzer functionality."""
    
    def test_initialization(self):
        """Test CoT analyzer initialization."""
        analyzer = ChangeOfThoughtAnalyzer()
        assert analyzer.previous_cot is None
        assert len(analyzer.cot_history) == 0
    
    def test_extract_reasoning_steps(self):
        """Test reasoning step extraction."""
        analyzer = ChangeOfThoughtAnalyzer()
        
        text = """
        First, I need to analyze the problem carefully.
        Then, I will consider multiple approaches.
        Finally, I will synthesize the best solution.
        """
        
        steps = analyzer.extract_reasoning_steps(text, iteration=0)
        
        assert len(steps) > 0
        assert all(isinstance(step, ReasoningStep) for step in steps)
        assert all(step.confidence > 0 for step in steps)
    
    def test_capture_cot(self):
        """Test CoT capture functionality."""
        analyzer = ChangeOfThoughtAnalyzer()
        
        response_text = "I think this is a complex problem that requires careful analysis."
        cot_capture = analyzer.capture_cot(response_text, iteration=0)
        
        assert cot_capture.iteration == 0
        assert cot_capture.overall_confidence > 0
        assert cot_capture.reasoning_coherence >= 0
        assert len(analyzer.cot_history) == 1
    
    def test_thought_changes_detection(self):
        """Test detection of thought changes between iterations."""
        analyzer = ChangeOfThoughtAnalyzer()
        
        # First iteration
        text1 = "I will analyze this step by step."
        cot1 = analyzer.capture_cot(text1, iteration=0)
        
        # Second iteration with different approach
        text2 = "Actually, I think a synthesis approach would be better."
        cot2 = analyzer.capture_cot(text2, iteration=1)
        
        assert len(cot2.thought_changes) > 0
    
    def test_reset(self):
        """Test analyzer reset functionality."""
        analyzer = ChangeOfThoughtAnalyzer()
        
        # Add some data
        analyzer.capture_cot("Test response", iteration=0)
        assert len(analyzer.cot_history) == 1
        
        # Reset
        analyzer.reset()
        assert analyzer.previous_cot is None
        assert len(analyzer.cot_history) == 0


class TestCritic:
    """Test Critic module functionality."""
    
    @pytest.fixture
    def mock_llm_manager(self):
        """Create a mock LLM manager."""
        manager = Mock()
        manager.generate_response = AsyncMock(
            return_value=('{"feedback": "Test feedback"}', "gpt-4o")
        )
        return manager
    
    def test_initialization(self, mock_llm_manager):
        """Test critic initialization."""
        critic = Critic(mock_llm_manager)
        assert critic.llm_manager == mock_llm_manager
        assert critic.config is not None
    
    @pytest.mark.asyncio
    async def test_analyze_response(self, mock_llm_manager):
        """Test response analysis."""
        critic = Critic(mock_llm_manager)
        
        feedback = await critic.analyze_response(
            original_prompt="Test prompt",
            response="Test response"
        )
        
        assert "feedback" in feedback
        assert "meta" in feedback
        mock_llm_manager.generate_response.assert_called_once()
    
    def test_parse_structured_feedback(self, mock_llm_manager):
        """Test parsing of structured feedback."""
        critic = Critic(mock_llm_manager)
        
        feedback_json = {
            "feedback": "This response has some factual errors and logic gaps."
        }
        
        feedback_items = critic.parse_structured_feedback(feedback_json)
        
        assert len(feedback_items) > 0
        assert all(isinstance(item, CriticFeedback) for item in feedback_items)
    
    def test_format_feedback_for_refinement(self, mock_llm_manager):
        """Test feedback formatting."""
        critic = Critic(mock_llm_manager)
        
        feedback_items = [
            CriticFeedback(
                feedback="High severity issue",
                category="factual_errors",
                severity="high",
                suggestion="Fix this",
                confidence=0.9
            ),
            CriticFeedback(
                feedback="Medium severity issue", 
                category="style_issues",
                severity="medium",
                suggestion="Improve this",
                confidence=0.7
            )
        ]
        
        formatted = critic.format_feedback_for_refinement(feedback_items)
        
        assert "CRITICAL ISSUES" in formatted
        assert "IMPROVEMENT AREAS" in formatted


class TestAdaptiveStoppingManager:
    """Test Adaptive Stopping Manager functionality."""
    
    def test_initialization(self):
        """Test stopping manager initialization."""
        manager = AdaptiveStoppingManager()
        assert manager.stopping_criterion is not None
        assert len(manager.decision_history) == 0
    
    def test_confidence_stopping_criterion(self):
        """Test confidence-based stopping."""
        criterion = ConfidenceStoppingCriterion(threshold=0.8)
        
        # Mock CoT captures with different confidence levels
        from pyrefine.core.cot_capture import CoTCapture
        
        low_confidence_cot = CoTCapture(
            iteration=0,
            reasoning_steps=[],
            overall_confidence=0.6,
            reasoning_coherence=0.7,
            thought_changes=[],
            metadata={}
        )
        
        high_confidence_cot = CoTCapture(
            iteration=1,
            reasoning_steps=[],
            overall_confidence=0.9,
            reasoning_coherence=0.8,
            thought_changes=[],
            metadata={}
        )
        
        # Test with low confidence
        decision1 = criterion.should_stop(
            iteration=0,
            responses=["response1"],
            cot_captures=[low_confidence_cot],
            critic_feedback=[]
        )
        assert not decision1.should_stop
        
        # Test with high confidence
        decision2 = criterion.should_stop(
            iteration=1,
            responses=["response1", "response2"],
            cot_captures=[low_confidence_cot, high_confidence_cot],
            critic_feedback=[]
        )
        assert decision2.should_stop
    
    def test_consistency_stopping_criterion(self):
        """Test consistency-based stopping."""
        criterion = ConsistencyStoppingCriterion(threshold=0.8, window_size=2)
        
        # Test with similar responses
        similar_responses = ["This is a response", "This is a similar response"]
        
        decision = criterion.should_stop(
            iteration=1,
            responses=similar_responses,
            cot_captures=[],
            critic_feedback=[]
        )
        
        # Should have some consistency score
        assert decision.confidence > 0
    
    def test_stopping_manager_decisions(self):
        """Test stopping manager decision making."""
        manager = AdaptiveStoppingManager()
        
        decision = manager.should_stop(
            iteration=0,
            responses=["test response"],
            cot_captures=[],
            critic_feedback=[],
            max_iterations=3,
            min_iterations=1
        )
        
        assert len(manager.decision_history) == 1
        assert decision.metadata["iteration"] == 0
    
    def test_reset(self):
        """Test stopping manager reset."""
        manager = AdaptiveStoppingManager()
        
        # Add a decision
        manager.should_stop(
            iteration=0,
            responses=["test"],
            cot_captures=[],
            critic_feedback=[],
            max_iterations=3,
            min_iterations=1
        )
        
        assert len(manager.decision_history) == 1
        
        # Reset
        manager.reset()
        assert len(manager.decision_history) == 0


class TestConfiguration:
    """Test configuration functionality."""
    
    def test_config_creation(self):
        """Test configuration object creation."""
        config = Config()
        
        assert config.llm_client.primary in ['openai', 'gemini']
        assert config.self_refine.max_iterations > 0
        assert config.adaptive_stopping.enabled is True
        assert config.critic.enabled is True
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid configuration
        config = Config(
            self_refine={'max_iterations': 5, 'min_iterations': 1}
        )
        assert config.self_refine.max_iterations == 5
        
        # Test invalid configuration should raise validation error
        with pytest.raises(Exception):  # Pydantic validation error
            Config(self_refine={'max_iterations': -1})


if __name__ == "__main__":
    pytest.main([__file__])