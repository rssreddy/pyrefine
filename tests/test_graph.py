"""
Tests for graph module of SELF-REFINE with Change-of-Thought framework.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from pyrefine.graph.self_refine_graph import SelfRefineGraph, SelfRefineState
from pyrefine.core.llm_clients import LLMClientManager
from pyrefine.core.cot_capture import ChangeOfThoughtAnalyzer
from pyrefine.core.critic import Critic
from pyrefine.core.adaptive_stopping import AdaptiveStoppingManager


class TestSelfRefineGraph:
    """Test SelfRefineGraph functionality."""
    
    def test_initialization(self):
        """Test graph initialization."""
        graph = SelfRefineGraph()
        
        assert graph.llm_manager is not None
        assert graph.cot_analyzer is not None
        assert graph.critic is not None
        assert graph.stopping_manager is not None
        assert graph.graph is not None
    
    def test_graph_structure(self):
        """Test graph structure and topology."""
        graph = SelfRefineGraph()
        
        # Check that the graph has the expected structure
        visualization = graph.get_graph_visualization()
        
        assert "initialize" in visualization
        assert "generate_initial" in visualization
        assert "capture_cot" in visualization
        assert "critic_analysis" in visualization
        assert "check_stopping" in visualization
        assert "refine_response" in visualization
        assert "finalize" in visualization
    
    @pytest.mark.asyncio
    async def test_initialize_node(self):
        """Test the initialize node."""
        graph = SelfRefineGraph()
        
        initial_state = SelfRefineState(
            original_prompt="Test prompt",
            iteration=0,
            max_iterations=3,
            min_iterations=1,
            responses=[],
            current_response="",
            cot_captures=[],
            current_cot=None,
            critic_feedback_history=[],
            current_critic_feedback=[],
            formatted_feedback="",
            stopping_decisions=[],
            should_stop=False,
            messages=[],
            session_id="",
            start_time=None,
            end_time=None,
            model_used=""
        )
        
        result = await graph._initialize_node(initial_state)
        
        assert result["iteration"] == 0
        assert result["session_id"].startswith("selfrefine_")
        assert "start_time" in result
    
    @pytest.mark.asyncio
    async def test_generate_initial_node_mock(self):
        """Test the generate_initial node with mocked LLM."""
        with patch.object(LLMClientManager, 'generate_response') as mock_generate:
            mock_generate.return_value = ("Test response", "gpt-4o")
            
            graph = SelfRefineGraph()
            
            state = SelfRefineState(
                original_prompt="Test prompt",
                iteration=0,
                max_iterations=3,
                min_iterations=1,
                responses=[],
                current_response="",
                cot_captures=[],
                current_cot=None,
                critic_feedback_history=[],
                current_critic_feedback=[],
                formatted_feedback="",
                stopping_decisions=[],
                should_stop=False,
                messages=[],
                session_id="test_session",
                start_time=None,
                end_time=None,
                model_used=""
            )
            
            result = await graph._generate_initial_node(state)
            
            assert result["current_response"] == "Test response"
            assert result["model_used"] == "gpt-4o"
            mock_generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_capture_cot_node(self):
        """Test the capture_cot node."""
        graph = SelfRefineGraph()
        
        state = SelfRefineState(
            original_prompt="Test prompt",
            iteration=0,
            max_iterations=3,
            min_iterations=1,
            responses=[],
            current_response="This is a test response with reasoning steps.",
            cot_captures=[],
            current_cot=None,
            critic_feedback_history=[],
            current_critic_feedback=[],
            formatted_feedback="",
            stopping_decisions=[],
            should_stop=False,
            messages=[],
            session_id="test_session",
            start_time=None,
            end_time=None,
            model_used=""
        )
        
        result = await graph._capture_cot_node(state)
        
        assert len(result["responses"]) == 1
        assert result["current_cot"] is not None
        assert len(result["cot_captures"]) == 1
    
    @pytest.mark.asyncio
    async def test_critic_analysis_node_mock(self):
        """Test the critic_analysis node with mocked critic."""
        with patch.object(Critic, 'analyze_response') as mock_analyze:
            mock_analyze.return_value = {"feedback": "Test feedback"}
            
            with patch.object(Critic, 'parse_structured_feedback') as mock_parse:
                from pyrefine.core.critic import CriticFeedback
                mock_parse.return_value = [
                    CriticFeedback(
                        feedback="Test feedback",
                        category="general",
                        severity="medium",
                        suggestion="Improve this",
                        confidence=0.8
                    )
                ]
                
                with patch.object(Critic, 'format_feedback_for_refinement') as mock_format:
                    mock_format.return_value = "Formatted feedback"
                    
                    graph = SelfRefineGraph()
                    
                    state = SelfRefineState(
                        original_prompt="Test prompt",
                        iteration=0,
                        max_iterations=3,
                        min_iterations=1,
                        responses=["Test response"],
                        current_response="Test response",
                        cot_captures=[],
                        current_cot=None,
                        critic_feedback_history=[],
                        current_critic_feedback=[],
                        formatted_feedback="",
                        stopping_decisions=[],
                        should_stop=False,
                        messages=[],
                        session_id="test_session",
                        start_time=None,
                        end_time=None,
                        model_used=""
                    )
                    
                    result = await graph._critic_analysis_node(state)
                    
                    assert len(result["current_critic_feedback"]) == 1
                    assert result["formatted_feedback"] == "Formatted feedback"
                    assert len(result["critic_feedback_history"]) == 1
    
    @pytest.mark.asyncio
    async def test_check_stopping_node(self):
        """Test the check_stopping node."""
        with patch.object(AdaptiveStoppingManager, 'should_stop') as mock_should_stop:
            from pyrefine.core.adaptive_stopping import StoppingDecision
            
            mock_should_stop.return_value = StoppingDecision(
                should_stop=True,
                reason="Test stopping",
                confidence=0.9,
                metadata={}
            )
            
            graph = SelfRefineGraph()
            
            state = SelfRefineState(
                original_prompt="Test prompt",
                iteration=1,
                max_iterations=3,
                min_iterations=1,
                responses=["Test response"],
                current_response="Test response",
                cot_captures=[],
                current_cot=None,
                critic_feedback_history=[],
                current_critic_feedback=[],
                formatted_feedback="",
                stopping_decisions=[],
                should_stop=False,
                messages=[],
                session_id="test_session",
                start_time=None,
                end_time=None,
                model_used=""
            )
            
            result = await graph._check_stopping_node(state)
            
            assert result["should_stop"] is True
            assert len(result["stopping_decisions"]) == 1
    
    def test_should_continue_decision(self):
        """Test the conditional edge decision."""
        graph = SelfRefineGraph()
        
        # Test continue case
        state_continue = SelfRefineState(
            original_prompt="Test prompt",
            iteration=1,
            max_iterations=3,
            min_iterations=1,
            responses=[],
            current_response="",
            cot_captures=[],
            current_cot=None,
            critic_feedback_history=[],
            current_critic_feedback=[],
            formatted_feedback="",
            stopping_decisions=[],
            should_stop=False,
            messages=[],
            session_id="test_session",
            start_time=None,
            end_time=None,
            model_used=""
        )
        
        decision = graph._should_continue(state_continue)
        assert decision == "continue"
        
        # Test stop case
        state_stop = state_continue.copy()
        state_stop["should_stop"] = True
        
        decision = graph._should_continue(state_stop)
        assert decision == "stop"
    
    @pytest.mark.asyncio
    async def test_refine_response_node_mock(self):
        """Test the refine_response node with mocked LLM."""
        with patch.object(LLMClientManager, 'generate_response') as mock_generate:
            mock_generate.return_value = ("Refined response", "gpt-4o")
            
            graph = SelfRefineGraph()
            
            state = SelfRefineState(
                original_prompt="Test prompt",
                iteration=1,
                max_iterations=3,
                min_iterations=1,
                responses=["Initial response"],
                current_response="Initial response",
                cot_captures=[],
                current_cot=None,
                critic_feedback_history=[],
                current_critic_feedback=[],
                formatted_feedback="Please improve this response",
                stopping_decisions=[],
                should_stop=False,
                messages=[],
                session_id="test_session",
                start_time=None,
                end_time=None,
                model_used=""
            )
            
            result = await graph._refine_response_node(state)
            
            assert result["current_response"] == "Refined response"
            assert result["iteration"] == 2
            assert result["model_used"] == "gpt-4o"
    
    @pytest.mark.asyncio
    async def test_finalize_node(self):
        """Test the finalize node."""
        graph = SelfRefineGraph()
        
        state = SelfRefineState(
            original_prompt="Test prompt",
            iteration=2,
            max_iterations=3,
            min_iterations=1,
            responses=["Response 1", "Response 2"],
            current_response="Response 2",
            cot_captures=[],
            current_cot=None,
            critic_feedback_history=[],
            current_critic_feedback=[],
            formatted_feedback="",
            stopping_decisions=[],
            should_stop=True,
            messages=[],
            session_id="test_session",
            start_time=None,
            end_time=None,
            model_used=""
        )
        
        result = await graph._finalize_node(state)
        
        assert "end_time" in result
    
    @pytest.mark.asyncio
    async def test_full_refine_process_mock(self):
        """Test the full refine process with extensive mocking."""
        with patch.object(LLMClientManager, 'generate_response') as mock_generate:
            # Mock different responses for initial and refinement
            mock_generate.side_effect = [
                ("Initial response", "gpt-4o"),
                ("Refined response", "gpt-4o")
            ]
            
            with patch.object(Critic, 'analyze_response') as mock_analyze:
                mock_analyze.return_value = {"feedback": "Needs improvement"}
                
                with patch.object(Critic, 'parse_structured_feedback') as mock_parse:
                    from pyrefine.core.critic import CriticFeedback
                    mock_parse.return_value = [
                        CriticFeedback(
                            feedback="Minor issues",
                            category="style",
                            severity="low",
                            suggestion="Polish the style",
                            confidence=0.7
                        )
                    ]
                    
                    with patch.object(Critic, 'format_feedback_for_refinement') as mock_format:
                        mock_format.return_value = "Please improve the style"
                        
                        # Create graph with minimal iterations
                        config = {
                            "self_refine": {
                                "max_iterations": 2,
                                "min_iterations": 1
                            },
                            "adaptive_stopping": {
                                "criteria_type": "confidence",
                                "confidence_threshold": 0.8
                            }
                        }
                        
                        graph = SelfRefineGraph(config=config)
                        
                        result = await graph.refine("Test prompt for refinement")
                        
                        assert "final_response" in result
                        assert "iteration_count" in result
                        assert "session_id" in result
                        assert "execution_time" in result
                        assert result["iteration_count"] >= 1


class TestUtilityFunctions:
    """Test utility functions in the graph module."""
    
    @pytest.mark.asyncio
    async def test_run_self_refine_function(self):
        """Test the run_self_refine utility function."""
        with patch.object(SelfRefineGraph, 'refine') as mock_refine:
            mock_refine.return_value = {
                "final_response": "Test response",
                "iteration_count": 1,
                "session_id": "test_session",
                "execution_time": 1.5
            }
            
            from pyrefine.graph.self_refine_graph import run_self_refine
            
            result = await run_self_refine("Test prompt")
            
            assert result["final_response"] == "Test response"
            mock_refine.assert_called_once()
    
    def test_create_self_refine_graph_function(self):
        """Test the create_self_refine_graph utility function."""
        from pyrefine.graph.self_refine_graph import create_self_refine_graph
        
        config = {"self_refine": {"max_iterations": 5}}
        graph = create_self_refine_graph(config)
        
        assert isinstance(graph, SelfRefineGraph)


if __name__ == "__main__":
    pytest.main([__file__])