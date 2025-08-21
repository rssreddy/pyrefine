"""
SELF-REFINE with Change-of-Thought LangGraph Implementation

This module implements the main LangGraph DAG for orchestrating the SELF-REFINE process
with Change-of-Thought capture as specified in the requirements.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, TypedDict, Annotated
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from pyrefine.config.settings import get_config
from pyrefine.core.llm_clients import LLMClientManager, create_conversation
from pyrefine.core.cot_capture import ChangeOfThoughtAnalyzer, CoTCapture
from pyrefine.core.critic import Critic, CriticFeedback, create_refinement_prompt
from pyrefine.core.adaptive_stopping import AdaptiveStoppingManager, StoppingDecision


logger = logging.getLogger(__name__)


class SelfRefineState(TypedDict):
    """State schema for the SELF-REFINE graph."""
    # Input
    original_prompt: str

    # Iteration tracking
    iteration: int
    max_iterations: int
    min_iterations: int

    # Responses and refinements
    responses: List[str]
    current_response: str

    # Change-of-Thought captures
    cot_captures: List[CoTCapture]
    current_cot: Optional[CoTCapture]

    # Critic feedback
    critic_feedback_history: List[List[CriticFeedback]]
    current_critic_feedback: List[CriticFeedback]
    formatted_feedback: str

    # Stopping decisions
    stopping_decisions: List[StoppingDecision]
    should_stop: bool

    # Messages for LLM communication
    messages: Annotated[List[BaseMessage], add_messages]

    # Metadata
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    model_used: str


class SelfRefineGraph:
    """
    Main SELF-REFINE with Change-of-Thought graph implementation.

    This class orchestrates the entire refinement process using LangGraph
    as specified in the project requirements.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the SELF-REFINE graph."""
        self.config = get_config()
        self.custom_config = config or {}
        
        if config:
            # Override with provided config
            for key, value in config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

        # Initialize components
        self.llm_manager = LLMClientManager()
        
        # Pass the cot_capture config to the analyzer if available
        cot_config = None
        if config and 'self_refine' in config and 'cot_capture' in config['self_refine']:
            cot_config = config['self_refine']['cot_capture']
        
        # Pass the critic config if available
        critic_config = None
        if config and 'critic' in config:
            critic_config = config['critic']
        
        # Pass the adaptive_stopping config if available
        stopping_config = None
        if config and 'adaptive_stopping' in config:
            stopping_config = config['adaptive_stopping']
        
        self.cot_analyzer = ChangeOfThoughtAnalyzer(config=cot_config)
        self.critic = Critic(self.llm_manager, config=critic_config)
        self.stopping_manager = AdaptiveStoppingManager(config=stopping_config)

        # Build the graph
        self.graph = self._build_graph()

        logger.info("SELF-REFINE graph initialized")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph DAG."""
        # Create state graph
        workflow = StateGraph(SelfRefineState)

        # Add nodes
        workflow.add_node("initialize", self._initialize_node)
        workflow.add_node("generate_initial", self._generate_initial_node)
        workflow.add_node("capture_cot", self._capture_cot_node)
        workflow.add_node("critic_analysis", self._critic_analysis_node)
        workflow.add_node("check_stopping", self._check_stopping_node)
        workflow.add_node("refine_response", self._refine_response_node)
        workflow.add_node("finalize", self._finalize_node)

        # Add edges
        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "generate_initial")
        workflow.add_edge("generate_initial", "capture_cot")
        workflow.add_edge("capture_cot", "critic_analysis")
        workflow.add_edge("critic_analysis", "check_stopping")

        # Conditional edge from check_stopping
        workflow.add_conditional_edges(
            "check_stopping",
            self._should_continue,
            {
                "continue": "refine_response",
                "stop": "finalize"
            }
        )

        workflow.add_edge("refine_response", "capture_cot")
        workflow.add_edge("finalize", END)

        # Add memory saver if configured
        memory = MemorySaver()

        return workflow.compile(checkpointer=memory)

    # Node implementations

    async def _initialize_node(self, state: SelfRefineState) -> Dict[str, Any]:
        """Initialize the refinement session."""
        logger.info("Initializing SELF-REFINE session")

        session_id = f"selfrefine_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Reset components
        self.cot_analyzer.reset()
        self.stopping_manager.reset()

        # Get max/min iterations from custom config or use defaults
        max_iterations = self.custom_config.get('self_refine', {}).get('max_iterations', 4)  # Default: 4
        min_iterations = self.custom_config.get('self_refine', {}).get('min_iterations', 1)  # Default: 1

        return {
            "iteration": 0,
            "max_iterations": max_iterations,
            "min_iterations": min_iterations,
            "responses": [],
            "cot_captures": [],
            "critic_feedback_history": [],
            "stopping_decisions": [],
            "should_stop": False,
            "session_id": session_id,
            "start_time": datetime.now(),
            "messages": []
        }

    async def _generate_initial_node(self, state: SelfRefineState) -> Dict[str, Any]:
        """Generate the initial response."""
        logger.info("Generating initial response")

        # Create initial conversation
        system_prompt = """You are an expert assistant providing comprehensive and accurate responses. 
Think through your response step-by-step and provide detailed reasoning for your conclusions."""

        messages = create_conversation(
            system_prompt=system_prompt,
            user_message=state["original_prompt"]
        )

        # Generate response
        response, model_used = await self.llm_manager.generate_response(messages)

        return {
            "current_response": response,
            "model_used": model_used,
            "messages": messages + [AIMessage(content=response)]
        }

    async def _capture_cot_node(self, state: SelfRefineState) -> Dict[str, Any]:
        """Capture Change-of-Thought from the current response."""
        logger.info(f"Capturing CoT for iteration {state['iteration']}")

        # Capture CoT
        cot_capture = self.cot_analyzer.capture_cot(
            state["current_response"], 
            state["iteration"]
        )

        # Update responses list
        updated_responses = state["responses"] + [state["current_response"]]
        updated_cot_captures = state["cot_captures"] + [cot_capture]

        return {
            "responses": updated_responses,
            "current_cot": cot_capture,
            "cot_captures": updated_cot_captures
        }

    async def _critic_analysis_node(self, state: SelfRefineState) -> Dict[str, Any]:
        """Analyze the response with the critic."""
        logger.info(f"Running critic analysis for iteration {state['iteration']}")

        # Get critic feedback
        feedback_json = await self.critic.analyze_response(
            original_prompt=state["original_prompt"],
            response=state["current_response"],
            context={"iteration": state["iteration"]}
        )

        # Parse structured feedback
        feedback_items = self.critic.parse_structured_feedback(feedback_json)

        # Format feedback for refinement
        formatted_feedback = self.critic.format_feedback_for_refinement(feedback_items)

        # Update feedback history
        updated_feedback_history = state["critic_feedback_history"] + [feedback_items]

        return {
            "current_critic_feedback": feedback_items,
            "formatted_feedback": formatted_feedback,
            "critic_feedback_history": updated_feedback_history
        }

    async def _check_stopping_node(self, state: SelfRefineState) -> Dict[str, Any]:
        """Check if refinement should stop."""
        logger.info(f"Checking stopping criteria for iteration {state['iteration']}")

        # Get stopping decision
        stopping_decision = self.stopping_manager.should_stop(
            iteration=state["iteration"],
            responses=state["responses"],
            cot_captures=state["cot_captures"],
            critic_feedback=state["critic_feedback_history"],
            max_iterations=state["max_iterations"],
            min_iterations=state["min_iterations"]
        )

        # Update stopping decisions
        updated_stopping_decisions = state["stopping_decisions"] + [stopping_decision]

        return {
            "stopping_decisions": updated_stopping_decisions,
            "should_stop": stopping_decision.should_stop
        }

    async def _refine_response_node(self, state: SelfRefineState) -> Dict[str, Any]:
        """Refine the response based on critic feedback."""
        logger.info(f"Refining response for iteration {state['iteration'] + 1}")

        # Create refinement prompt
        refinement_prompt = create_refinement_prompt(
            original_prompt=state["original_prompt"],
            previous_response=state["current_response"],
            feedback=state["formatted_feedback"]
        )

        # Generate refined response
        messages = [
            SystemMessage(content="You are an expert assistant refining responses based on feedback."),
            HumanMessage(content=refinement_prompt)
        ]

        refined_response, model_used = await self.llm_manager.generate_response(messages)

        return {
            "current_response": refined_response,
            "iteration": state["iteration"] + 1,
            "model_used": model_used,
            "messages": state["messages"] + [HumanMessage(content=refinement_prompt)] + [AIMessage(content=refined_response)]
        }

    async def _finalize_node(self, state: SelfRefineState) -> Dict[str, Any]:
        """Finalize the refinement session."""
        logger.info("Finalizing SELF-REFINE session")

        return {
            "end_time": datetime.now()
        }

    def _should_continue(self, state: SelfRefineState) -> str:
        """Determine if refinement should continue."""
        return "stop" if state["should_stop"] else "continue"

    # Public interface

    async def refine(
        self, 
        prompt: str, 
        config_override: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute the SELF-REFINE process.

        Args:
            prompt: The original user prompt to refine
            config_override: Optional configuration overrides

        Returns:
            Dict containing the final result and metadata
        """
        logger.info("Starting SELF-REFINE process")

        # Generate session ID first
        session_id = f"selfrefine_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get max/min iterations from custom config or use defaults
        max_iterations = self.custom_config.get('self_refine', {}).get('max_iterations', 4)  # Default: 4
        min_iterations = self.custom_config.get('self_refine', {}).get('min_iterations', 1)  # Default: 1
        
        # Create initial state
        initial_state = SelfRefineState(
            original_prompt=prompt,
            iteration=0,
            max_iterations=max_iterations,
            min_iterations=min_iterations,
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
            session_id=session_id,
            start_time=datetime.now(),
            end_time=None,
            model_used=""
        )

        # Execute the graph with thread_id for checkpointer
        config = {"configurable": {"thread_id": session_id}}
        final_state = await self.graph.ainvoke(initial_state, config=config)

        # Format results
        result = {
            "final_response": final_state["responses"][-1] if final_state["responses"] else "",
            "iteration_count": final_state["iteration"],
            "session_id": final_state["session_id"],
            "execution_time": (final_state["end_time"] - final_state["start_time"]).total_seconds(),
            "model_used": final_state["model_used"],
            "cot_analysis": self.cot_analyzer.get_cot_analysis(),
            "stopping_history": [
                {
                    "iteration": i,
                    "should_stop": decision.should_stop,
                    "reason": decision.reason,
                    "confidence": decision.confidence
                }
                for i, decision in enumerate(final_state["stopping_decisions"])
            ],
            "refinement_history": [
                {
                    "iteration": i,
                    "response": response,
                    "cot_confidence": cot.overall_confidence if i < len(final_state["cot_captures"]) else 0.0,
                    "critic_feedback_count": len(feedback) if i < len(final_state["critic_feedback_history"]) else 0
                }
                for i, (response, cot, feedback) in enumerate(zip(
                    final_state["responses"],
                    final_state["cot_captures"] + [None] * len(final_state["responses"]),
                    final_state["critic_feedback_history"] + [None] * len(final_state["responses"])
                ))
            ]
        }

        logger.info(f"SELF-REFINE completed in {result['iteration_count']} iterations")

        return result

    def get_graph_visualization(self) -> str:
        """Get a textual representation of the graph topology."""
        return """
SELF-REFINE with Change-of-Thought Graph Topology:

START
  ↓
initialize
  ↓
generate_initial
  ↓
capture_cot
  ↓
critic_analysis
  ↓
check_stopping
  ↓ ↙
  │ refine_response
  │   ↓
  │ capture_cot (loop back)
  ↓
finalize
  ↓
END

Nodes:
- initialize: Set up session and reset components
- generate_initial: Generate initial response using LLM
- capture_cot: Extract Change-of-Thought reasoning
- critic_analysis: Analyze response and provide feedback
- check_stopping: Apply adaptive stopping criteria
- refine_response: Generate improved response based on feedback
- finalize: Clean up and prepare final results

Conditional Flow:
- check_stopping → continue (refine_response) or stop (finalize)
- The refinement loop continues until stopping criteria are met
"""


# Utility functions for working with the graph

async def run_self_refine(
    prompt: str,
    config_path: Optional[str] = None,
    config_override: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to run SELF-REFINE process.

    Args:
        prompt: User prompt to refine
        config_path: Optional path to configuration file
        config_override: Optional configuration overrides

    Returns:
        Refinement results
    """
    graph = SelfRefineGraph()
    return await graph.refine(prompt, config_override)


def create_self_refine_graph(config: Optional[Dict[str, Any]] = None) -> SelfRefineGraph:
    """
    Factory function to create a SELF-REFINE graph.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured SelfRefineGraph instance
    """
    return SelfRefineGraph(config)
