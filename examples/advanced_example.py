"""
Advanced SELF-REFINE with Change-of-Thought Example

This example demonstrates advanced features and configuration options.
"""

import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyrefine.graph.self_refine_graph import SelfRefineGraph
from pyrefine.config.settings import get_config, set_config, Config
from pyrefine.core.llm_clients import LLMClientManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def advanced_example():
    """Run an advanced example with custom configuration."""

    # Complex prompt requiring multiple refinements
    prompt = """
    Design a comprehensive data science project plan for predicting customer churn 
    in a subscription-based business. Your plan should include:

    1. Problem definition and business impact
    2. Data requirements and collection strategy
    3. Exploratory data analysis approach
    4. Feature engineering considerations
    5. Model selection and evaluation metrics
    6. Implementation and deployment strategy
    7. Monitoring and maintenance plan
    8. Expected timeline and resource requirements

    Make sure to address potential challenges and mitigation strategies throughout.
    """

    print("=" * 80)
    print("SELF-REFINE with Change-of-Thought - Advanced Example")
    print("=" * 80)

    # Custom configuration
    custom_config = {
        "self_refine": {
            "max_iterations": 5,
            "min_iterations": 2,
            "enable_cot_capture": True,
            "cot_capture": {
                "track_reasoning_changes": True,
                "confidence_threshold": 0.7,
                "max_cot_tokens": 1024
            }
        },
        "adaptive_stopping": {
            "enabled": True,
            "criteria_type": "composite",
            "confidence_threshold": 0.9,
            "consistency_threshold": 0.8,
            "improvement_threshold": 0.03
        },
        "critic": {
            "enabled": True,
            "use_primary_model": True,
            "feedback_categories": [
                "factual_errors",
                "logic_gaps", 
                "style_issues",
                "completeness",
                "technical_accuracy"
            ]
        }
    }

    # Create graph with custom config
    graph = SelfRefineGraph(config=custom_config)

    print(f"Prompt: {prompt[:200]}...")
    print("\nStarting refinement process...")
    print("-" * 80)

    # Track start time
    start_time = datetime.now()

    # Run refinement
    try:
        result = await graph.refine(prompt)

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        # Detailed results analysis
        print("\nRESULTS ANALYSIS:")
        print("=" * 80)

        # Final response
        print("FINAL RESPONSE:")
        print("-" * 40)
        final_response = result['final_response']
        print(f"{final_response[:500]}...")
        if len(final_response) > 500:
            print(f"\n[Response truncated. Full length: {len(final_response)} characters]")

        # Performance metrics
        print(f"\nPERFORMANCE METRICS:")
        print("-" * 40)
        print(f"Total Iterations: {result['iteration_count']}")
        print(f"Execution Time: {total_time:.2f} seconds")
        print(f"Model Used: {result['model_used']}")
        print(f"Session ID: {result['session_id']}")

        # Change-of-Thought Analysis
        cot_analysis = result['cot_analysis']
        if cot_analysis:
            print(f"\nCHANGE-OF-THOUGHT ANALYSIS:")
            print("-" * 40)

            # Confidence progression
            confidence_prog = cot_analysis.get('confidence_progression', [])
            print("Confidence Progression:")
            for i, conf in enumerate(confidence_prog):
                print(f"  Iteration {i}: {conf:.3f}")

            # Coherence progression  
            coherence_prog = cot_analysis.get('coherence_progression', [])
            if coherence_prog:
                print("\nCoherence Progression:")
                for i, coh in enumerate(coherence_prog):
                    print(f"  Iteration {i}: {coh:.3f}")

            # Reasoning evolution
            reasoning_evolution = cot_analysis.get('reasoning_evolution', [])
            if reasoning_evolution:
                print("\nReasoning Evolution:")
                for evolution in reasoning_evolution:
                    print(f"  Iteration {evolution['iteration']}: "
                          f"{evolution['num_steps']} steps, "
                          f"type: {evolution['dominant_reasoning_type']}")

        # Refinement history
        print(f"\nREFINEMENT HISTORY:")
        print("-" * 40)
        refinement_history = result['refinement_history']
        for entry in refinement_history:
            print(f"Iteration {entry['iteration']}: "
                  f"CoT confidence: {entry['cot_confidence']:.3f}, "
                  f"Critic feedback items: {entry['critic_feedback_count']}")

        # Stopping decisions
        print(f"\nSTOPPING DECISIONS:")
        print("-" * 40)
        for decision in result['stopping_history']:
            print(f"Iteration {decision['iteration']}: "
                  f"{'STOP' if decision['should_stop'] else 'CONTINUE'} - "
                  f"{decision['reason']} (conf: {decision['confidence']:.3f})")

        # Save detailed results
        results_file = f"advanced_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Prepare JSON-serializable result
        json_result = {
            **result,
            "execution_metadata": {
                "total_execution_time": total_time,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "configuration": custom_config
            }
        }

        # Remove non-serializable objects
        for key in ["cot_analysis"]:
            if key in json_result:
                json_result[key] = str(json_result[key])

        with open(results_file, 'w') as f:
            json.dump(json_result, f, indent=2, default=str)

        print(f"\nDetailed results saved to: {results_file}")

        print("\n" + "=" * 80)
        print("Advanced example completed successfully!")

    except Exception as e:
        logger.error(f"Error in advanced example: {e}")
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


async def test_client_switching():
    """Test LLM client switching functionality."""
    print("\nTESTING CLIENT SWITCHING:")
    print("-" * 40)

    try:
        # Test client manager
        manager = LLMClientManager()

        print(f"Current client: {manager.get_selected_client_name()}")
        print(f"Model: {manager.get_model_name()}")

        # Test availability
        print(f"OpenAI available: {manager.is_client_available('openai')}")
        print(f"Gemini available: {manager.is_client_available('gemini')}")

        # Test switching
        if manager.is_client_available("openai"):
            print("\nTesting client switch to OpenAI...")
            manager.switch_client("openai")
            print(f"New client: {manager.get_selected_client_name()}")

        print("Client switching test completed!")

    except Exception as e:
        print(f"Client switching test failed: {e}")


async def main():
    """Run the advanced example."""
    await advanced_example()
    await test_client_switching()


if __name__ == "__main__":
    asyncio.run(main())
