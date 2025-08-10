"""
Basic SELF-REFINE with Change-of-Thought Example

This example demonstrates the basic usage of the framework.
"""

import asyncio
import logging
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyrefine.graph.self_refine_graph import run_self_refine
from pyrefine.config.settings import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Run a basic example of SELF-REFINE."""

    # Example prompt
    prompt = """
    Explain the concept of machine learning to a beginner. Include:
    1. What machine learning is
    2. How it differs from traditional programming
    3. Common types of machine learning
    4. Real-world applications
    5. Benefits and limitations

    Make it accessible and engaging for someone with no technical background.
    """

    print("=" * 60)
    print("SELF-REFINE with Change-of-Thought - Basic Example")
    print("=" * 60)
    print(f"Original Prompt: {prompt}")
    print("=" * 60)

    try:
        # Run SELF-REFINE process
        result = await run_self_refine(
            prompt=prompt,
            config_override={
                "self_refine": {
                    "max_iterations": 3,
                    "min_iterations": 1
                },
                "adaptive_stopping": {
                    "criteria_type": "confidence",
                    "confidence_threshold": 0.85
                }
            }
        )

        # Display results
        print("\nRESULTS:")
        print("=" * 60)
        print(f"Final Response:\n{result['final_response']}")
        print("=" * 60)
        print(f"Iterations: {result['iteration_count']}")
        print(f"Execution Time: {result['execution_time']:.2f} seconds")
        print(f"Model Used: {result['model_used']}")
        print("=" * 60)

        # Display CoT analysis
        cot_analysis = result['cot_analysis']
        if cot_analysis:
            print("\nCHANGE-OF-THOUGHT ANALYSIS:")
            print("-" * 40)
            print(f"Total Iterations: {cot_analysis.get('total_iterations', 0)}")

            confidence_progression = cot_analysis.get('confidence_progression', [])
            if confidence_progression:
                print("Confidence Progression:")
                for i, conf in enumerate(confidence_progression):
                    print(f"  Iteration {i}: {conf:.3f}")

            thought_changes = cot_analysis.get('thought_changes_summary', [])
            if thought_changes:
                print("\nThought Changes:")
                for change in thought_changes[:5]:  # Show first 5
                    print(f"  - {change}")

        # Display stopping history
        print("\nSTOPPING DECISIONS:")
        print("-" * 40)
        for decision in result['stopping_history']:
            print(f"Iteration {decision['iteration']}: {decision['reason']} "
                  f"(confidence: {decision['confidence']:.3f})")

        print("\n" + "=" * 60)
        print("Example completed successfully!")

    except Exception as e:
        logger.error(f"Error running example: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
