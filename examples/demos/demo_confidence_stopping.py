"""
Demo: Confidence-based Stopping Criteria

This demo shows how confidence-based stopping works in the SELF-REFINE framework.
Extracted and organized from the original script files.
"""

import asyncio
import logging
from pyrefine.graph.self_refine_graph import run_self_refine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_confidence_stopping():
    """Demonstrate confidence-based stopping criteria."""
    
    print("=" * 70)
    print("DEMO: Confidence-Based Stopping Criteria")
    print("=" * 70)
    
    # Prompt that should build confidence over iterations
    prompt = """
    Write a comprehensive guide on how to start a small business. Include:
    1. Business planning fundamentals
    2. Legal requirements and registration
    3. Financial planning and funding options
    4. Marketing strategies for new businesses
    5. Common pitfalls to avoid
    
    Make it practical and actionable for first-time entrepreneurs.
    """
    
    # Configuration for confidence-based stopping
    config = {
        "self_refine": {
            "max_iterations": 4,
            "min_iterations": 1
        },
        "adaptive_stopping": {
            "enabled": True,
            "criteria_type": "confidence",
            "confidence_threshold": 0.85  # Stop when confidence reaches 85%
        },
        "critic": {
            "enabled": True
        }
    }
    
    print(f"Prompt: {prompt[:150]}...")
    print(f"Confidence Threshold: {config['adaptive_stopping']['confidence_threshold']}")
    print("=" * 70)
    
    try:
        result = await run_self_refine(prompt, config_override=config)
        
        print("\\nRESULTS:")
        print("=" * 70)
        print(f"Final Response Length: {len(result['final_response'])} characters")
        print(f"Total Iterations: {result['iteration_count']}")
        print(f"Execution Time: {result['execution_time']:.2f} seconds")
        
        # Show confidence progression
        cot_analysis = result.get('cot_analysis', {})
        confidence_progression = cot_analysis.get('confidence_progression', [])
        
        if confidence_progression:
            print("\\nConfidence Progression:")
            for i, conf in enumerate(confidence_progression):
                status = "✅ STOPPED" if i == len(confidence_progression) - 1 and conf >= 0.85 else ""
                print(f"  Iteration {i}: {conf:.3f} {status}")
        
        # Show stopping decisions
        print("\\nStopping Decisions:")
        for decision in result['stopping_history']:
            status = "🛑 STOP" if decision['should_stop'] else "▶️ CONTINUE"
            print(f"  Iteration {decision['iteration']}: {status} - {decision['reason']}")
        
        print("\\n" + "=" * 70)
        print("Demo completed! Notice how the process stops when confidence threshold is reached.")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(demo_confidence_stopping())