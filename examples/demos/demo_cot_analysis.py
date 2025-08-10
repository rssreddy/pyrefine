"""
Demo: Change-of-Thought Analysis

This demo shows how the CoT capture module tracks reasoning evolution.
Extracted and organized from the original script files.
"""

import asyncio
import logging
from pyrefine.graph.self_refine_graph import run_self_refine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_cot_analysis():
    """Demonstrate Change-of-Thought analysis capabilities."""
    
    print("=" * 70)
    print("DEMO: Change-of-Thought (CoT) Analysis")
    print("=" * 70)
    
    # Complex prompt that should show reasoning evolution
    prompt = """
    Analyze the ethical implications of artificial intelligence in healthcare.
    Consider multiple perspectives:
    1. Patient privacy and data security
    2. Diagnostic accuracy vs human oversight
    3. Healthcare accessibility and equity
    4. Economic impact on healthcare workers
    5. Long-term societal implications
    
    Provide a balanced analysis with concrete recommendations.
    """
    
    # Configuration to enable detailed CoT capture
    config = {
        "self_refine": {
            "max_iterations": 4,
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
            "criteria_type": "composite"  # Use multiple criteria
        }
    }
    
    print(f"Prompt: {prompt[:150]}...")
    print("CoT Tracking: ENABLED")
    print("=" * 70)
    
    try:
        result = await run_self_refine(prompt, config_override=config)
        
        print("\\nCHANGE-OF-THOUGHT ANALYSIS:")
        print("=" * 70)
        
        # Detailed CoT analysis
        cot_analysis = result.get('cot_analysis', {})
        
        if cot_analysis:
            print(f"Total Iterations: {cot_analysis.get('total_iterations', 0)}")
            
            # Confidence progression
            confidence_progression = cot_analysis.get('confidence_progression', [])
            if confidence_progression:
                print("\\nConfidence Evolution:")
                for i, conf in enumerate(confidence_progression):
                    trend = ""
                    if i > 0:
                        diff = conf - confidence_progression[i-1]
                        if diff > 0.05:
                            trend = "📈 +{:.3f}".format(diff)
                        elif diff < -0.05:
                            trend = "📉 {:.3f}".format(diff)
                        else:
                            trend = "➡️ {:.3f}".format(diff)
                    
                    print(f"  Iteration {i}: {conf:.3f} {trend}")
            
            # Coherence progression
            coherence_progression = cot_analysis.get('coherence_progression', [])
            if coherence_progression:
                print("\\nCoherence Evolution:")
                for i, coh in enumerate(coherence_progression):
                    print(f"  Iteration {i}: {coh:.3f}")
            
            # Reasoning evolution
            reasoning_evolution = cot_analysis.get('reasoning_evolution', [])
            if reasoning_evolution:
                print("\\nReasoning Type Evolution:")
                for evolution in reasoning_evolution:
                    print(f"  Iteration {evolution['iteration']}: "
                          f"{evolution['num_steps']} steps, "
                          f"dominant type: {evolution['dominant_reasoning_type']}")
            
            # Thought changes summary
            thought_changes = cot_analysis.get('thought_changes_summary', [])
            if thought_changes:
                print("\\nThought Changes Detected:")
                for i, change in enumerate(thought_changes[:5]):  # Show first 5
                    print(f"  {i+1}. {change}")
                
                if len(thought_changes) > 5:
                    print(f"  ... and {len(thought_changes) - 5} more changes")
        
        # Show refinement progression
        print("\\nREFINEMENT PROGRESSION:")
        print("=" * 50)
        refinement_history = result.get('refinement_history', [])
        for entry in refinement_history:
            print(f"Iteration {entry['iteration']}:")
            print(f"  Response Length: {len(entry['response'])} chars")
            print(f"  CoT Confidence: {entry['cot_confidence']:.3f}")
            print(f"  Critic Feedback: {entry['critic_feedback_count']} items")
            print()
        
        print("FINAL METRICS:")
        print("=" * 50)
        print(f"Total Iterations: {result['iteration_count']}")
        print(f"Execution Time: {result['execution_time']:.2f} seconds")
        print(f"Final Response Length: {len(result['final_response'])} characters")
        
        # Show final response preview
        print("\\nFINAL RESPONSE PREVIEW:")
        print("-" * 50)
        preview = result['final_response'][:300]
        print(f"{preview}...")
        
        print("\\n" + "=" * 70)
        print("Demo completed! Notice how reasoning evolves and confidence builds over iterations.")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(demo_cot_analysis())