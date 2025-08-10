"""
Demo: LLM Model Switching

This demo shows how to switch between OpenAI and Gemini models.
Extracted and organized from the original script files.
"""

import asyncio
import logging
from pyrefine.core.llm_clients import LLMClientManager
from pyrefine.graph.self_refine_graph import run_self_refine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_model_switching():
    """Demonstrate switching between OpenAI and Gemini models."""
    
    print("=" * 70)
    print("DEMO: LLM Model Switching (OpenAI ↔ Gemini)")
    print("=" * 70)
    
    prompt = "Explain the concept of artificial intelligence in simple terms."
    
    # Test with OpenAI first
    print("\\n🤖 Testing with OpenAI as primary...")
    openai_config = {
        "llm_client": {
            "primary": "openai",
            "fallback": "gemini"
        },
        "self_refine": {
            "max_iterations": 2,
            "min_iterations": 1
        }
    }
    
    try:
        openai_result = await run_self_refine(prompt, config_override=openai_config)
        print(f"✅ OpenAI Result: {len(openai_result['final_response'])} chars, "
              f"{openai_result['iteration_count']} iterations")
        print(f"Model Used: {openai_result['model_used']}")
        
    except Exception as e:
        print(f"❌ OpenAI failed: {e}")
        openai_result = None
    
    # Test with Gemini
    print("\\n🧠 Testing with Gemini as primary...")
    gemini_config = {
        "llm_client": {
            "primary": "gemini", 
            "fallback": "openai"
        },
        "self_refine": {
            "max_iterations": 2,
            "min_iterations": 1
        }
    }
    
    try:
        gemini_result = await run_self_refine(prompt, config_override=gemini_config)
        print(f"✅ Gemini Result: {len(gemini_result['final_response'])} chars, "
              f"{gemini_result['iteration_count']} iterations")
        print(f"Model Used: {gemini_result['model_used']}")
        
    except Exception as e:
        print(f"❌ Gemini failed: {e}")
        gemini_result = None
    
    # Test client manager directly
    print("\\n🔄 Testing LLM Client Manager...")
    try:
        manager = LLMClientManager()
        
        print(f"Available clients: {manager.get_available_clients()}")
        print(f"Primary client: {manager.primary_client_name}")
        print(f"Fallback client: {manager.fallback_client_name}")
        
        # Test switching
        if "gemini" in manager.get_available_clients():
            print("\\nSwitching to Gemini...")
            manager.switch_primary_client("gemini")
            print(f"New primary client: {manager.primary_client_name}")
        
        if "openai" in manager.get_available_clients():
            print("\\nSwitching back to OpenAI...")
            manager.switch_primary_client("openai")
            print(f"New primary client: {manager.primary_client_name}")
            
    except Exception as e:
        print(f"❌ Client manager test failed: {e}")
    
    # Compare results if both succeeded
    if openai_result and gemini_result:
        print("\\n📊 COMPARISON:")
        print("=" * 50)
        print(f"OpenAI: {len(openai_result['final_response'])} chars, "
              f"{openai_result['execution_time']:.2f}s")
        print(f"Gemini: {len(gemini_result['final_response'])} chars, "
              f"{gemini_result['execution_time']:.2f}s")
        
        # Simple similarity check
        from pyrefine.utils.helpers import calculate_similarity
        similarity = calculate_similarity(
            openai_result['final_response'], 
            gemini_result['final_response']
        )
        print(f"Response Similarity: {similarity:.3f}")
    
    print("\\n" + "=" * 70)
    print("Demo completed! The framework can seamlessly switch between models.")


if __name__ == "__main__":
    asyncio.run(demo_model_switching())