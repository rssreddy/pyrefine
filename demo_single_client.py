#!/usr/bin/env python3
"""
Demo: Single Client Configuration (No Fallback)

This demonstrates the new single client configuration where you specify
either 'openai' or 'gemini' at startup, with no fallback behavior.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

async def demo_gemini_configuration():
    """Demo using Gemini configuration."""
    
    print("🧠 Demo: Using Gemini Configuration")
    print("=" * 50)
    
    try:
        from pyrefine.graph.self_refine_graph import run_self_refine
        
        # Configure to use Gemini explicitly
        config = {
            "llm_client": {
                "selected": "gemini"  # Only Gemini will be used
            },
            "self_refine": {
                "max_iterations": 1,  # Keep it simple for demo
                "min_iterations": 1
            }
        }
        
        prompt = "What is artificial intelligence in one sentence?"
        
        print(f"Prompt: {prompt}")
        print("Configuration: Using Gemini only (no fallback)")
        print("\nRunning refinement...")
        
        result = await run_self_refine(prompt, config_override=config)
        
        print(f"\n✅ Success!")
        print(f"Model used: {result['model_used']}")
        print(f"Response: {result['final_response']}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        if "quota" in str(e).lower():
            print("💡 This is expected if you've hit API quota limits")
        return False

async def demo_openai_configuration():
    """Demo using OpenAI configuration."""
    
    print("\n🤖 Demo: Using OpenAI Configuration")
    print("=" * 50)
    
    try:
        from pyrefine.graph.self_refine_graph import run_self_refine
        
        # Configure to use OpenAI explicitly
        config = {
            "llm_client": {
                "selected": "openai"  # Only OpenAI will be used
            },
            "self_refine": {
                "max_iterations": 1,
                "min_iterations": 1
            }
        }
        
        prompt = "What is machine learning in one sentence?"
        
        print(f"Prompt: {prompt}")
        print("Configuration: Using OpenAI only (no fallback)")
        print("\nRunning refinement...")
        
        result = await run_self_refine(prompt, config_override=config)
        
        print(f"\n✅ Success!")
        print(f"Model used: {result['model_used']}")
        print(f"Response: {result['final_response']}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        if "api key" in str(e).lower():
            print("💡 This is expected if you don't have a valid OpenAI API key")
        return False

def demo_client_manager():
    """Demo the LLM Client Manager directly."""
    
    print("\n⚙️ Demo: LLM Client Manager")
    print("=" * 40)
    
    try:
        from pyrefine.core.llm_clients import LLMClientManager
        
        # Test 1: Default configuration
        print("1. Default Configuration:")
        manager = LLMClientManager()
        print(f"   Selected client: {manager.get_selected_client_name()}")
        print(f"   Model: {manager.get_model_name()}")
        
        # Test 2: Explicit configuration
        print("\n2. Explicit Gemini Configuration:")
        gemini_manager = LLMClientManager(config_override={
            "llm_client": {"selected": "gemini"}
        })
        print(f"   Selected client: {gemini_manager.get_selected_client_name()}")
        print(f"   Model: {gemini_manager.get_model_name()}")
        
        # Test 3: Client availability check
        print("\n3. Client Availability:")
        print(f"   OpenAI available: {manager.is_client_available('openai')}")
        print(f"   Gemini available: {manager.is_client_available('gemini')}")
        
        # Test 4: Client switching
        print("\n4. Client Switching:")
        original_client = manager.get_selected_client_name()
        print(f"   Original: {original_client}")
        
        try:
            new_client = "openai" if original_client == "gemini" else "gemini"
            manager.switch_client(new_client)
            print(f"   Switched to: {manager.get_selected_client_name()}")
            
            # Switch back
            manager.switch_client(original_client)
            print(f"   Switched back to: {manager.get_selected_client_name()}")
            
        except Exception as e:
            print(f"   Switch failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"�� Client manager demo failed: {e}")
        return False

def show_configuration_guide():
    """Show how to configure the single client setup."""
    
    print("\n📚 Single Client Configuration Guide")
    print("=" * 60)
    
    print("\n🔧 Method 1: YAML Configuration (config.yaml)")
    print("-" * 45)
    print("""
# Edit pyrefine/config/config.yaml
llm_client:
  selected: 'gemini'  # or 'openai'
  
  openai:
    model: 'gpt-4o'
    api_key_env: 'OPENAI_API_KEY'
    temperature: 0.7
    max_tokens: 2048
    
  gemini:
    model: 'gemini-2.0-flash'
    api_key_env: 'GEMINI_API_KEY'
    temperature: 0.7
    max_tokens: 2048
""")
    
    print("\n💻 Method 2: Programmatic Configuration")
    print("-" * 42)
    print("""
# Use Gemini
config = {
    "llm_client": {"selected": "gemini"}
}
result = await run_self_refine(prompt, config_override=config)

# Use OpenAI  
config = {
    "llm_client": {"selected": "openai"}
}
result = await run_self_refine(prompt, config_override=config)
""")
    
    print("\n🔄 Method 3: Runtime Client Switching")
    print("-" * 38)
    print("""
from pyrefine.core.llm_clients import LLMClientManager

manager = LLMClientManager()
print(f"Current: {manager.get_selected_client_name()}")

# Switch to different client
manager.switch_client("openai")  # or "gemini"
print(f"Switched to: {manager.get_selected_client_name()}")
""")
    
    print("\n✨ Benefits of Single Client Configuration:")
    print("   • No unexpected fallbacks")
    print("   • Cleaner error handling") 
    print("   • Predictable behavior")
    print("   • Easier debugging")
    print("   • Simpler configuration")
    print("   • Clear cost control")

async def main():
    """Main demo function."""
    
    print("🚀 PyRefine Single Client Configuration Demo")
    print("=" * 70)
    
    print("\n🎯 Overview:")
    print("This demo shows the new single client configuration where you")
    print("specify either 'openai' or 'gemini' at startup with no fallback.")
    
    # Demo client manager
    manager_success = demo_client_manager()
    
    # Demo configurations (may fail due to API limits/keys)
    print("\n" + "=" * 70)
    print("API TESTS (may fail due to quota/key limits):")
    
    gemini_success = await demo_gemini_configuration()
    openai_success = await demo_openai_configuration()
    
    # Show configuration guide
    show_configuration_guide()
    
    print("\n" + "=" * 70)
    print("📊 Demo Results:")
    print(f"   Client Manager: {'✅ Success' if manager_success else '❌ Failed'}")
    print(f"   Gemini Test: {'✅ Success' if gemini_success else '❌ Failed (expected if quota exceeded)'}")
    print(f"   OpenAI Test: {'✅ Success' if openai_success else '❌ Failed (expected if no API key)'}")
    
    if manager_success:
        print("\n🎉 Single client configuration is working!")
        print("\n🚀 Ready to use:")
        print("   1. Set your preferred client in config.yaml")
        print("   2. Or use config_override in your code")
        print("   3. No more unexpected fallbacks!")
    else:
        print("\n⚠️  Some issues detected. Check your setup.")

if __name__ == "__main__":
    asyncio.run(main())