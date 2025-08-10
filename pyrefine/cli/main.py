"""
Command-line interface for SELF-REFINE with Change-of-Thought framework.

This consolidates CLI functionality from various script files.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from ..graph.self_refine_graph import run_self_refine
from ..config.settings import get_config, load_config
from ..utils.logging_config import setup_logging, get_logger
from ..utils.helpers import format_execution_time, parse_iteration_results


logger = get_logger(__name__)


def create_cli_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="SELF-REFINE with Change-of-Thought Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  self-refine "Explain quantum computing to a beginner"
  
  # With custom configuration
  self-refine "Complex prompt here" --config custom_config.yaml
  
  # Interactive mode
  self-refine --interactive
  
  # Save results to file
  self-refine "Prompt" --output results.json
  
  # Verbose logging
  self-refine "Prompt" --verbose
        """
    )
    
    # Main arguments
    parser.add_argument(
        "prompt",
        nargs="?",
        help="The prompt to refine (required unless using --interactive)"
    )
    
    # Configuration options
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file (default: config/config.yaml)"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        help="Maximum number of refinement iterations"
    )
    
    parser.add_argument(
        "--min-iterations", 
        type=int,
        help="Minimum number of refinement iterations"
    )
    
    parser.add_argument(
        "--model",
        choices=["openai", "gemini"],
        help="Primary LLM model to use"
    )
    
    parser.add_argument(
        "--stopping-criteria",
        choices=["confidence", "consistency", "improvement", "critic", "composite"],
        help="Stopping criteria type"
    )
    
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        help="Confidence threshold for stopping (0.0-1.0)"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path for results (JSON format)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true", 
        help="Suppress output except errors"
    )
    
    # Interactive mode
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    # Utility commands
    parser.add_argument(
        "--version",
        action="version",
        version="SELF-REFINE with Change-of-Thought v1.0.0"
    )
    
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Show current configuration and exit"
    )
    
    parser.add_argument(
        "--create-flowchart",
        action="store_true",
        help="Create LangGraph flowchart visualization and exit"
    )
    
    return parser


async def run_refinement(
    prompt: str,
    config_override: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
    quiet: bool = False
) -> Dict[str, Any]:
    """
    Run the refinement process with the given prompt.
    
    Args:
        prompt: User prompt to refine
        config_override: Configuration overrides
        verbose: Enable verbose output
        quiet: Suppress output
        
    Returns:
        Refinement results
    """
    if not quiet:
        print("🚀 Starting SELF-REFINE with Change-of-Thought...")
        print(f"📝 Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        print("=" * 60)
    
    try:
        # Run refinement
        result = await run_self_refine(prompt, config_override=config_override)
        
        if not quiet:
            # Display results
            print("✅ REFINEMENT COMPLETE")
            print("=" * 60)
            
            # Parse and display summary
            summary = parse_iteration_results(result)
            
            print(f"📊 Summary:")
            print(f"   Iterations: {summary['total_iterations']}")
            print(f"   Execution Time: {summary['execution_time']}")
            print(f"   Model Used: {summary['model_used']}")
            
            if 'final_confidence' in summary:
                print(f"   Final Confidence: {summary['final_confidence']:.3f}")
            
            if 'stopping_reason' in summary:
                print(f"   Stopping Reason: {summary['stopping_reason']}")
            
            print("\n📄 Final Response:")
            print("-" * 40)
            print(result['final_response'])
            
            if verbose:
                print("\n🔍 Detailed Analysis:")
                print("-" * 40)
                
                # Show iteration history
                refinement_history = result.get('refinement_history', [])
                for entry in refinement_history:
                    print(f"Iteration {entry['iteration']}: "
                          f"CoT confidence: {entry['cot_confidence']:.3f}")
                
                # Show stopping decisions
                stopping_history = result.get('stopping_history', [])
                if stopping_history:
                    print("\nStopping Decisions:")
                    for decision in stopping_history:
                        status = "STOP" if decision['should_stop'] else "CONTINUE"
                        print(f"  {decision['iteration']}: {status} - {decision['reason']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Refinement failed: {e}")
        if not quiet:
            print(f"❌ Error: {e}")
        raise


def interactive_mode():
    """Run in interactive mode."""
    print("🎯 SELF-REFINE Interactive Mode")
    print("Type 'quit' or 'exit' to stop, 'help' for commands")
    print("=" * 50)
    
    while True:
        try:
            prompt = input("\n📝 Enter your prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif prompt.lower() == 'help':
                print("\nCommands:")
                print("  help - Show this help")
                print("  quit/exit/q - Exit interactive mode")
                print("  config - Show current configuration")
                print("  Any other text - Use as refinement prompt")
                continue
            elif prompt.lower() == 'config':
                config = get_config()
                print(f"\nCurrent Configuration:")
                print(f"  Primary Model: {config.llm_client.primary}")
                print(f"  Max Iterations: {config.self_refine.max_iterations}")
                print(f"  Stopping Criteria: {config.adaptive_stopping.criteria_type}")
                continue
            elif not prompt:
                print("Please enter a prompt or command.")
                continue
            
            # Run refinement
            result = asyncio.run(run_refinement(prompt, quiet=False))
            
            # Ask if user wants to save results
            save = input("\n💾 Save results to file? (y/N): ").strip().lower()
            if save in ['y', 'yes']:
                filename = input("📁 Enter filename (default: results.json): ").strip()
                if not filename:
                    filename = "results.json"
                
                with open(filename, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"✅ Results saved to {filename}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def show_configuration():
    """Display current configuration."""
    config = get_config()
    
    print("📋 Current Configuration:")
    print("=" * 40)
    print(f"LLM Client:")
    print(f"  Primary: {config.llm_client.primary}")
    print(f"  Fallback: {config.llm_client.fallback}")
    print(f"  OpenAI Model: {config.llm_client.openai.model}")
    print(f"  Gemini Model: {config.llm_client.gemini.model}")
    
    print(f"\nSelf-Refine:")
    print(f"  Max Iterations: {config.self_refine.max_iterations}")
    print(f"  Min Iterations: {config.self_refine.min_iterations}")
    print(f"  CoT Capture: {config.self_refine.enable_cot_capture}")
    
    print(f"\nAdaptive Stopping:")
    print(f"  Enabled: {config.adaptive_stopping.enabled}")
    print(f"  Criteria: {config.adaptive_stopping.criteria_type}")
    print(f"  Confidence Threshold: {config.adaptive_stopping.confidence_threshold}")
    
    print(f"\nCritic:")
    print(f"  Enabled: {config.critic.enabled}")
    print(f"  Categories: {', '.join(config.critic.feedback_categories)}")


def create_flowchart():
    """Create and save the LangGraph flowchart."""
    try:
        from ..utils.visualization import create_flowchart_visualization
        
        output_path = "langgraph_flowchart.png"
        create_flowchart_visualization(output_path)
        print(f"✅ Flowchart saved to {output_path}")
        
    except ImportError:
        print("❌ Visualization dependencies not available. Install with: pip install plotly kaleido")
    except Exception as e:
        print(f"❌ Error creating flowchart: {e}")


async def main():
    """Main CLI entry point."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Set up logging
    log_level = "DEBUG" if args.verbose else "WARNING" if args.quiet else "INFO"
    setup_logging(level=log_level)
    
    # Handle utility commands
    if args.show_config:
        show_configuration()
        return
    
    if args.create_flowchart:
        create_flowchart()
        return
    
    # Load configuration
    config_override = {}
    
    if args.config:
        try:
            from ..config.settings import load_config
            custom_config = load_config(args.config)
            # Convert to dict for override
            config_override = custom_config.model_dump()
        except Exception as e:
            print(f"❌ Error loading config file: {e}")
            sys.exit(1)
    
    # Apply CLI overrides
    if args.max_iterations:
        config_override.setdefault('self_refine', {})['max_iterations'] = args.max_iterations
    
    if args.min_iterations:
        config_override.setdefault('self_refine', {})['min_iterations'] = args.min_iterations
    
    if args.model:
        config_override.setdefault('llm_client', {})['primary'] = args.model
    
    if args.stopping_criteria:
        config_override.setdefault('adaptive_stopping', {})['criteria_type'] = args.stopping_criteria
    
    if args.confidence_threshold:
        config_override.setdefault('adaptive_stopping', {})['confidence_threshold'] = args.confidence_threshold
    
    # Handle interactive mode
    if args.interactive:
        interactive_mode()
        return
    
    # Validate prompt
    if not args.prompt:
        print("❌ Error: Prompt is required unless using --interactive mode")
        parser.print_help()
        sys.exit(1)
    
    try:
        # Run refinement
        result = await run_refinement(
            args.prompt,
            config_override=config_override,
            verbose=args.verbose,
            quiet=args.quiet
        )
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            if not args.quiet:
                print(f"\n💾 Results saved to {args.output}")
        
    except Exception as e:
        logger.error(f"CLI execution failed: {e}")
        if not args.quiet:
            print(f"❌ Fatal error: {e}")
        sys.exit(1)


def cli_entry_point():
    """Entry point for console script."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_entry_point()