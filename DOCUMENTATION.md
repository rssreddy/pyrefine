# PyRefine Framework

A Python implementation of the SELF-REFINE iterative refinement engine enhanced with Change-of-Thought (CoT) capture, built using LangGraph for orchestration and supporting both OpenAI ChatGPT and Google Gemini models.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation & Setup](#installation--setup)
4. [Quick Start](#quick-start)
5. [Configuration](#configuration)
6. [Architecture](#architecture)
7. [Core Components](#core-components)
8. [Usage Examples](#usage-examples)
9. [API Reference](#api-reference)
10. [Advanced Features](#advanced-features)
11. [Testing](#testing)
12. [Troubleshooting](#troubleshooting)
13. [Project Structure](#project-structure)

---

## Overview

The SELF-REFINE with Change-of-Thought framework is an advanced iterative refinement system that enhances Large Language Model (LLM) responses through multiple rounds of self-improvement. Unlike traditional approaches, this implementation focuses on **Change-of-Thought (CoT) capture** rather than simple performance monitoring, providing deep insights into reasoning evolution.

### Key Features

- ✅ **Iterative Self-Refinement**: Based on the SELF-REFINE paper methodology
- ✅ **Change-of-Thought Capture**: Replaces performance monitoring with reasoning evolution tracking  
- ✅ **Adaptive Stopping Criteria**: Multiple intelligent stopping mechanisms
- ✅ **Single LLM Client Selection**: Choose between OpenAI and Gemini (no fallback)
- ✅ **LangGraph Orchestration**: Robust DAG-based workflow management
- ✅ **Structured Critic Feedback**: JSON-formatted analysis and suggestions
- ✅ **Comprehensive Logging**: Detailed process tracking and analysis

---

## Installation & Setup

### Prerequisites

- Python 3.10 or higher
- At least one API key (OpenAI or Gemini)

### Install Dependencies

```bash
cd /path/to/project
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the project root:

```env
# OpenAI Configuration (optional)
OPENAI_API_KEY=your_openai_api_key_here

# Gemini Configuration (optional)
GEMINI_API_KEY=your_gemini_api_key_here
```

**Note**: You need at least one valid API key. The framework uses single client selection with no fallback behavior.

---

## Quick Start

### Method 1: Basic Usage

```python
import asyncio
from pyrefine.graph.self_refine_graph import run_self_refine

async def main():
    prompt = "Explain quantum computing to a beginner."

    result = await run_self_refine(prompt)

    print(f"Final Response: {result['final_response']}")
    print(f"Iterations: {result['iteration_count']}")
    print(f"Execution Time: {result['execution_time']:.2f}s")

asyncio.run(main())
```

### Method 2: With Configuration

```python
from pyrefine.graph.self_refine_graph import SelfRefineGraph

# Custom configuration
config = {
    "llm_client": {
        "selected": "gemini"  # Single client selection
    },
    "self_refine": {
        "max_iterations": 5,
        "enable_cot_capture": True
    },
    "adaptive_stopping": {
        "criteria_type": "composite",
        "confidence_threshold": 0.9
    }
}

graph = SelfRefineGraph(config=config)
result = await graph.refine("Your complex prompt here")
```

### Method 3: Running Examples

```bash
# Basic example
python examples/basic_example.py

# Advanced example with detailed analysis
python examples/advanced_example.py

# Demo scripts
python examples/demos/demo_confidence_stopping.py
python examples/demos/demo_cot_analysis.py
python examples/demos/demo_model_switching.py
```

### Method 4: Quick Test

```bash
# Test the framework
python demo_single_client.py
```

---

## Configuration

### Single Client Configuration (Recommended)

The framework uses **single client configuration** with no fallback behavior for predictable execution and cost control.

#### YAML Configuration

Edit `pyrefine/config/config.yaml`:

```yaml
# LLM Client Configuration
llm_client:
  selected: 'gemini'  # or 'openai' - ONLY this client will be used

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

# Self-Refine Parameters
self_refine:
  max_iterations: 4
  min_iterations: 1
  enable_cot_capture: true

# Adaptive Stopping
adaptive_stopping:
  enabled: true
  criteria_type: 'confidence'  # or 'consistency', 'improvement', 'critic', 'composite'
  confidence_threshold: 0.9

# Critic Configuration
critic:
  enabled: true
  feedback_categories:
    - "factual_errors"
    - "logic_gaps"
    - "style_issues"
    - "completeness"
```

#### Programmatic Configuration

```python
from pyrefine.config.settings import Config, set_config

config = Config(
    llm_client=LLMClientConfig(selected="gemini"),
    self_refine=SelfRefineConfig(max_iterations=6),
    adaptive_stopping=AdaptiveStoppingConfig(criteria_type="composite")
)

set_config(config)
```

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    SELF-REFINE Framework                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ LangGraph   │  │ LLM Client  │  │ Config      │        │
│  │ DAG         │  │ Manager     │  │ System      │        │
│  │ Orchestrator│  │ (Single)    │  │ (YAML/Code) │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├────────────────────────────────────���────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Change-of-  │  │ Critic      │  │ Adaptive    │        │
│  │ Thought     │  │ Module      │  │ Stopping    │        │
│  │ Analyzer    │  │ (JSON)      │  │ Manager     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### Graph Topology

```
START → initialize → generate_initial → capture_cot → critic_analysis → 
check_stopping → [continue: refine_response → capture_cot] OR [stop: finalize] → END
```

---

## Core Components

### 1. LLM Client Manager (`core/llm_clients.py`)

Manages single LLM client selection with no fallback behavior.

```python
class LLMClientManager:
    def __init__(self, config_override: Optional[Dict[str, Any]] = None)
    async def generate_response(self, messages: List[BaseMessage]) -> tuple[str, str]
    def switch_client(self, client_name: str)
    def get_selected_client_name(self) -> str
```

**Benefits of Single Client Configuration:**
- 🎯 **Predictable Behavior**: No unexpected fallbacks
- 🐛 **Easier Debugging**: Clear error messages when client fails
- 💰 **Cost Control**: Explicit control over which API is used
- ⚙️ **Simpler Configuration**: Single `selected` field

### 2. Change-of-Thought Capture (`core/cot_capture.py`)

Tracks reasoning evolution throughout the refinement process.

**Key Features:**
- **Reasoning Step Extraction**: Identifies individual reasoning steps
- **Confidence Tracking**: Monitors confidence levels across iterations  
- **Thought Evolution**: Tracks changes in reasoning approach
- **Coherence Analysis**: Measures logical flow between steps

```python
@dataclass
class CoTCapture:
    iteration: int
    reasoning_steps: List[ReasoningStep]
    overall_confidence: float
    reasoning_coherence: float
    thought_changes: List[str]
    metadata: Dict[str, Any]
```

### 3. Critic Module (`core/critic.py`)

Provides structured feedback analysis with JSON output format.

**Feedback Categories:**
- **Factual Errors**: Numerical accuracy, date consistency, claim verification
- **Logic Gaps**: Logical flow, assumption validation, conclusion support
- **Style Issues**: Clarity, structure, tone, technical jargon
- **Completeness**: Topic coverage, missing aspects, depth analysis

**JSON Output Format:**
```json
{
  "feedback": "Detailed analysis of the response with specific issues identified",
  "category": "factual_errors",
  "severity": "medium",
  "suggestion": "Specific recommendation for improvement"
}
```

### 4. Adaptive Stopping (`core/adaptive_stopping.py`)

Multiple intelligent stopping mechanisms:

#### Confidence-Based Stopping
```yaml
adaptive_stopping:
  criteria_type: 'confidence'
  confidence_threshold: 0.9  # Stop when confidence ≥ 90%
```

#### Consistency-Based Stopping
```yaml
adaptive_stopping:
  criteria_type: 'consistency'
  consistency_threshold: 0.85  # Stop when consistency ≥ 85%
  window_size: 2  # Compare last 2 responses
```

#### Improvement-Based Stopping
```yaml
adaptive_stopping:
  criteria_type: 'improvement'
  improvement_threshold: 0.05  # Stop when improvement < 5%
  window_size: 2  # Analyze last 2 iterations
```

#### Composite Stopping
```yaml
adaptive_stopping:
  criteria_type: 'composite'
  confidence_threshold: 0.9
  consistency_threshold: 0.85
  improvement_threshold: 0.05
```

### 5. LangGraph Orchestration (`graph/self_refine_graph.py`)

DAG-based workflow management with:
- State management
- Conditional execution
- Memory persistence
- Robust error handling

---

## Usage Examples

### Basic Refinement

```python
import asyncio
from pyrefine import run_self_refine

async def basic_example():
    prompt = "Explain the benefits of renewable energy"
    
    result = await run_self_refine(prompt)
    
    print(f"Final Response: {result['final_response']}")
    print(f"Iterations: {result['iteration_count']}")
    print(f"Model Used: {result['model_used']}")

asyncio.run(basic_example())
```

### Advanced Configuration

```python
async def advanced_example():
    prompt = "Design a sustainable transportation system for a city"
    
    config = {
        "llm_client": {"selected": "gemini"},
        "self_refine": {"max_iterations": 4},
        "adaptive_stopping": {
            "criteria_type": "composite",
            "confidence_threshold": 0.95,
            "consistency_threshold": 0.9
        }
    }
    
    result = await run_self_refine(prompt, config_override=config)
    
    # Access detailed results
    print(f"Confidence progression: {result['cot_analysis']['confidence_progression']}")
    print(f"Thought changes: {result['cot_analysis']['thought_changes_summary']}")

asyncio.run(advanced_example())
```

### Batch Processing

```python
async def batch_process():
    prompts = [
        "Explain climate change",
        "Describe machine learning",
        "What is blockchain?"
    ]
    
    config = {"llm_client": {"selected": "gemini"}}
    
    results = []
    for prompt in prompts:
        result = await run_self_refine(prompt, config_override=config)
        results.append(result)
    
    return results

results = asyncio.run(batch_process())
```

### Jupyter Notebook Usage

```python
import asyncio
import sys
sys.path.append('/path/to/project')

from pyrefine import run_self_refine

# Configure for single client
config = {
    "llm_client": {"selected": "gemini"}
}

# In Jupyter, use this pattern:
result = await run_self_refine("Explain quantum computing", config_override=config)
print(result['final_response'])
```

---

## API Reference

### Core Functions

#### `run_self_refine()`

Main function for executing the SELF-REFINE process.

```python
async def run_self_refine(
    prompt: str,
    config_path: Optional[str] = None,
    config_override: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute the SELF-REFINE process with Change-of-Thought capture.
    
    Args:
        prompt: The user prompt to refine
        config_path: Optional path to configuration file
        config_override: Optional configuration overrides
    
    Returns:
        Dict containing:
        - final_response: The refined response
        - iteration_count: Number of iterations performed
        - execution_time: Total execution time in seconds
        - model_used: LLM model used for generation
        - cot_analysis: Change-of-Thought analysis results
        - stopping_history: Stopping decision history
        - refinement_history: Detailed iteration history
        - session_id: Unique session identifier
    """
```

### Core Classes

#### `SelfRefineGraph`

Main orchestration class using LangGraph.

```python
class SelfRefineGraph:
    def __init__(self, config: Optional[Dict[str, Any]] = None)
    async def refine(self, prompt: str, config_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]
    def get_graph_visualization(self) -> str
```

#### `LLMClientManager`

Manages LLM client selection and communication.

```python  
class LLMClientManager:
    def __init__(self, config_override: Optional[Dict[str, Any]] = None)
    async def generate_response(self, messages: List[BaseMessage]) -> tuple[str, str]
    def switch_client(self, client_name: str)
    def get_selected_client_name(self) -> str
```

#### `ChangeOfThoughtAnalyzer`

Captures and analyzes reasoning evolution.

```python
class ChangeOfThoughtAnalyzer:
    def __init__(self, config: Optional[Dict[str, Any]] = None)
    def capture_cot(self, response_text: str, iteration: int) -> CoTCapture
    def get_cot_analysis(self) -> Dict[str, Any]
    def should_continue_based_on_cot(self) -> bool
```

#### `Critic`

Provides structured feedback analysis.

```python
class Critic:
    def __init__(self, llm_manager: LLMClientManager, config: Optional[Dict[str, Any]] = None)
    async def analyze_response(self, original_prompt: str, response: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]
    def parse_structured_feedback(self, feedback_json: Dict[str, Any]) -> List[CriticFeedback]
```

---

## Advanced Features

### CoT Evolution Visualization

```python
async def visualize_cot_evolution():
    result = await run_self_refine("Complex reasoning task")
    
    # Analyze CoT evolution
    cot_analysis = result['cot_analysis']
    print(f"Confidence progression: {cot_analysis['confidence_progression']}")
    print(f"Thought changes: {cot_analysis['thought_changes_summary']}")
```

### Custom Stopping Criteria

```python
from pyrefine.core.adaptive_stopping import BaseStoppingCriterion

class CustomStoppingCriterion(BaseStoppingCriterion):
    def should_stop(self, iteration, responses, cot_captures, critic_feedback, **kwargs):
        # Implement custom logic
        if iteration >= 2:
            latest_cot = cot_captures[-1]
            if latest_cot.overall_confidence > 0.8:
                return True, "Custom threshold exceeded"
        return False, "Custom criterion not met"
```

### Runtime Client Switching

```python
from pyrefine.core.llm_clients import LLMClientManager

# Initialize with specific client
manager = LLMClientManager(config_override={
    "llm_client": {"selected": "gemini"}
})

print(f"Current client: {manager.get_selected_client_name()}")

# Switch to different client
manager.switch_client("openai")
print(f"Switched to: {manager.get_selected_client_name()}")
```

---

## Testing

```bash
# Run tests
pytest tests/

# Run specific test modules
pytest tests/test_core.py
pytest tests/test_graph.py

# Test single client configuration
python demo_single_client.py
```

### Quick Test Script

```python
# quick_test.py
import sys
import os
sys.path.append('/path/to/project')

def test_imports():
    try:
        from pyrefine import SelfRefineGraph, run_self_refine
        from pyrefine.core import LLMClientManager
        from pyrefine.config import get_config
        print("✅ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_api_keys():
    from dotenv import load_dotenv
    load_dotenv()
    
    openai_key = os.getenv('OPENAI_API_KEY')
    gemini_key = os.getenv('GEMINI_API_KEY')
    
    if openai_key:
        print("✅ OpenAI API key found")
    if gemini_key:
        print("✅ Gemini API key found")
    
    if not openai_key and not gemini_key:
        print("⚠️  No API keys found. Please set up .env file")
        return False
    
    return True

if __name__ == "__main__":
    print("🧪 Testing SELF-REFINE Framework Setup...")
    print("=" * 50)
    
    all_good = True
    all_good &= test_imports()
    all_good &= test_api_keys()
    
    print("=" * 50)
    if all_good:
        print("🎉 Setup complete! Framework ready to use.")
    else:
        print("❌ Setup issues detected. Please check the errors above.")
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Solution: Add to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/project"

# Or install in development mode
pip install -e .
```

#### 2. API Key Issues
```bash
# Check if API keys are set
python -c "import os; print('OpenAI:', bool(os.getenv('OPENAI_API_KEY'))); print('Gemini:', bool(os.getenv('GEMINI_API_KEY')))"

# Load environment variables manually
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('OPENAI_API_KEY'))"
```

#### 3. Client Initialization Failed
```
ValueError: Cannot initialize OpenAI client: API key not found
```
**Solution**: Set the correct API key in your `.env` file

#### 4. Invalid Client Selection
```
ValueError: Unknown client type: invalid_name. Must be 'openai' or 'gemini'
```
**Solution**: Use only 'openai' or 'gemini' in the `selected` field

#### 5. Missing Dependencies
```bash
pip install langgraph langchain langchain-openai langchain-google-genai
pip install openai google-genai pydantic pyyaml python-dotenv numpy
```

### Migration from Old Configuration

If you were using the old fallback system:

#### Old Configuration (No Longer Works):
```yaml
llm_client:
  primary: 'openai'
  fallback: 'gemini'
```

#### New Configuration:
```yaml
llm_client:
  selected: 'gemini'  # Choose your preferred client
```

---

## Project Structure

```
pyrefine/
├── config/                 # Configuration management
│   ├── config.yaml        # Default configuration
│   └── settings.py        # Configuration loader
├── core/                  # Core components
│   ├── llm_clients.py     # LLM client implementations
│   ├── cot_capture.py     # Change-of-Thought capture
│   ├── critic.py          # Critic feedback module
│   ├── adaptive_stopping.py # Stopping criteria
│   └── state_manager.py   # State management
├── graph/                 # LangGraph orchestration
│   ├── self_refine_graph.py # Main DAG implementation
│   ├── nodes.py           # Individual graph nodes
│   └── edges.py           # Graph edge definitions
├── examples/              # Usage examples
│   ├── basic_example.py   # Simple usage
│   ├── advanced_example.py # Advanced features
│   └── demos/             # Demo scripts
├── tests/                 # Test suite
└── utils/                 # Utility functions
```

---

## Key Features Explained

### Change-of-Thought (CoT) Capture

Replaces the original SELF-REFINE performance monitoring with reasoning evolution tracking:

- **Reasoning Step Extraction**: Identifies individual reasoning steps
- **Confidence Tracking**: Monitors confidence levels across iterations  
- **Thought Evolution**: Tracks changes in reasoning approach
- **Coherence Analysis**: Measures logical flow between steps

### Adaptive Stopping Criteria

Multiple intelligent stopping mechanisms:

- **Confidence-based**: Stop when confidence exceeds threshold
- **Consistency-based**: Stop when responses become consistent
- **Improvement-based**: Stop when improvement rate drops
- **Critic-based**: Stop when critic issues are resolved
- **Composite**: Weighted combination of multiple criteria

### Single LLM Client Selection

Predictable client management:

- Configuration-based client selection
- No fallback behavior for predictable execution
- Runtime client switching capability
- Clear error handling when client fails

---

## Example Use Cases

### 1. Academic Writing with Gemini
```python
config = {"llm_client": {"selected": "gemini"}}
prompt = "Write a research proposal on renewable energy"
result = await run_self_refine(prompt, config_override=config)
```

### 2. Technical Documentation with OpenAI
```python
config = {"llm_client": {"selected": "openai"}}
prompt = "Explain how to implement a REST API in Python"
result = await run_self_refine(prompt, config_override=config)
```

### 3. Creative Writing with Multiple Iterations
```python
config = {
    "llm_client": {"selected": "gemini"},
    "self_refine": {"max_iterations": 5}  # More iterations for creativity
}
prompt = "Write a short story about time travel"
result = await run_self_refine(prompt, config_override=config)
```

### 4. Problem Solving with Cost Control
```python
# Use Gemini for cost-effective problem solving
config = {"llm_client": {"selected": "gemini"}}
prompt = "How can we reduce plastic waste in oceans?"
result = await run_self_refine(prompt, config_override=config)
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## References

- [SELF-REFINE: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)

---

## Acknowledgments

- Original SELF-REFINE paper authors
- LangChain/LangGraph development team
- OpenAI and Google AI teams for model APIs

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Success!

Your SELF-REFINE with Change-of-Thought framework features:

- ✅ **Single Client Configuration**: No more fallback behavior
- ✅ **Predictable Execution**: Only your selected client is used
- ✅ **Clean Error Handling**: Clear failures instead of hidden fallbacks
- ✅ **Cost Control**: Explicit control over API usage
- ✅ **Easy Debugging**: Know exactly which model is running

The framework successfully:
- Processes prompts through multiple refinement iterations
- Captures and analyzes reasoning evolution (Change-of-Thought)
- Provides structured critic feedback
- Uses adaptive stopping criteria
- Supports both OpenAI and Gemini models with single client selection
- Maintains comprehensive logging and analysis

**Ready to refine with confidence!** 🚀