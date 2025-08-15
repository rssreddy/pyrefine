# PyRefine Framework

A Python implementation of the SELF-REFINE iterative refinement engine enhanced with Change-of-Thought (CoT) capture, built using LangGraph for orchestration and supporting both OpenAI ChatGPT and Google Gemini models.

## Features

- ✅ **Iterative Self-Refinement**: Based on the SELF-REFINE paper methodology
- ✅ **Change-of-Thought Capture**: Tracks reasoning evolution across iterations
- ✅ **Adaptive Stopping Criteria**: Multiple intelligent stopping mechanisms
- ✅ **Single LLM Client Selection**: Choose between OpenAI and Gemini
- ✅ **LangGraph Orchestration**: Robust DAG-based workflow management
- ✅ **Structured Critic Feedback**: JSON-formatted analysis and suggestions

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file:

```env
# At least one API key required
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### Basic Usage

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

### With Configuration

```python
# Custom configuration
config = {
    "llm_client": {
        "selected": "gemini"  # or "openai"
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

result = await run_self_refine("Your prompt here", config_override=config)
```

## Configuration

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
  criteria_type: 'confidence'  # 'consistency', 'improvement', 'critic', 'composite'
  confidence_threshold: 0.9
```

## Core Components

### 1. Change-of-Thought Capture
Tracks reasoning evolution throughout refinement:
- **Reasoning Step Extraction**: Identifies individual reasoning steps
- **Confidence Tracking**: Monitors confidence levels across iterations
- **Thought Evolution**: Tracks changes in reasoning approach
- **Coherence Analysis**: Measures logical flow between steps

### 2. Critic Module
Provides structured feedback in JSON format:
- **Factual Errors**: Accuracy and consistency checks
- **Logic Gaps**: Reasoning flow analysis
- **Style Issues**: Clarity and structure assessment
- **Completeness**: Coverage and depth evaluation

### 3. Adaptive Stopping
Multiple intelligent stopping mechanisms:
- **Confidence-based**: Stop when confidence exceeds threshold
- **Consistency-based**: Stop when responses become consistent
- **Improvement-based**: Stop when improvement rate drops
- **Composite**: Weighted combination of criteria

## Examples

### Basic Refinement
```python
result = await run_self_refine("Explain renewable energy benefits")
```

### Advanced Configuration
```python
config = {
    "llm_client": {"selected": "gemini"},
    "self_refine": {"max_iterations": 4},
    "adaptive_stopping": {
        "criteria_type": "composite",
        "confidence_threshold": 0.95
    }
}

result = await run_self_refine("Design a sustainable city", config_override=config)
```

### Batch Processing
```python
prompts = ["Explain AI", "Describe blockchain", "What is quantum computing?"]
results = []

for prompt in prompts:
    result = await run_self_refine(prompt, config_override={"llm_client": {"selected": "gemini"}})
    results.append(result)
```

## Testing

```bash
# Run tests
pytest tests/

# Quick test
python demo_single_client.py
```

## Project Structure

```
pyrefine/
├── config/                 # Configuration management
├── core/                   # Core components (LLM clients, CoT, critic)
├── graph/                  # LangGraph orchestration
├── examples/               # Usage examples
├── tests/                  # Test suite
└── utils/                  # Utility functions
```

## Key Features

### Single Client Selection
- **Predictable Behavior**: No unexpected fallbacks
- **Cost Control**: Explicit control over API usage
- **Easier Debugging**: Clear error messages
- **Simpler Configuration**: Single `selected` field

### Change-of-Thought vs Performance Monitoring
Unlike traditional approaches, this framework focuses on **reasoning evolution** rather than simple metrics, providing insights into how the model's thinking changes across iterations.

## Troubleshooting

### Common Issues

1. **Import Errors**: Add project to Python path
   ```bash
   export PYTHONPATH="${PYTHONPATH}:/path/to/project"
   ```

2. **API Key Issues**: Check environment variables
   ```bash
   python -c "import os; print('OpenAI:', bool(os.getenv('OPENAI_API_KEY')))"
   ```

3. **Client Initialization**: Ensure valid API key in `.env` file

## References

- [SELF-REFINE: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Ready to refine with confidence!** 🚀

For detailed documentation, see [DOCUMENTATION.md](DOCUMENTATION.md)