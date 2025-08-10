"""
Configuration loader for SELF-REFINE with Change-of-Thought framework.
"""

import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv


class OpenAIConfig(BaseModel):
    """OpenAI client configuration."""
    model: str = "gpt-4o"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.7
    max_tokens: int = 2048


class GeminiConfig(BaseModel):
    """Gemini client configuration."""
    model: str = "gemini-2.0-flash"
    api_key_env: str = "GEMINI_API_KEY"
    temperature: float = 0.7
    max_tokens: int = 2048


class LLMClientConfig(BaseModel):
    """LLM client configuration."""
    primary: str = "openai"
    fallback: str = "gemini"
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)


class CoTCaptureConfig(BaseModel):
    """Change-of-Thought capture configuration."""
    track_reasoning_changes: bool = True
    confidence_threshold: float = 0.8
    max_cot_tokens: int = 512


class SelfRefineConfig(BaseModel):
    """Self-refine process configuration."""
    max_iterations: int = 4
    min_iterations: int = 1
    enable_cot_capture: bool = True
    cot_capture: CoTCaptureConfig = Field(default_factory=CoTCaptureConfig)


class AdaptiveStoppingConfig(BaseModel):
    """Adaptive stopping configuration."""
    enabled: bool = True
    criteria_type: str = "confidence"
    confidence_threshold: float = 0.9
    consistency_threshold: float = 0.85
    improvement_threshold: float = 0.05


class CriticConfig(BaseModel):
    """Critic module configuration."""
    enabled: bool = True
    use_primary_model: bool = True
    feedback_categories: list[str] = Field(default_factory=lambda: [
        "factual_errors", "logic_gaps", "style_issues", "completeness"
    ])


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "structured"
    output_file: str = "pyrefine.log"


class GraphConfig(BaseModel):
    """Graph configuration."""
    persist_state: bool = True
    state_storage: str = "memory"
    enable_visualization: bool = True


class Config(BaseModel):
    """Main configuration class."""
    llm_client: LLMClientConfig = Field(default_factory=LLMClientConfig)
    self_refine: SelfRefineConfig = Field(default_factory=SelfRefineConfig)
    adaptive_stopping: AdaptiveStoppingConfig = Field(default_factory=AdaptiveStoppingConfig)
    critic: CriticConfig = Field(default_factory=CriticConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file. If None, uses default path.

    Returns:
        Config: Loaded configuration object.
    """
    # Load environment variables
    load_dotenv()

    if config_path is None:
        # Default to config.yaml in the same directory as this file
        config_path = Path(__file__).parent / "config.yaml"

    config_dict = {}

    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f) or {}

    return Config(**config_dict)


def get_api_key(env_var_name: str) -> str:
    """
    Get API key from environment variables.

    Args:
        env_var_name: Name of the environment variable containing the API key.

    Returns:
        str: API key value.

    Raises:
        ValueError: If the API key is not found.
    """
    api_key = os.getenv(env_var_name)
    if not api_key:
        raise ValueError(f"API key not found in environment variable: {env_var_name}")
    return api_key


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
