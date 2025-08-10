"""
LLM Client implementations for OpenAI and Gemini with fallback support.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import logging

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from pyrefine.config.settings import get_config, get_api_key


logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def generate_response(
        self, 
        messages: List[BaseMessage], 
        **kwargs
    ) -> str:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name."""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI client implementation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize OpenAI client."""
        self.config = config or get_config().llm_client.openai.model_dump()

        # Get API key
        api_key = get_api_key(self.config["api_key_env"])

        # Initialize LangChain OpenAI client
        self.client = ChatOpenAI(
            model=self.config["model"],
            openai_api_key=api_key,
            temperature=self.config["temperature"],
            max_tokens=self.config["max_tokens"],
        )

        logger.info(f"Initialized OpenAI client with model: {self.config['model']}")

    async def generate_response(
        self, 
        messages: List[BaseMessage], 
        **kwargs
    ) -> str:
        """Generate response using OpenAI."""
        try:
            response = await self.client.ainvoke(messages, **kwargs)
            return response.content
        except Exception as e:
            logger.error(f"OpenAI client error: {e}")
            raise

    def get_model_name(self) -> str:
        """Get the OpenAI model name."""
        return self.config["model"]


class GeminiClient(BaseLLMClient):
    """Gemini client implementation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Gemini client."""
        self.config = config or get_config().llm_client.gemini.model_dump()

        # Get API key
        api_key = get_api_key(self.config["api_key_env"])

        # Initialize LangChain Gemini client
        self.client = ChatGoogleGenerativeAI(
            model=self.config["model"],
            google_api_key=api_key,
            temperature=self.config["temperature"],
            max_output_tokens=self.config["max_tokens"],
        )

        logger.info(f"Initialized Gemini client with model: {self.config['model']}")

    async def generate_response(
        self, 
        messages: List[BaseMessage], 
        **kwargs
    ) -> str:
        """Generate response using Gemini."""
        try:
            response = await self.client.ainvoke(messages, **kwargs)
            return response.content
        except Exception as e:
            logger.error(f"Gemini client error: {e}")
            raise

    def get_model_name(self) -> str:
        """Get the Gemini model name."""
        return self.config["model"]


class LLMClientManager:
    """
    Manager for LLM clients with fallback support.

    This class manages primary and fallback LLM clients as specified
    in the SELF-REFINE paper requirements.
    """

    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """Initialize LLM client manager."""
        self.config = get_config().llm_client
        if config_override:
            # Override specific config values
            for key, value in config_override.items():
                setattr(self.config, key, value)

        # Initialize clients
        self.clients = {}
        self._init_clients()

        # Set primary and fallback
        self.primary_client_name = self.config.primary
        self.fallback_client_name = self.config.fallback

        logger.info(
            f"LLM Client Manager initialized. "
            f"Primary: {self.primary_client_name}, "
            f"Fallback: {self.fallback_client_name}"
        )

    def _init_clients(self):
        """Initialize all available clients."""
        try:
            self.clients["openai"] = OpenAIClient()
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI client: {e}")

        try:
            self.clients["gemini"] = GeminiClient()
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini client: {e}")

    def get_primary_client(self) -> BaseLLMClient:
        """Get the primary LLM client."""
        client = self.clients.get(self.primary_client_name)
        if not client:
            raise ValueError(f"Primary client '{self.primary_client_name}' not available")
        return client

    def get_fallback_client(self) -> BaseLLMClient:
        """Get the fallback LLM client."""
        client = self.clients.get(self.fallback_client_name)
        if not client:
            raise ValueError(f"Fallback client '{self.fallback_client_name}' not available")
        return client

    async def generate_response(
        self, 
        messages: List[BaseMessage], 
        use_fallback: bool = False,
        **kwargs
    ) -> tuple[str, str]:
        """
        Generate response with automatic fallback support.

        Args:
            messages: List of messages for the conversation
            use_fallback: Whether to use fallback client directly
            **kwargs: Additional arguments for the LLM

        Returns:
            tuple: (response_text, model_used)

        Raises:
            Exception: If both primary and fallback clients fail
        """
        clients_to_try = []

        if use_fallback:
            clients_to_try = [
                (self.fallback_client_name, self.get_fallback_client()),
                (self.primary_client_name, self.get_primary_client())
            ]
        else:
            clients_to_try = [
                (self.primary_client_name, self.get_primary_client()),
                (self.fallback_client_name, self.get_fallback_client())
            ]

        last_exception = None

        for client_name, client in clients_to_try:
            try:
                logger.info(f"Attempting to generate response using {client_name}")
                response = await client.generate_response(messages, **kwargs)
                logger.info(f"Successfully generated response using {client_name}")
                return response, client.get_model_name()
            except Exception as e:
                logger.warning(f"Client {client_name} failed: {e}")
                last_exception = e
                continue

        # If we get here, both clients failed
        error_msg = f"All LLM clients failed. Last error: {last_exception}"
        logger.error(error_msg)
        raise Exception(error_msg)

    def switch_primary_client(self, client_name: str):
        """Switch the primary client."""
        if client_name not in self.clients:
            raise ValueError(f"Client '{client_name}' not available")

        old_primary = self.primary_client_name
        self.primary_client_name = client_name

        logger.info(f"Switched primary client from {old_primary} to {client_name}")

    def get_available_clients(self) -> List[str]:
        """Get list of available client names."""
        return list(self.clients.keys())

    def is_client_available(self, client_name: str) -> bool:
        """Check if a client is available."""
        return client_name in self.clients


# Utility functions for creating messages
def create_system_message(content: str) -> SystemMessage:
    """Create a system message."""
    return SystemMessage(content=content)


def create_human_message(content: str) -> HumanMessage:
    """Create a human message."""
    return HumanMessage(content=content)


def create_ai_message(content: str) -> AIMessage:
    """Create an AI message."""
    return AIMessage(content=content)


def create_conversation(
    system_prompt: str,
    user_message: str,
    ai_responses: Optional[List[str]] = None
) -> List[BaseMessage]:
    """
    Create a conversation with system prompt, user message, and optional AI responses.

    Args:
        system_prompt: System instruction
        user_message: User's input
        ai_responses: Optional list of previous AI responses

    Returns:
        List of BaseMessage objects
    """
    messages = [
        create_system_message(system_prompt),
        create_human_message(user_message)
    ]

    if ai_responses:
        for i, response in enumerate(ai_responses):
            messages.append(create_ai_message(response))
            # Add a placeholder human message if not the last response
            if i < len(ai_responses) - 1:
                messages.append(create_human_message("Continue refining..."))

    return messages
