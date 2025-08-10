"""
LLM Client implementations for OpenAI and Gemini with single client selection.
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
    Manager for LLM clients with single client selection.
    
    This class initializes and manages only the selected LLM client
    without fallback support for cleaner and more predictable behavior.
    """

    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """Initialize LLM client manager."""
        self.config = get_config().llm_client
        
        # Apply config overrides if provided
        if config_override:
            if 'llm_client' in config_override:
                llm_config = config_override['llm_client']
                if 'selected' in llm_config:
                    self.selected_client_name = llm_config['selected']
                else:
                    self.selected_client_name = self.config.selected
            else:
                self.selected_client_name = self.config.selected
        else:
            self.selected_client_name = self.config.selected

        # Initialize only the selected client
        self.client = self._init_selected_client()
        
        logger.info(f"LLM Client Manager initialized with selected client: {self.selected_client_name}")

    def _init_selected_client(self) -> BaseLLMClient:
        """Initialize only the selected client."""
        if self.selected_client_name == "openai":
            try:
                return OpenAIClient()
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                raise ValueError(f"Cannot initialize OpenAI client: {e}")
        
        elif self.selected_client_name == "gemini":
            try:
                return GeminiClient()
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
                raise ValueError(f"Cannot initialize Gemini client: {e}")
        
        else:
            raise ValueError(f"Unknown client type: {self.selected_client_name}. Must be 'openai' or 'gemini'")

    def get_client(self) -> BaseLLMClient:
        """Get the selected LLM client."""
        return self.client

    async def generate_response(
        self, 
        messages: List[BaseMessage], 
        **kwargs
    ) -> tuple[str, str]:
        """
        Generate response using the selected client.

        Args:
            messages: List of messages for the conversation
            **kwargs: Additional arguments for the LLM

        Returns:
            tuple: (response_text, model_used)

        Raises:
            Exception: If the selected client fails
        """
        try:
            logger.info(f"Generating response using {self.selected_client_name}")
            response = await self.client.generate_response(messages, **kwargs)
            model_name = self.client.get_model_name()
            logger.info(f"Successfully generated response using {self.selected_client_name}")
            return response, model_name
        except Exception as e:
            error_msg = f"Client {self.selected_client_name} failed: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def get_selected_client_name(self) -> str:
        """Get the name of the selected client."""
        return self.selected_client_name

    def get_model_name(self) -> str:
        """Get the model name of the selected client."""
        return self.client.get_model_name()

    def switch_client(self, client_name: str):
        """
        Switch to a different client.
        
        Args:
            client_name: Name of the client to switch to ('openai' or 'gemini')
        """
        if client_name not in ['openai', 'gemini']:
            raise ValueError(f"Invalid client name: {client_name}. Must be 'openai' or 'gemini'")
        
        old_client = self.selected_client_name
        self.selected_client_name = client_name
        
        try:
            self.client = self._init_selected_client()
            logger.info(f"Switched client from {old_client} to {client_name}")
        except Exception as e:
            # Revert on failure
            self.selected_client_name = old_client
            logger.error(f"Failed to switch to {client_name}, reverted to {old_client}: {e}")
            raise

    def is_client_available(self, client_name: str) -> bool:
        """
        Check if a client is available by trying to initialize it.
        
        Args:
            client_name: Name of the client to check
            
        Returns:
            bool: True if the client can be initialized
        """
        try:
            if client_name == "openai":
                OpenAIClient()
            elif client_name == "gemini":
                GeminiClient()
            else:
                return False
            return True
        except Exception:
            return False


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