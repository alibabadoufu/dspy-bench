"""Base client interface for LLM interactions."""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel


class LLMResponse(BaseModel):
    """Response from LLM API."""

    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    latency: float
    finish_reason: Optional[str] = None
    request_id: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(
        self,
        model: str,
        max_retries: int = 3,
        timeout: int = 60,
        max_concurrency: int = 5,
    ):
        """Initialize client.

        Args:
            model: Model name
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
            max_concurrency: Maximum concurrent requests
        """
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_concurrency = max_concurrency

    @abstractmethod
    async def async_generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response asynchronously.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional model-specific parameters

        Returns:
            LLM response
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response synchronously.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional model-specific parameters

        Returns:
            LLM response
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model.

        Returns:
            Dictionary with model information
        """
        pass

    def health_check(self) -> bool:
        """Check if the client is healthy and can make requests.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Simple test prompt
            response = self.generate("Hello", max_tokens=10)
            return bool(response.content)
        except Exception:
            return False