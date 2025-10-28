"""Mock LLM client for testing purposes."""

import random
import time
from typing import Any, Dict, List, Optional

from dspy_bench.llm.base_client import BaseLLMClient, LLMResponse


class MockLLMClient(BaseLLMClient):
    """Mock LLM client that returns deterministic responses for testing."""

    def __init__(
        self,
        model: str = "mock-model",
        max_retries: int = 3,
        timeout: int = 60,
        max_concurrency: int = 5,
        response_delay: float = 0.1,
        deterministic: bool = True,
        seed: int = 42,
    ):
        """Initialize mock client.

        Args:
            model: Model name
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
            max_concurrency: Maximum concurrent requests
            response_delay: Simulated response delay in seconds
            deterministic: Whether to return deterministic responses
            seed: Random seed for reproducible responses
        """
        super().__init__(model, max_retries, timeout, max_concurrency)
        self.response_delay = response_delay
        self.deterministic = deterministic
        self.seed = seed

        if deterministic:
            random.seed(seed)

        # Predefined responses for common prompts
        self._responses = {
            "hello": "Hello! How can I help you today?",
            "translate": "This is a mock translation response.",
            "summarize": "This is a mock summary of the provided text.",
            "default": "This is a mock response from the testing LLM client.",
        }

        self._call_count = 0

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate mock response."""
        self._call_count += 1

        # Simulate network delay
        time.sleep(self.response_delay)

        # Generate deterministic or random response
        if self.deterministic:
            content = self._get_deterministic_response(prompt, self._call_count)
        else:
            content = self._get_random_response(prompt)

        # Simulate token usage
        prompt_tokens = len(prompt.split())
        completion_tokens = len(content.split())

        return LLMResponse(
            content=content,
            model=self.model,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            latency=self.response_delay,
            finish_reason="stop",
            request_id=f"mock_req_{self._call_count}",
        )

    async def async_generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate mock response asynchronously."""
        # For mock client, just call the synchronous version
        return self.generate(prompt, max_tokens, temperature, **kwargs)

    def _get_deterministic_response(self, prompt: str, call_count: int) -> str:
        """Get deterministic response based on prompt and call count."""
        prompt_lower = prompt.lower().strip()

        # Check for predefined responses
        for key, response in self._responses.items():
            if key in prompt_lower:
                if key == "default":
                    return f"{response} (Call #{call_count})"
                else:
                    return f"{response} (Call #{call_count})"

        # Generate response based on prompt hash
        prompt_hash = hash(prompt.lower())
        base_response = f"Mock response for prompt hash {abs(prompt_hash) % 10000}"
        return f"{base_response} (Call #{call_count})"

    def _get_random_response(self, prompt: str) -> str:
        """Get random response."""
        responses = [
            "This is a randomly generated mock response.",
            "Here's another mock response for testing.",
            "Mock LLM response with random content.",
            "Testing response from the mock client.",
            "Random mock output for the given prompt.",
        ]
        return random.choice(responses)

    def get_model_info(self) -> Dict[str, Any]:
        """Get mock model information."""
        return {
            "model": self.model,
            "type": "mock",
            "deterministic": self.deterministic,
            "response_delay": self.response_delay,
            "seed": self.seed,
            "call_count": self._call_count,
        }

    def reset(self) -> None:
        """Reset the mock client state."""
        self._call_count = 0
        if self.deterministic:
            random.seed(self.seed)