"""OpenAI-compatible LLM client with retry logic and rate limiting."""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union

import aiohttp
import openai
from openai import AsyncOpenAI, OpenAI
from rich.console import Console

from dspy_bench.llm.base_client import BaseLLMClient, LLMResponse


class OpenAICompatClient(BaseLLMClient):
    """OpenAI-compatible LLM client with retry logic and rate limiting."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 60,
        max_concurrency: int = 5,
        organization: Optional[str] = None,
    ):
        """Initialize OpenAI-compatible client.

        Args:
            model: Model name
            api_key: API key
            base_url: Base URL for API (optional, defaults to OpenAI)
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
            max_concurrency: Maximum concurrent requests
            organization: OpenAI organization (optional)
        """
        super().__init__(model, max_retries, timeout, max_concurrency)
        self.api_key = api_key
        self.base_url = base_url
        self.organization = organization

        # Initialize clients
        client_kwargs = {
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        if organization:
            client_kwargs["organization"] = organization

        self._client = OpenAI(**client_kwargs)
        self._async_client = AsyncOpenAI(**client_kwargs)

        # Rate limiting
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._console = Console()

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
            **kwargs: Additional parameters

        Returns:
            LLM response
        """
        return asyncio.run(self.async_generate(prompt, max_tokens, temperature, **kwargs))

    async def async_generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response asynchronously with rate limiting.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            LLM response
        """
        async with self._semaphore:
            return await self._generate_with_retry(prompt, max_tokens, temperature, **kwargs)

    async def _generate_with_retry(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate with retry logic."""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()

                # Prepare request parameters
                request_params = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                }

                if max_tokens:
                    request_params["max_tokens"] = max_tokens

                # Add additional parameters
                for key, value in kwargs.items():
                    if key not in request_params:
                        request_params[key] = value

                # Make API call
                response = await self._async_client.chat.completions.create(**request_params)

                latency = time.time() - start_time

                # Extract response
                content = response.choices[0].message.content or ""
                finish_reason = response.choices[0].finish_reason
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                }

                return LLMResponse(
                    content=content,
                    model=response.model,
                    usage=usage,
                    latency=latency,
                    finish_reason=finish_reason,
                    request_id=response.id,
                    raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
                )

            except openai.RateLimitError as e:
                last_exception = e
                if attempt < self.max_retries:
                    # Exponential backoff for rate limits
                    wait_time = 2 ** attempt + 1
                    self._console.print(
                        f"Rate limit hit, waiting {wait_time}s... (attempt {attempt + 1}/{self.max_retries + 1})",
                        style="yellow"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self._console.print(f"Rate limit error after {self.max_retries} retries", style="red")

            except openai.APITimeoutError as e:
                last_exception = e
                if attempt < self.max_retries:
                    self._console.print(
                        f"Timeout, retrying... (attempt {attempt + 1}/{self.max_retries + 1})",
                        style="yellow"
                    )
                    await asyncio.sleep(1)

            except openai.APIError as e:
                last_exception = e
                if attempt < self.max_retries:
                    self._console.print(
                        f"API error: {e}, retrying... (attempt {attempt + 1}/{self.max_retries + 1})",
                        style="yellow"
                    )
                    await asyncio.sleep(1)

            except Exception as e:
                last_exception = e
                self._console.print(f"Unexpected error: {e}", style="red")
                break

        # All retries exhausted
        raise RuntimeError(f"Failed to generate response after {self.max_retries} retries") from last_exception

    async def batch_generate(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> List[LLMResponse]:
        """Generate responses for multiple prompts concurrently.

        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            List of LLM responses
        """
        tasks = [
            self.async_generate(prompt, max_tokens, temperature, **kwargs)
            for prompt in prompts
        ]

        return await asyncio.gather(*tasks, return_exceptions=False)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model.

        Returns:
            Dictionary with model information
        """
        return {
            "model": self.model,
            "base_url": self.base_url or "https://api.openai.com/v1",
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "max_concurrency": self.max_concurrency,
            "organization": self.organization,
        }

    def __del__(self):
        """Cleanup clients."""
        try:
            if hasattr(self, '_client'):
                self._client.close()
            if hasattr(self, '_async_client'):
                # Note: AsyncOpenAI doesn't have a close method in older versions
                pass
        except Exception:
            pass