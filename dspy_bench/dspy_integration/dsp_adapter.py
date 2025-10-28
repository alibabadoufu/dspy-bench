"""Adapter for integrating with DSPy framework."""

import inspect
import os
from typing import Any, Dict, Optional

import dspy
from rich.console import Console

from dspy_bench.config import LLMConfig
from dspy_bench.llm.base_client import BaseLLMClient


class DSPyAdapter:
    """Adapter for configuring DSPy with custom LLM clients."""

    def __init__(self, llm_client: BaseLLMClient, llm_config: LLMConfig):
        """Initialize DSPy adapter.

        Args:
            llm_client: LLM client instance
            llm_config: LLM configuration
        """
        self.llm_client = llm_client
        self.llm_config = llm_config
        self.console = Console()

    def configure_dspy(self) -> None:
        """Configure DSPy settings with the LLM client."""
        # Create a DSPy-compatible LM wrapper
        lm_wrapper = DSPyLMWrapper(self.llm_client)

        # Configure DSPy settings
        dspy.settings.configure(
            lm=lm_wrapper,
            rm=None,  # Use default retrieval model
            experimental=True,
        )

        self.console.print(
            f"✓ Configured DSPy with model: {self.llm_config.model}",
            style="green"
        )

    def get_dspy_version(self) -> str:
        """Get the installed DSPy version.

        Returns:
            DSPy version string
        """
        try:
            import dspy
            return dspy.__version__
        except AttributeError:
            return "unknown"

    def validate_dspy_compatibility(self) -> bool:
        """Validate DSPy compatibility.

        Returns:
            True if compatible, False otherwise
        """
        try:
            # Test basic DSPy functionality
            import dspy

            # Check if we can create a basic signature
            class TestSignature(dspy.Signature):
                input = dspy.InputField()
                output = dspy.OutputField()

            # Check if we can create a basic predictor
            predictor = dspy.Predict(TestSignature)

            # Check version compatibility
            version = self.get_dspy_version()
            if version.startswith("2.") or version.startswith("3."):
                return True
            else:
                self.console.print(
                    f"⚠️  DSPy version {version} may not be fully supported",
                    style="yellow"
                )
                return True  # Assume compatibility but warn

        except Exception as e:
            self.console.print(f"❌ DSPy compatibility check failed: {e}", style="red")
            return False

    def list_available_optimizers(self) -> Dict[str, Dict[str, Any]]:
        """List available DSPy optimizers.

        Returns:
            Dictionary of available optimizers with their info
        """
        optimizers = {}

        try:
            # BootstrapFewShot
            optimizers["bootstrap_fewshot"] = {
                "class": "BootstrapFewShot",
                "description": "Few-shot example selection and prompt augmentation",
                "parameters": ["max_bootstrapped_demos", "max_labeled_demos", "max_rounds"],
            }

            # GEPA (if available)
            try:
                import gepa
                optimizers["gepa"] = {
                    "class": "GEPA",
                    "description": "Generalized Error-driven Prompt Augmentation",
                    "parameters": ["budget", "early_stop", "temperature"],
                }
            except ImportError:
                pass

            # GRPO (if available in DSPy)
            if hasattr(dspy, "GRPO"):
                optimizers["grpo"] = {
                    "class": "GRPO",
                    "description": "Group Relative Policy Optimization",
                    "parameters": ["batch_size", "learning_rate", "max_iterations"],
                }

            # BetterTogether (if available)
            if hasattr(dspy, "BetterTogether"):
                optimizers["better_together"] = {
                    "class": "BetterTogether",
                    "description": "Combined weight & prompt optimization",
                    "parameters": ["max_steps", "learning_rate", "temperature"],
                }

            # Add any other optimizers available in the DSPy installation
            for attr_name in dir(dspy):
                attr = getattr(dspy, attr_name)
                if (
                    inspect.isclass(attr)
                    and hasattr(attr, "optimize")
                    and attr_name not in optimizers
                    and not attr_name.startswith("_")
                ):
                    optimizers[attr_name.lower()] = {
                        "class": attr_name,
                        "description": f"DSPy optimizer: {attr_name}",
                        "parameters": [],  # Would need inspection to determine
                    }

        except Exception as e:
            self.console.print(f"Error listing optimizers: {e}", style="red")

        return optimizers


class DSPyLMWrapper:
    """Wrapper to make our LLM client compatible with DSPy."""

    def __init__(self, llm_client: BaseLLMClient):
        """Initialize LM wrapper.

        Args:
            llm_client: LLM client to wrap
        """
        self.llm_client = llm_client

    def __call__(self, prompt: str, **kwargs) -> str:
        """Call the LLM client.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters

        Returns:
            Generated text response
        """
        # Map DSPy parameters to our client parameters
        client_kwargs = {}

        if "max_tokens" in kwargs:
            client_kwargs["max_tokens"] = kwargs["max_tokens"]
        if "temperature" in kwargs:
            client_kwargs["temperature"] = kwargs["temperature"]

        # Add any other parameters
        for key, value in kwargs.items():
            if key not in ["max_tokens", "temperature"]:
                client_kwargs[key] = value

        response = self.llm_client.generate(prompt, **client_kwargs)
        return response.content

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.

        Returns:
            Model information dictionary
        """
        return self.llm_client.get_model_info()