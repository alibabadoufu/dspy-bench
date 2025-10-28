"""Configuration management for dspy-bench experiments."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class DatasetConfig(BaseModel):
    """Configuration for dataset loading and splitting."""

    path: Union[str, Path] = Field(..., description="Path to JSONL dataset file")
    input_key: str = Field(default="input", description="Key for input field in JSONL")
    label_key: str = Field(default="label", description="Key for label field in JSONL")
    id_key: Optional[str] = Field(default=None, description="Optional key for example ID")
    split: Dict[str, float] = Field(
        default={"train": 0.7, "val": 0.15, "test": 0.15},
        description="Train/val/test split ratios"
    )

    @validator("split")
    def validate_split(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate that split ratios sum to 1.0."""
        total = sum(v.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        return v


class LLMConfig(BaseModel):
    """Configuration for LLM client."""

    host: Optional[str] = Field(default=None, description="OpenAI-compatible API host")
    api_key_env: str = Field(default="OPENAI_API_KEY", description="Environment variable for API key")
    model: str = Field(default="gpt-3.5-turbo", description="Model name")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    timeout: int = Field(default=60, description="Request timeout in seconds")
    max_concurrency: int = Field(default=5, description="Maximum concurrent requests")

    def get_api_key(self) -> str:
        """Get API key from environment."""
        key = os.getenv(self.api_key_env)
        if key is None:
            raise ValueError(f"API key not found in environment variable {self.api_key_env}")
        return key


class DSPyConfig(BaseModel):
    """Configuration for DSPy program and seed prompt."""

    seed_prompt: str = Field(..., description="Seed prompt template")
    instruction: Optional[str] = Field(default=None, description="Instruction for the task")
    max_tokens: int = Field(default=1024, description="Maximum tokens for generation")
    temperature: float = Field(default=0.0, description="Temperature for generation")


class OptimizerConfig(BaseModel):
    """Configuration for a single optimizer."""

    name: str = Field(..., description="Optimizer name")
    config: Dict[str, Any] = Field(default_factory=dict, description="Optimizer-specific config")


class OutputConfig(BaseModel):
    """Configuration for output and artifacts."""

    experiments_dir: Union[str, Path] = Field(default="experiments", description="Experiments output directory")
    save_checkpoints: bool = Field(default=True, description="Save optimizer checkpoints")
    save_predictions: bool = Field(default=True, description="Save predictions on test set")


class MetricsConfig(BaseModel):
    """Configuration for evaluation metrics."""

    metrics: List[str] = Field(default=["accuracy"], description="List of metrics to compute")
    custom_metrics: Dict[str, str] = Field(default_factory=dict, description="Custom metric functions")


class Config(BaseModel):
    """Main configuration for dspy-bench experiments."""

    experiment_name: str = Field(..., description="Name of the experiment")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    dataset: DatasetConfig = Field(..., description="Dataset configuration")
    dspy: DSPyConfig = Field(..., description="DSPy configuration")
    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM configuration")
    optimizers: List[OptimizerConfig] = Field(..., description="List of optimizers to run")
    output: OutputConfig = Field(default_factory=OutputConfig, description="Output configuration")
    metrics: MetricsConfig = Field(default_factory=MetricsConfig, description="Metrics configuration")

    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Forbid extra fields to catch typos

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file."""
        import yaml

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        import yaml

        with open(yaml_path, "w") as f:
            yaml.dump(self.dict(), f, default_flow_style=False, indent=2)

    def get_experiment_id(self) -> str:
        """Generate a unique experiment ID based on timestamp and name."""
        from datetime import datetime
        import hashlib

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_hash = hashlib.md5(self.experiment_name.encode()).hexdigest()[:6]
        return f"exp_{timestamp}_{name_hash}"