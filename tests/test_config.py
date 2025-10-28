"""Tests for configuration management."""

import pytest
import tempfile
from pathlib import Path

from dspy_bench.config import Config, DatasetConfig, LLMConfig, DSPyConfig


class TestConfig:
    """Test configuration classes."""

    def test_dataset_config_validation(self):
        """Test dataset configuration validation."""
        # Valid config
        config = DatasetConfig(
            path="test.jsonl",
            input_key="input",
            label_key="label",
            split={"train": 0.7, "val": 0.15, "test": 0.15}
        )
        assert config.input_key == "input"
        assert config.label_key == "label"

        # Invalid split ratios
        with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
            DatasetConfig(
                path="test.jsonl",
                split={"train": 0.5, "val": 0.3, "test": 0.3}  # Sum = 1.1
            )

    def test_llm_config_get_api_key(self):
        """Test LLM configuration API key retrieval."""
        import os

        # Test with existing environment variable
        os.environ["TEST_API_KEY"] = "test_key"
        config = LLMConfig(api_key_env="TEST_API_KEY")
        assert config.get_api_key() == "test_key"

        # Test with missing environment variable
        config = LLMConfig(api_key_env="MISSING_KEY")
        with pytest.raises(ValueError, match="API key not found"):
            config.get_api_key()

        # Clean up
        del os.environ["TEST_API_KEY"]

    def test_config_yaml_roundtrip(self):
        """Test configuration YAML save/load."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"

            # Create config
            config = Config(
                experiment_name="test_experiment",
                seed=42,
                dataset=DatasetConfig(
                    path="data.jsonl",
                    input_key="input",
                    label_key="label"
                ),
                dspy=DSPyConfig(
                    seed_prompt="Test prompt"
                ),
                llm=LLMConfig(model="gpt-3.5-turbo"),
                optimizers=[],
            )

            # Save and reload
            config.to_yaml(config_path)
            loaded_config = Config.from_yaml(config_path)

            assert loaded_config.experiment_name == config.experiment_name
            assert loaded_config.seed == config.seed
            assert loaded_config.dataset.input_key == config.dataset.input_key

    def test_get_experiment_id(self):
        """Test experiment ID generation."""
        config = Config(
            experiment_name="test_experiment",
            seed=42,
            dataset=DatasetConfig(path="test.jsonl"),
            dspy=DSPyConfig(seed_prompt="test"),
            optimizers=[],
        )

        experiment_id = config.get_experiment_id()
        assert experiment_id.startswith("exp_")
        assert len(experiment_id) > 10  # Should be substantial length