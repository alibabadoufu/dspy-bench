"""Integration tests for dspy-bench."""

import json
import pytest
import tempfile
from pathlib import Path

from dspy_bench.config import Config
from dspy_bench.data import JSONLDataset
from dspy_bench.llm import MockLLMClient
from dspy_bench.optimizers import MockOptimizer
from dspy_bench.runner import Experiment


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Test end-to-end workflow with mock components."""

    def test_complete_experiment_workflow(self):
        """Test complete experiment workflow from config to results."""
        # Create test dataset
        test_data = [
            {"input": "What is 2+2?", "label": "4"},
            {"input": "What is 3+3?", "label": "6"},
            {"input": "What is 4+4?", "label": "8"},
            {"input": "What is 5+5?", "label": "10"},
            {"input": "What is 6+6?", "label": "12"},
        ]

        # Create temporary files
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Save dataset
            dataset_path = tmp_path / "test_data.jsonl"
            with dataset_path.open('w') as f:
                for item in test_data:
                    f.write(json.dumps(item) + '\n')

            # Create configuration
            config = Config(
                experiment_name="integration_test",
                seed=42,
                dataset={
                    "path": str(dataset_path),
                    "input_key": "input",
                    "label_key": "label",
                    "split": {"train": 0.6, "val": 0.2, "test": 0.2},
                },
                dspy={
                    "seed_prompt": "Solve this math problem: {input}\nAnswer:",
                    "max_tokens": 10,
                    "temperature": 0.0,
                },
                llm={
                    "model": "mock-model",
                    "max_retries": 1,
                    "timeout": 10,
                },
                optimizers=[
                    {"name": "mock_optimizer", "config": {"improvement_rate": 1.0, "max_improvements": 2}},
                ],
                metrics={
                    "metrics": ["accuracy", "exact_match"],
                },
            )

            # Create experiment with mock LLM
            llm_client = MockLLMClient(deterministic=True, seed=42)
            experiment_dir = tmp_path / "experiments"
            experiment = Experiment(config, llm_client, experiment_dir)

            # Load and split dataset
            dataset = JSONLDataset.load(str(dataset_path), "input", "label")
            from dspy_bench.data import SplitsManager

            splits_manager = SplitsManager(dataset)
            splits = splits_manager.create_splits(seed=42, **config.dataset.split)

            # Prepare experiment
            experiment.prepare(
                train_split=splits["train"],
                val_split=splits["val"],
                test_split=splits["test"],
            )

            # Run optimization
            results = experiment.run_all_optimizers()
            assert len(results) == 1
            assert "mock_optimizer" in results

            # Check optimization result
            result = results["mock_optimizer"]
            assert result.success is True
            assert result.original_prompt == config.dspy.seed_prompt
            assert result.final_prompt is not None
            assert "accuracy" in result.metrics

            # Evaluate results
            eval_results = experiment.evaluate_results()
            assert len(eval_results) == 1
            assert "mock_optimizer" in eval_results

            # Finalize experiment
            summary = experiment.finalize()
            assert "experiment_id" in summary
            assert "optimizers" in summary
            assert "evaluation" in summary

            # Check experiment directory structure
            assert experiment_dir.exists()
            assert (experiment_dir / "config.yaml").exists()
            assert (experiment_dir / "summary.json").exists()
            assert (experiment_dir / "logs.log").exists()
            assert (experiment_dir / "splits").exists()
            assert (experiment_dir / "optimizers").exists()
            assert (experiment_dir / "evaluation").exists()
            assert (experiment_dir / "README.md").exists()

    def test_experiment_with_failing_optimizer(self):
        """Test experiment handling of failing optimizer."""
        # Create minimal dataset
        test_data = [
            {"input": "test input 1", "label": "output 1"},
            {"input": "test input 2", "label": "output 2"},
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Save dataset
            dataset_path = tmp_path / "test_data.jsonl"
            with dataset_path.open('w') as f:
                for item in test_data:
                    f.write(json.dumps(item) + '\n')

            # Create configuration with non-existent optimizer
            config = Config(
                experiment_name="failing_test",
                seed=42,
                dataset={
                    "path": str(dataset_path),
                    "input_key": "input",
                    "label_key": "label",
                    "split": {"train": 0.5, "val": 0.25, "test": 0.25},
                },
                dspy={
                    "seed_prompt": "Test prompt: {input}",
                },
                llm={
                    "model": "mock-model",
                },
                optimizers=[
                    {"name": "nonexistent_optimizer", "config": {}},
                    {"name": "mock_optimizer", "config": {}},
                ],
            )

            # Create experiment
            llm_client = MockLLMClient()
            experiment = Experiment(config, llm_client, tmp_path / "experiments")

            # Load and split dataset
            dataset = JSONLDataset.load(str(dataset_path), "input", "label")
            from dspy_bench.data import SplitsManager

            splits_manager = SplitsManager(dataset)
            splits = splits_manager.create_splits(seed=42, **config.dataset.split)

            # Prepare and run experiment
            experiment.prepare(
                train_split=splits["train"],
                val_split=splits["val"],
                test_split=splits["test"],
            )

            results = experiment.run_all_optimizers()

            # One optimizer should fail, one should succeed
            assert len(results) == 2
            assert "nonexistent_optimizer" in results
            assert "mock_optimizer" in results
            assert results["nonexistent_optimizer"].success is False
            assert results["mock_optimizer"].success is True

    def test_dataset_validation_integration(self):
        """Test dataset validation in integration context."""
        # Create dataset with issues
        test_data = [
            {"input": "test input 1", "label": "output 1", "id": "test1"},
            {"input": "test input 2", "label": "output 2", "id": "test1"},  # Duplicate ID
            {"input": "", "label": "output 3", "id": "test3"},  # Empty input
            {"input": "test input 4", "label": None, "id": "test4"},  # Null label
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Save dataset
            dataset_path = tmp_path / "test_data.jsonl"
            with dataset_path.open('w') as f:
                for item in test_data:
                    f.write(json.dumps(item) + '\n')

            # Load and validate
            dataset = JSONLDataset.load(str(dataset_path), "input", "label", "id")
            validation_result = dataset.validate()

            # Should have warnings but still be valid
            assert validation_result.is_valid is True
            assert len(validation_result.warnings) > 0
            assert validation_result.total_examples == 4

    def test_multiple_optimizers_comparison(self):
        """Test comparison between multiple optimizers."""
        # Create test dataset
        test_data = [
            {"input": f"Question {i}", "label": f"Answer {i}"}
            for i in range(10)
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Save dataset
            dataset_path = tmp_path / "test_data.jsonl"
            with dataset_path.open('w') as f:
                for item in test_data:
                    f.write(json.dumps(item) + '\n')

            # Create configuration with multiple optimizers
            config = Config(
                experiment_name="comparison_test",
                seed=42,
                dataset={
                    "path": str(dataset_path),
                    "input_key": "input",
                    "label_key": "label",
                    "split": {"train": 0.6, "val": 0.2, "test": 0.2},
                },
                dspy={
                    "seed_prompt": "Answer: {input}",
                },
                llm={
                    "model": "mock-model",
                },
                optimizers=[
                    {"name": "mock_optimizer", "config": {"improvement_rate": 0.1}},
                    {"name": "mock_optimizer", "config": {"improvement_rate": 0.5}},
                    {"name": "mock_optimizer", "config": {"improvement_rate": 0.9}},
                ],
                metrics={
                    "metrics": ["accuracy"],
                },
            )

            # Create experiment
            llm_client = MockLLMClient(seed=42)
            experiment = Experiment(config, llm_client, tmp_path / "experiments")

            # Load and split dataset
            dataset = JSONLDataset.load(str(dataset_path), "input", "label")
            from dspy_bench.data import SplitsManager

            splits_manager = SplitsManager(dataset)
            splits = splits_manager.create_splits(seed=42, **config.dataset.split)

            # Prepare and run experiment
            experiment.prepare(
                train_split=splits["train"],
                val_split=splits["val"],
                test_split=splits["test"],
            )

            results = experiment.run_all_optimizers()
            eval_results = experiment.evaluate_results()

            # Should have results from all optimizers
            assert len(results) == 3
            assert len(eval_results) == 3

            # All should have different improvement rates, so potentially different results
            improvement_rates = [r.diagnostics.get("improvement_rate", 0) for r in results.values()]
            assert len(set(improvement_rates)) == 3  # All different