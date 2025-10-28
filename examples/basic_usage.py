#!/usr/bin/env python3
"""Basic usage example for dspy-bench.

This example demonstrates how to use dspy-bench programmatically
to run experiments on a simple dataset.
"""

import json
import tempfile
from pathlib import Path

from dspy_bench.config import Config
from dspy_bench.data import JSONLDataset
from dspy_bench.llm import MockLLMClient
from dspy_bench.runner import ExperimentOrchestrator


def create_sample_dataset():
    """Create a sample dataset for demonstration."""
    dataset = [
        {"input": "What is 2+2?", "label": "4"},
        {"input": "What is 3+3?", "label": "6"},
        {"input": "What is 4+4?", "label": "8"},
        {"input": "What is 5+5?", "label": "10"},
        {"input": "What is 6+6?", "label": "12"},
        {"input": "What is 7+7?", "label": "14"},
        {"input": "What is 8+8?", "label": "16"},
        {"input": "What is 9+9?", "label": "18"},
        {"input": "What is 10+10?", "label": "20"},
        {"input": "What is 1+1?", "label": "2"},
    ]
    return dataset


def create_config(dataset_path: Path, experiments_dir: Path) -> Config:
    """Create experiment configuration."""
    return Config(
        experiment_name="basic_example",
        seed=42,
        dataset={
            "path": str(dataset_path),
            "input_key": "input",
            "label_key": "label",
            "split": {"train": 0.6, "val": 0.2, "test": 0.2},
        },
        dspy={
            "seed_prompt": "Solve this math problem: {input}\nAnswer:",
            "instruction": "Provide the numerical answer to the math problem.",
            "max_tokens": 5,
            "temperature": 0.0,
        },
        llm={
            "model": "mock-model",
            "max_retries": 1,
            "timeout": 10,
        },
        optimizers=[
            {"name": "mock_optimizer", "config": {"improvement_rate": 0.5, "max_improvements": 3}},
        ],
        metrics={
            "metrics": ["accuracy", "exact_match"],
        },
        output={
            "experiments_dir": str(experiments_dir),
            "save_checkpoints": True,
            "save_predictions": True,
        },
    )


def main():
    """Run the basic usage example."""
    print("🚀 dspy-bench Basic Usage Example")
    print("=" * 40)

    # Create temporary directory for the example
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create sample dataset
        print("📊 Creating sample dataset...")
        dataset = create_sample_dataset()
        dataset_path = tmp_path / "math_problems.jsonl"

        with dataset_path.open('w') as f:
            for item in dataset:
                f.write(json.dumps(item) + '\n')

        print(f"   Dataset saved to: {dataset_path}")
        print(f"   Examples: {len(dataset)}")

        # Create configuration
        print("\n⚙️  Creating configuration...")
        experiments_dir = tmp_path / "experiments"
        config = create_config(dataset_path, experiments_dir)

        # Create mock LLM client
        print("\n🤖 Creating mock LLM client...")
        llm_client = MockLLMClient(
            model="mock-model",
            deterministic=True,
            seed=42,
            response_delay=0.1,  # Fast for demo
        )

        # Run experiment
        print("\n🧪 Running experiment...")
        orchestrator = ExperimentOrchestrator(llm_client, str(experiments_dir))

        try:
            experiment = orchestrator.run_experiment(config)
            print(f"✅ Experiment completed successfully!")
            print(f"📁 Results saved to: {experiment.experiment_dir}")

            # Show results summary
            print("\n📈 Results Summary:")
            for optimizer_name, result in experiment.results.items():
                if result.success:
                    print(f"   {optimizer_name}: ✅ Success")
                    for metric_name, metric_value in result.metrics.items():
                        print(f"      {metric_name}: {metric_value:.4f}")
                else:
                    print(f"   {optimizer_name}: ❌ Failed ({result.error_message})")

            # Show experiment directory structure
            print(f"\n📂 Experiment directory structure:")
            for item in sorted(experiment.experiment_dir.rglob("*")):
                if item.is_file():
                    relative_path = item.relative_to(experiment.experiment_dir)
                    print(f"   {relative_path}")

        except Exception as e:
            print(f"❌ Experiment failed: {e}")
            return 1

    print("\n🎉 Example completed successfully!")
    print("\nNext steps:")
    print("1. Replace the mock LLM client with a real one")
    print("2. Use your own dataset")
    print("3. Try different optimizers")
    print("4. Customize the evaluation metrics")
    print("5. Add custom optimizers if needed")

    return 0


if __name__ == "__main__":
    exit(main())