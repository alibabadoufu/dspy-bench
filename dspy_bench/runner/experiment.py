"""Experiment class for managing individual optimization runs."""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from dspy_bench.config import Config
from dspy_bench.data.schema import DatasetSplit
from dspy_bench.dspy_integration.dsp_adapter import DSPyAdapter
from dspy_bench.dspy_integration.program_builder import ProgramBuilder
from dspy_bench.llm.base_client import BaseLLMClient
from dspy_bench.metrics.evaluator import Evaluator, EvaluationResult
from dspy_bench.optimizers.base import BaseOptimizer, OptimizationResult


class Experiment:
    """Manages a single experiment run with multiple optimizers."""

    def __init__(
        self,
        config: Config,
        llm_client: BaseLLMClient,
        experiment_dir: Optional[Path] = None,
    ):
        """Initialize experiment.

        Args:
            config: Experiment configuration
            llm_client: LLM client to use
            experiment_dir: Directory to store experiment artifacts
        """
        self.config = config
        self.llm_client = llm_client
        self.experiment_id = config.get_experiment_id()
        self.experiment_dir = experiment_dir or Path(config.output.experiments_dir) / self.experiment_id

        # Initialize components
        self.console = Console()
        self.logger = self._setup_logging()
        self.program_builder: Optional[ProgramBuilder] = None
        self.dspy_adapter: Optional[DSPyAdapter] = None
        self.evaluator: Optional[Evaluator] = None

        # Results storage
        self.results: Dict[str, OptimizationResult] = {}
        self.evaluation_results: Dict[str, EvaluationResult] = {}
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # Dataset splits
        self.train_split: Optional[DatasetSplit] = None
        self.val_split: Optional[DatasetSplit] = None
        self.test_split: Optional[DatasetSplit] = None

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the experiment.

        Returns:
            Configured logger
        """
        # Create experiment directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Setup logger
        logger = logging.getLogger(f"dspy_bench.experiment.{self.experiment_id}")
        logger.setLevel(logging.INFO)

        # File handler
        log_file = self.experiment_dir / "logs.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        if not logger.handlers:
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        return logger

    def prepare(self, train_split: DatasetSplit, val_split: DatasetSplit, test_split: DatasetSplit) -> None:
        """Prepare experiment components.

        Args:
            train_split: Training data split
            val_split: Validation data split
            test_split: Test data split
        """
        self.start_time = datetime.now()
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

        self.logger.info(f"Preparing experiment {self.experiment_id}")
        self.console.print(f"🚀 Starting experiment: {self.experiment_id}", style="bold blue")

        # Initialize DSPy adapter
        self.dspy_adapter = DSPyAdapter(self.llm_client, self.config.llm)
        self.dspy_adapter.configure_dspy()

        # Initialize program builder
        self.program_builder = ProgramBuilder(self.config.dspy)

        # Initialize evaluator
        metric_names = self.config.metrics.metrics
        self.evaluator = Evaluator(metric_names)

        # Save initial configuration
        self.save_config()

        # Save dataset splits
        self.save_dataset_splits()

        self.logger.info("Experiment preparation completed")

    def run_optimizer(self, optimizer_config: Dict[str, Any]) -> OptimizationResult:
        """Run a single optimizer.

        Args:
            optimizer_config: Optimizer configuration

        Returns:
            Optimization result
        """
        optimizer_name = optimizer_config["name"]
        optimizer_params = optimizer_config.get("config", {})

        self.logger.info(f"Running optimizer: {optimizer_name}")
        self.console.print(f"🔧 Running optimizer: {optimizer_name}", style="bold green")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task(f"Optimizing with {optimizer_name}...", total=None)

            try:
                # Create base program
                program = self.program_builder.build_program(
                    task_type="classification",  # TODO: Make configurable
                    few_shot_examples=None,
                )

                # Create optimizer
                from dspy_bench.optimizers.base import OptimizationConfig
                from dspy_bench.optimizers.registry import OptimizerRegistry

                opt_config = OptimizationConfig(**optimizer_params)
                optimizer = OptimizerRegistry.get(optimizer_name, opt_config)

                # Define metric function
                def metric_fn(example, prediction):
                    """Simple metric function for optimization."""
                    return str(prediction).strip().lower() == str(example.output).strip().lower()

                # Run optimization
                result = optimizer.optimize(
                    program=program,
                    train_data=list(self.train_split.examples),
                    val_data=list(self.val_split.examples),
                    metric_fn=metric_fn,
                    **optimizer_params,
                )

                self.results[optimizer_name] = result
                progress.update(task, description=f"✅ Completed {optimizer_name}")

                # Save optimizer result
                self.save_optimizer_result(optimizer_name, result)

                return result

            except Exception as e:
                error_msg = f"Optimizer {optimizer_name} failed: {str(e)}"
                self.logger.error(error_msg)
                progress.update(task, description=f"❌ {optimizer_name} failed")

                # Create failed result
                result = OptimizationResult(
                    optimizer_name=optimizer_name,
                    original_prompt=self.config.dspy.seed_prompt,
                    metrics={},
                    diagnostics={"error": str(e)},
                    runtime_seconds=0.0,
                    iterations_completed=0,
                    success=False,
                    error_message=str(e),
                )

                self.results[optimizer_name] = result
                return result

    def run_all_optimizers(self) -> Dict[str, OptimizationResult]:
        """Run all configured optimizers.

        Returns:
            Dictionary of optimization results by optimizer name
        """
        self.logger.info(f"Running {len(self.config.optimizers)} optimizers")
        self.console.print(f"📊 Running {len(self.config.optimizers)} optimizers", style="bold blue")

        for optimizer_config in self.config.optimizers:
            optimizer_name = optimizer_config["name"]
            self.console.print(f"\n--- Optimizer: {optimizer_name} ---")

            result = self.run_optimizer(optimizer_config)

            if result.success:
                self.console.print(f"✅ {optimizer_name} completed successfully", style="green")
                self.console.print(f"   Metrics: {result.metrics}")
            else:
                self.console.print(f"❌ {optimizer_name} failed: {result.error_message}", style="red")

        return self.results

    def evaluate_results(self) -> Dict[str, EvaluationResult]:
        """Evaluate optimization results on test set.

        Returns:
            Dictionary of evaluation results by optimizer name
        """
        if not self.test_split:
            raise ValueError("Test split not available for evaluation")

        self.logger.info("Evaluating optimization results on test set")
        self.console.print("🎯 Evaluating results on test set", style="bold blue")

        for optimizer_name, result in self.results.items():
            if not result.success or not result.optimized_program:
                self.console.print(f"⏭️  Skipping {optimizer_name} (no successful optimization)", style="yellow")
                continue

            self.console.print(f"Evaluating {optimizer_name}...")

            try:
                # Create prediction function from optimized program
                def prediction_func(input_text: str) -> Any:
                    program = result.optimized_program.program
                    response = program(input=input_text)
                    return response.output if hasattr(response, 'output') else str(response)

                # Evaluate on test set
                eval_result = self.evaluator.evaluate_dataset(
                    dataset=list(self.test_split.examples),
                    prediction_func=prediction_func,
                )

                self.evaluation_results[optimizer_name] = eval_result

                # Print results
                self.console.print(f"  Results: {eval_result.metrics}")

                # Save evaluation result
                self.save_evaluation_result(optimizer_name, eval_result)

            except Exception as e:
                self.logger.error(f"Evaluation failed for {optimizer_name}: {e}")
                self.console.print(f"❌ Evaluation failed for {optimizer_name}: {e}", style="red")

        return self.evaluation_results

    def save_config(self) -> None:
        """Save experiment configuration."""
        config_path = self.experiment_dir / "config.yaml"
        self.config.to_yaml(config_path)

        # Also save as JSON for easy parsing
        config_json_path = self.experiment_dir / "config.json"
        with config_json_path.open("w") as f:
            json.dump(self.config.dict(), f, indent=2, default=str)

        self.logger.info(f"Configuration saved to {config_path}")

    def save_dataset_splits(self) -> None:
        """Save dataset splits to experiment directory."""
        splits_dir = self.experiment_dir / "splits"
        splits_dir.mkdir(exist_ok=True)

        if self.train_split:
            self.save_split_to_jsonl(self.train_split, splits_dir / "train.jsonl")
        if self.val_split:
            self.save_split_to_jsonl(self.val_split, splits_dir / "val.jsonl")
        if self.test_split:
            self.save_split_to_jsonl(self.test_split, splits_dir / "test.jsonl")

    def save_split_to_jsonl(self, split: DatasetSplit, file_path: Path) -> None:
        """Save a dataset split to JSONL file.

        Args:
            split: Dataset split to save
            file_path: Output file path
        """
        with file_path.open("w") as f:
            for example in split.examples:
                record = {
                    "input": example.input,
                    "label": example.label,
                }
                if example.id:
                    record["id"] = example.id
                record.update(example.metadata)
                f.write(json.dumps(record) + "\n")

    def save_optimizer_result(self, optimizer_name: str, result: OptimizationResult) -> None:
        """Save optimization result.

        Args:
            optimizer_name: Name of the optimizer
            result: Optimization result
        """
        results_dir = self.experiment_dir / "optimizers"
        results_dir.mkdir(exist_ok=True)

        # Save result as JSON
        result_path = results_dir / f"{optimizer_name}_result.json"
        with result_path.open("w") as f:
            json.dump(result.dict(), f, indent=2, default=str)

        # Save optimized prompt if available
        if result.final_prompt:
            prompt_path = results_dir / f"{optimizer_name}_prompt.txt"
            with prompt_path.open("w") as f:
                f.write(result.final_prompt)

    def save_evaluation_result(self, optimizer_name: str, result: EvaluationResult) -> None:
        """Save evaluation result.

        Args:
            optimizer_name: Name of the optimizer
            result: Evaluation result
        """
        eval_dir = self.experiment_dir / "evaluation"
        eval_dir.mkdir(exist_ok=True)

        # Save evaluation result
        eval_path = eval_dir / f"{optimizer_name}_evaluation.json"
        result_data = {
            "summary": result.summary_dict(),
            "metrics": {name: {"value": metric.value, "details": metric.details}
                       for name, metric in result.metrics.items()},
            "predictions": result.predictions,
        }

        with eval_path.open("w") as f:
            json.dump(result_data, f, indent=2, default=str)

        # Save predictions separately
        predictions_path = eval_dir / f"{optimizer_name}_predictions.jsonl"
        with predictions_path.open("w") as f:
            for pred in result.predictions:
                f.write(json.dumps(pred, default=str) + "\n")

    def finalize(self) -> Dict[str, Any]:
        """Finalize experiment and save final results.

        Returns:
            Final experiment summary
        """
        self.end_time = datetime.now()
        runtime = (self.end_time - self.start_time).total_seconds() if self.start_time else 0

        # Create final summary
        summary = {
            "experiment_id": self.experiment_id,
            "experiment_name": self.config.experiment_name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "runtime_seconds": runtime,
            "dataset_info": {
                "train_size": len(self.train_split) if self.train_split else 0,
                "val_size": len(self.val_split) if self.val_split else 0,
                "test_size": len(self.test_split) if self.test_split else 0,
            },
            "optimizers": {
                name: {
                    "success": result.success,
                    "metrics": result.metrics,
                    "runtime_seconds": result.runtime_seconds,
                    "error": result.error_message,
                }
                for name, result in self.results.items()
            },
            "evaluation": {
                name: {
                    "metrics": {metric_name: metric.value for metric_name, metric in result.metrics.items()},
                    "evaluation_time": result.evaluation_time,
                }
                for name, result in self.evaluation_results.items()
            },
        }

        # Save final summary
        summary_path = self.experiment_dir / "summary.json"
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Create README
        self.create_readme()

        self.logger.info(f"Experiment {self.experiment_id} completed")
        self.console.print(f"🎉 Experiment {self.experiment_id} completed!", style="bold green")
        self.console.print(f"📁 Results saved to: {self.experiment_dir}")

        return summary

    def create_readme(self) -> None:
        """Create README file for the experiment."""
        readme_path = self.experiment_dir / "README.md"

        readme_content = f"""# Experiment: {self.experiment_id}

## Overview
- **Experiment Name**: {self.config.experiment_name}
- **Start Time**: {self.start_time}
- **End Time**: {self.end_time}
- **Runtime**: {(self.end_time - self.start_time).total_seconds():.2f} seconds

## Dataset
- **Train Size**: {len(self.train_split) if self.train_split else 0}
- **Validation Size**: {len(self.val_split) if self.val_split else 0}
- **Test Size**: {len(self.test_split) if self.test_split else 0}

## Optimizers
{chr(10).join([f"- **{name}**: {'✅ Success' if result.success else '❌ Failed'}" for name, result in self.results.items()])}

## Results
- **Optimizers Run**: {len(self.results)}
- **Successful Optimizations**: {sum(1 for r in self.results.values() if r.success)}

## Files
- `config.yaml` - Experiment configuration
- `config.json` - Configuration in JSON format
- `summary.json` - Final experiment summary
- `logs.log` - Detailed logs
- `splits/` - Dataset splits used
- `optimizers/` - Optimization results and prompts
- `evaluation/` - Final evaluation results

## Metrics
{chr(10).join([f"### {name}{chr(10)}{chr(10).join([f'- {metric_name}: {metric.value:.4f}' for metric_name, metric in result.metrics.items()])}" for name, result in self.evaluation_results.items()])}
"""

        with readme_path.open("w") as f:
            f.write(readme_content)