"""Experiment orchestrator for managing multiple experiment runs."""

import concurrent.futures
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from dspy_bench.config import Config
from dspy_bench.data import JSONLDataset, SplitsManager
from dspy_bench.dspy_integration.dsp_adapter import DSPyAdapter
from dspy_bench.llm.base_client import BaseLLMClient
from dspy_bench.runner.experiment import Experiment


class ExperimentOrchestrator:
    """Orchestrates multiple experiment runs."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        experiments_dir: Union[str, Path] = "experiments",
        max_concurrent_experiments: int = 1,
    ):
        """Initialize orchestrator.

        Args:
            llm_client: LLM client to use for all experiments
            experiments_dir: Base directory for experiments
            max_concurrent_experiments: Maximum concurrent experiments
        """
        self.llm_client = llm_client
        self.experiments_dir = Path(experiments_dir)
        self.max_concurrent_experiments = max_concurrent_experiments
        self.console = Console()

        self.experiments: Dict[str, Experiment] = {}
        self.completed_experiments: List[Experiment] = []

    def create_experiment(self, config: Config) -> Experiment:
        """Create a new experiment.

        Args:
            config: Experiment configuration

        Returns:
            Created experiment
        """
        experiment_dir = self.experiments_dir / config.get_experiment_id()
        experiment = Experiment(config, self.llm_client, experiment_dir)

        self.experiments[experiment.experiment_id] = experiment
        return experiment

    def run_experiment(
        self,
        config: Config,
        dataset_path: Optional[Union[str, Path]] = None,
        load_dataset: bool = True,
    ) -> Experiment:
        """Run a single experiment end-to-end.

        Args:
            config: Experiment configuration
            dataset_path: Path to dataset (overrides config if provided)
            load_dataset: Whether to load dataset from config

        Returns:
            Completed experiment
        """
        experiment = self.create_experiment(config)

        try:
            # Load dataset if needed
            if load_dataset:
                if dataset_path:
                    config.dataset.path = dataset_path

                dataset = JSONLDataset.load(
                    config.dataset.path,
                    input_key=config.dataset.input_key,
                    label_key=config.dataset.label_key,
                    id_key=config.dataset.id_key,
                )

                # Create splits
                splits_manager = SplitsManager(dataset)
                splits = splits_manager.create_splits(
                    seed=config.seed,
                    **config.dataset.split,
                )

                train_split, val_split, test_split = splits["train"], splits["val"], splits["test"]
            else:
                # Use provided splits (for testing)
                train_split = val_split = test_split = None

            # Prepare experiment
            experiment.prepare(train_split, val_split, test_split)

            # Run optimizers
            experiment.run_all_optimizers()

            # Evaluate results
            experiment.evaluate_results()

            # Finalize
            summary = experiment.finalize()

            self.completed_experiments.append(experiment)
            return experiment

        except Exception as e:
            self.console.print(f"❌ Experiment {experiment.experiment_id} failed: {e}", style="red")
            raise

    def run_multiple_experiments(
        self,
        configs: List[Config],
        dataset_path: Optional[Union[str, Path]] = None,
        parallel: bool = False,
    ) -> List[Experiment]:
        """Run multiple experiments.

        Args:
            configs: List of experiment configurations
            dataset_path: Path to dataset (same for all experiments)
            parallel: Whether to run experiments in parallel

        Returns:
            List of completed experiments
        """
        if parallel and self.max_concurrent_experiments > 1:
            return self._run_experiments_parallel(configs, dataset_path)
        else:
            return self._run_experiments_sequential(configs, dataset_path)

    def _run_experiments_sequential(
        self,
        configs: List[Config],
        dataset_path: Optional[Union[str, Path]] = None,
    ) -> List[Experiment]:
        """Run experiments sequentially.

        Args:
            configs: List of experiment configurations
            dataset_path: Path to dataset

        Returns:
            List of completed experiments
        """
        completed = []

        for i, config in enumerate(configs):
            self.console.print(f"\n🚀 Running experiment {i+1}/{len(configs)}: {config.experiment_name}")

            try:
                experiment = self.run_experiment(config, dataset_path)
                completed.append(experiment)
                self.console.print(f"✅ Experiment {experiment.experiment_id} completed", style="green")
            except Exception as e:
                self.console.print(f"❌ Experiment failed: {e}", style="red")
                continue

        return completed

    def _run_experiments_parallel(
        self,
        configs: List[Config],
        dataset_path: Optional[Union[str, Path]] = None,
    ) -> List[Experiment]:
        """Run experiments in parallel.

        Args:
            configs: List of experiment configurations
            dataset_path: Path to dataset

        Returns:
            List of completed experiments
        """
        completed = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent_experiments) as executor:
            # Submit all experiments
            future_to_config = {
                executor.submit(self.run_experiment, config, dataset_path): config
                for config in configs
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_config):
                config = future_to_config[future]

                try:
                    experiment = future.result()
                    completed.append(experiment)
                    self.console.print(f"✅ Experiment {experiment.experiment_id} completed", style="green")
                except Exception as e:
                    self.console.print(f"❌ Experiment {config.experiment_name} failed: {e}", style="red")

        return completed

    def compare_experiments(self, experiments: Optional[List[Experiment]] = None) -> Dict[str, Any]:
        """Compare multiple experiments.

        Args:
            experiments: List of experiments to compare (uses all completed if None)

        Returns:
            Comparison results
        """
        if experiments is None:
            experiments = self.completed_experiments

        if not experiments:
            self.console.print("No experiments to compare", style="yellow")
            return {}

        # Create comparison table
        table = Table(title="Experiment Comparison")
        table.add_column("Experiment", style="cyan")
        table.add_column("Optimizer", style="magenta")
        table.add_column("Success", style="green")
        table.add_column("Runtime (s)", style="blue")
        table.add_column("Best Metric", style="yellow")

        comparison_data = {}

        for experiment in experiments:
            exp_data = {
                "experiment_id": experiment.experiment_id,
                "experiment_name": experiment.config.experiment_name,
                "optimizers": {},
            }

            for optimizer_name, result in experiment.results.items():
                success = "✅" if result.success else "❌"
                runtime = f"{result.runtime_seconds:.2f}"

                # Get best metric from evaluation results
                best_metric = "N/A"
                if optimizer_name in experiment.evaluation_results:
                    eval_result = experiment.evaluation_results[optimizer_name]
                    if eval_result.metrics:
                        # Use the first metric as "best"
                        first_metric = next(iter(eval_result.metrics.values()))
                        best_metric = f"{first_metric.value:.4f}"

                table.add_row(
                    experiment.experiment_id[:16] + "...",
                    optimizer_name,
                    success,
                    runtime,
                    best_metric,
                )

                exp_data["optimizers"][optimizer_name] = {
                    "success": result.success,
                    "runtime_seconds": result.runtime_seconds,
                    "metrics": result.metrics,
                }

            comparison_data[experiment.experiment_id] = exp_data

        self.console.print(table)
        return comparison_data

    def get_best_experiment(
        self,
        metric_name: str = "accuracy",
        experiments: Optional[List[Experiment]] = None,
    ) -> Optional[Experiment]:
        """Get the best experiment based on a metric.

        Args:
            metric_name: Metric to use for comparison
            experiments: List of experiments to consider (uses all completed if None)

        Returns:
            Best experiment or None
        """
        if experiments is None:
            experiments = self.completed_experiments

        best_experiment = None
        best_score = -float("inf")

        for experiment in experiments:
            for optimizer_name, eval_result in experiment.evaluation_results.items():
                if metric_name in eval_result.metrics:
                    score = eval_result.metrics[metric_name].value
                    if score > best_score:
                        best_score = score
                        best_experiment = experiment

        if best_experiment:
            self.console.print(
                f"🏆 Best experiment: {best_experiment.experiment_id} ({metric_name}: {best_score:.4f})",
                style="bold green"
            )

        return best_experiment

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments (completed and in-progress).

        Returns:
            List of experiment information
        """
        all_experiments = list(self.experiments.values()) + self.completed_experiments

        experiments_info = []
        for exp in all_experiments:
            info = {
                "experiment_id": exp.experiment_id,
                "experiment_name": exp.config.experiment_name,
                "status": "completed" if exp in self.completed_experiments else "in_progress",
                "start_time": exp.start_time,
                "end_time": exp.end_time,
                "optimizers_run": len(exp.results),
                "successful_optimizations": sum(1 for r in exp.results.values() if r.success),
            }
            experiments_info.append(info)

        return experiments_info

    def cleanup_experiments(self, keep_count: Optional[int] = None) -> None:
        """Clean up old experiments.

        Args:
            keep_count: Number of most recent experiments to keep (keeps all if None)
        """
        if keep_count is None:
            return

        # Sort experiments by start time
        sorted_experiments = sorted(
            self.completed_experiments,
            key=lambda x: x.start_time or datetime.min,
            reverse=True,
        )

        # Remove old experiments
        for experiment in sorted_experiments[keep_count:]:
            import shutil
            if experiment.experiment_dir.exists():
                shutil.rmtree(experiment.experiment_dir)
                self.console.print(f"🗑️  Removed experiment: {experiment.experiment_id}", style="yellow")

        self.completed_experiments = sorted_experiments[:keep_count]