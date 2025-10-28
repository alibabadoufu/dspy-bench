"""Command-line interface for dspy-bench."""

import os
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from dspy_bench.config import Config
from dspy_bench.llm.openai_compat import OpenAICompatClient
from dspy_bench.llm.mock_client import MockLLMClient
from dspy_bench.optimizers.registry import OptimizerRegistry


console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="dspy-bench")
def main():
    """dspy-bench: Open-source extensible Python project to evaluate DSPy strategies & optimizers."""
    pass


@main.command()
@click.argument("output_dir", type=click.Path(), default="dspy_bench_project")
@click.option("--force", is_flag=True, help="Overwrite existing directory")
def init(output_dir: str, force: bool):
    """Initialize a new dspy-bench project with template configuration."""
    output_path = Path(output_dir)

    if output_path.exists() and not force:
        console.print(f"❌ Directory {output_dir} already exists. Use --force to overwrite.", style="red")
        sys.exit(1)

    # Create project structure
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "data").mkdir(exist_ok=True)
    (output_path / "experiments").mkdir(exist_ok=True)

    # Create example configuration
    example_config = {
        "experiment_name": "my_experiment",
        "seed": 42,
        "dataset": {
            "path": "data/mytask.jsonl",
            "input_key": "input",
            "label_key": "label",
            "split": {"train": 0.7, "val": 0.15, "test": 0.15},
        },
        "dspy": {
            "seed_prompt": """Instruction: {instruction}
Input: {input}
Output:""",
            "instruction": "Please provide the appropriate output for the given input.",
            "max_tokens": 1024,
            "temperature": 0.0,
        },
        "llm": {
            "host": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
            "model": "gpt-3.5-turbo",
            "max_retries": 3,
            "timeout": 60,
            "max_concurrency": 5,
        },
        "optimizers": [
            {"name": "bootstrap_fewshot", "config": {"max_bootstrapped_demos": 5}},
            {"name": "mock_optimizer", "config": {"improvement_rate": 0.3}},
        ],
        "output": {
            "experiments_dir": "experiments",
            "save_checkpoints": True,
            "save_predictions": True,
        },
        "metrics": {
            "metrics": ["accuracy", "exact_match"],
            "custom_metrics": {},
        },
    }

    # Save configuration
    config_path = output_path / "config.yaml"
    from dspy_bench.utils import save_yaml
    save_yaml(example_config, config_path)

    # Create example dataset
    example_dataset = [
        {"input": "What is 2+2?", "label": "4"},
        {"input": "What is the capital of France?", "label": "Paris"},
        {"input": "What color is the sky?", "label": "Blue"},
        {"input": "How many days in a week?", "label": "7"},
        {"input": "What is H2O?", "label": "Water"},
    ]

    import json
    dataset_path = output_path / "data" / "mytask.jsonl"
    with dataset_path.open("w") as f:
        for item in example_dataset:
            f.write(json.dumps(item) + "\n")

    # Create README
    readme_content = f"""# dspy-bench Project: {output_dir}

## Getting Started

1. Install dspy-bench:
   ```bash
   pip install dspy-bench
   ```

2. Set up your API key:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

3. Run the experiment:
   ```bash
   dspy-bench run --config config.yaml
   ```

## Project Structure

- `config.yaml` - Experiment configuration
- `data/` - Dataset files
- `experiments/` - Experiment results

## Next Steps

1. Replace the example dataset in `data/mytask.jsonl` with your own data
2. Modify `config.yaml` to suit your needs
3. Run experiments with different optimizers
4. Analyze results in the `experiments/` directory

## Documentation

For full documentation, see: https://dspy-bench.readthedocs.io
"""

    readme_path = output_path / "README.md"
    readme_path.write_text(readme_content)

    console.print(f"✅ Project initialized in {output_dir}", style="bold green")
    console.print(f"📝 Configuration: {config_path}")
    console.print(f"📊 Example dataset: {dataset_path}")
    console.print(f"\nNext steps:")
    console.print(f"1. cd {output_dir}")
    console.print(f"2. export OPENAI_API_KEY='your-api-key'")
    console.print(f"3. dspy-bench run --config config.yaml")


@main.command()
@click.option("--config", "-c", type=click.Path(exists=True), required=True, help="Configuration file")
@click.option("--dataset", "-d", type=click.Path(exists=True), help="Dataset path (overrides config)")
@click.option("--experiment-name", "-n", help="Override experiment name")
@click.option("--mock", is_flag=True, help="Use mock LLM client for testing")
def run(config: str, dataset: Optional[str], experiment_name: Optional[str], mock: bool):
    """Run experiments defined in configuration file."""
    console.print("🚀 Starting dspy-bench experiment", style="bold blue")

    try:
        # Load configuration
        cfg = Config.from_yaml(config)

        # Override experiment name if provided
        if experiment_name:
            cfg.experiment_name = experiment_name

        # Override dataset path if provided
        if dataset:
            cfg.dataset.path = dataset

        console.print(f"📋 Experiment: {cfg.experiment_name}")
        console.print(f"📊 Dataset: {cfg.dataset.path}")
        console.print(f"🤖 Optimizers: {[opt['name'] for opt in cfg.optimizers]}")

        # Create LLM client
        if mock:
            console.print("🧪 Using mock LLM client", style="yellow")
            llm_client = MockLLMClient(model=cfg.llm.model)
        else:
            api_key = cfg.llm.get_api_key()
            llm_client = OpenAICompatClient(
                model=cfg.llm.model,
                api_key=api_key,
                base_url=cfg.llm.host,
                max_retries=cfg.llm.max_retries,
                timeout=cfg.llm.timeout,
                max_concurrency=cfg.llm.max_concurrency,
            )

        # Test LLM connection
        if not mock and not llm_client.health_check():
            console.print("❌ LLM health check failed", style="red")
            sys.exit(1)

        # Create and run experiment
        from dspy_bench.runner.orchestrator import ExperimentOrchestrator

        orchestrator = ExperimentOrchestrator(llm_client, cfg.output.experiments_dir)
        experiment = orchestrator.run_experiment(cfg)

        console.print(f"🎉 Experiment completed successfully!", style="bold green")
        console.print(f"📁 Results: {experiment.experiment_dir}")

    except Exception as e:
        console.print(f"❌ Experiment failed: {e}", style="red")
        sys.exit(1)


@main.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file")
@click.option("--optimizer", "-o", help="Run specific optimizer")
@click.option("--dataset", "-d", type=click.Path(exists=True), help="Dataset path")
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option("--mock", is_flag=True, help="Use mock LLM client")
def run_one(
    optimizer: str,
    config: Optional[str],
    dataset: Optional[str],
    seed: int,
    mock: bool
):
    """Run a single optimizer on a dataset."""
    if optimizer is None:
        console.print("❌ Optimizer name is required", style="red")
        sys.exit(1)

    console.print(f"🔧 Running single optimizer: {optimizer}", style="bold blue")

    try:
        # Create minimal configuration
        if config:
            cfg = Config.from_yaml(config)
        else:
            cfg = Config(
                experiment_name=f"single_optimizer_{optimizer}",
                seed=seed,
                dataset={
                    "path": dataset or "data.jsonl",
                    "input_key": "input",
                    "label_key": "label",
                },
                dspy={
                    "seed_prompt": "Input: {input}\nOutput:",
                    "max_tokens": 100,
                },
                optimizers=[{"name": optimizer, "config": {}}],
            )

        # Override dataset if provided
        if dataset:
            cfg.dataset.path = dataset

        # Create LLM client
        if mock:
            llm_client = MockLLMClient()
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                console.print("❌ OPENAI_API_KEY environment variable not set", style="red")
                sys.exit(1)

            llm_client = OpenAICompatClient(
                model="gpt-3.5-turbo",
                api_key=api_key,
            )

        # Run experiment
        from dspy_bench.runner.orchestrator import ExperimentOrchestrator

        orchestrator = ExperimentOrchestrator(llm_client)
        experiment = orchestrator.run_experiment(cfg)

        console.print(f"✅ Optimization completed", style="green")
        console.print(f"📁 Results: {experiment.experiment_dir}")

    except Exception as e:
        console.print(f"❌ Optimization failed: {e}", style="red")
        sys.exit(1)


@main.command()
def list_optimizers():
    """List all available optimizers."""
    console.print("🤖 Available Optimizers", style="bold blue")
    OptimizerRegistry.print_optimizers_table()


@main.command()
@click.argument("experiment_id", required=False)
@click.option("--experiments-dir", "-e", type=click.Path(), default="experiments", help="Experiments directory")
def list_experiments(experiment_id: Optional[str], experiments_dir: str):
    """List experiments, optionally filtering by ID."""
    exp_path = Path(experiments_dir)

    if not exp_path.exists():
        console.print(f"❌ Experiments directory not found: {experiments_dir}", style="red")
        sys.exit(1)

    experiments = []
    for exp_dir in exp_path.iterdir():
        if exp_dir.is_dir():
            summary_file = exp_dir / "summary.json"
            if summary_file.exists():
                from dspy_bench.utils import load_json
                summary = load_json(summary_file)
                experiments.append(summary)

    if not experiments:
        console.print("No experiments found", style="yellow")
        return

    # Filter by experiment ID if provided
    if experiment_id:
        experiments = [exp for exp in experiments if experiment_id in exp.get("experiment_id", "")]

    # Create table
    table = Table(title="Experiments")
    table.add_column("Experiment ID", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Start Time", style="green")
    table.add_column("Runtime (s)", style="blue")
    table.add_column("Successful", style="yellow")

    for exp in sorted(experiments, key=lambda x: x.get("start_time", ""), reverse=True):
        successful = sum(1 for opt in exp.get("optimizers", {}).values() if opt.get("success", False))
        total = len(exp.get("optimizers", {}))
        runtime = exp.get("runtime_seconds", 0)

        table.add_row(
            exp.get("experiment_id", "unknown")[:16] + "...",
            exp.get("experiment_name", "unknown"),
            exp.get("start_time", "unknown")[:19] if exp.get("start_time") else "unknown",
            f"{runtime:.1f}",
            f"{successful}/{total}",
        )

    console.print(table)


@main.command()
@click.argument("experiment_id")
@click.option("--format", "output_format", type=click.Choice(["zip", "tar"]), default="zip")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--experiments-dir", "-e", type=click.Path(), default="experiments", help="Experiments directory")
def export(experiment_id: str, output_format: str, output: Optional[str], experiments_dir: str):
    """Export experiment results to a portable archive."""
    exp_path = Path(experiments_dir) / experiment_id

    if not exp_path.exists():
        console.print(f"❌ Experiment not found: {experiment_id}", style="red")
        sys.exit(1)

    if output is None:
        output = f"{experiment_id}.{output_format}"

    output_path = Path(output)

    # Create archive
    if output_format == "zip":
        import zipfile
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in exp_path.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(exp_path)
                    zipf.write(file_path, arcname)
    else:  # tar
        import tarfile
        with tarfile.open(output_path, "w:gz") as tarf:
            tarf.add(exp_path, arcname=experiment_id)

    console.print(f"✅ Experiment exported to: {output_path}", style="green")


if __name__ == "__main__":
    main()