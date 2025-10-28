"""Tests for optimizer system."""

import pytest

from dspy_bench.optimizers.base import OptimizationConfig, OptimizationResult
from dspy_bench.optimizers.registry import OptimizerRegistry
from dspy_bench.optimizers.mock_optimizer import MockOptimizer


class TestOptimizationConfig:
    """Test optimization configuration."""

    def test_optimization_config_creation(self):
        """Test creating optimization configuration."""
        config = OptimizationConfig(
            budget=100,
            max_iterations=50,
            early_stop=True,
        )
        assert config.budget == 100
        assert config.max_iterations == 50
        assert config.early_stop is True

    def test_optimization_config_defaults(self):
        """Test optimization configuration defaults."""
        config = OptimizationConfig()
        assert config.max_iterations == 100
        assert config.early_stop is True
        assert config.save_checkpoints is True


class TestOptimizationResult:
    """Test optimization result."""

    def test_optimization_result_creation(self):
        """Test creating optimization result."""
        result = OptimizationResult(
            optimizer_name="test_optimizer",
            original_prompt="Test prompt",
            metrics={"accuracy": 0.85},
            runtime_seconds=120.5,
            success=True,
        )
        assert result.optimizer_name == "test_optimizer"
        assert result.original_prompt == "Test prompt"
        assert result.metrics["accuracy"] == 0.85
        assert result.runtime_seconds == 120.5
        assert result.success is True


class TestOptimizerRegistry:
    """Test optimizer registry."""

    def test_register_optimizer(self):
        """Test registering optimizers."""
        # Registry should have some optimizers already registered
        available = OptimizerRegistry.list_optimizers()
        assert len(available) > 0
        assert "mock_optimizer" in available

    def test_get_optimizer(self):
        """Test getting optimizer from registry."""
        config = OptimizationConfig()
        optimizer = OptimizerRegistry.get("mock_optimizer", config)
        assert optimizer.name == "mock_optimizer"
        assert isinstance(optimizer, MockOptimizer)

    def test_get_nonexistent_optimizer(self):
        """Test getting non-existent optimizer."""
        config = OptimizationConfig()
        with pytest.raises(KeyError, match="Optimizer 'nonexistent' not found"):
            OptimizerRegistry.get("nonexistent", config)

    def test_is_registered(self):
        """Test checking if optimizer is registered."""
        assert OptimizerRegistry.is_registered("mock_optimizer")
        assert not OptimizerRegistry.is_registered("nonexistent_optimizer")

    def test_unregister_optimizer(self):
        """Test unregistering optimizer."""
        # Register a test optimizer first
        class TestTempOptimizer(MockOptimizer):
            name = "test_temp"

        OptimizerRegistry.register("test_temp", TestTempOptimizer)

        # Verify it's registered
        assert OptimizerRegistry.is_registered("test_temp")

        # Unregister it
        OptimizerRegistry.unregister("test_temp")
        assert not OptimizerRegistry.is_registered("test_temp")


class TestMockOptimizer:
    """Test mock optimizer."""

    def test_mock_optimizer_creation(self):
        """Test creating mock optimizer."""
        config = OptimizationConfig(
            max_iterations=10,
            random_seed=42,
        )
        optimizer = MockOptimizer(config)
        assert optimizer.name == "mock_optimizer"
        assert optimizer.config.max_iterations == 10

    def test_mock_optimizer_optimization(self):
        """Test mock optimizer optimization."""
        from dspy_bench.dspy_integration.program_builder import DSPyProgramWrapper
        import dspy

        config = OptimizationConfig(random_seed=42)
        optimizer = MockOptimizer(config)

        # Create mock program
        program = DSPyProgramWrapper(
            program=dspy.Predict(dspy.Signature("input -> output")),
            seed_prompt="Test prompt",
            instruction="Test instruction",
            signature="test_signature",
        )

        # Create mock data
        train_data = [
            type('Example', (), {'input': 'input1', 'output': 'output1'})(),
            type('Example', (), {'input': 'input2', 'output': 'output2'})(),
        ]
        val_data = [
            type('Example', (), {'input': 'val_input', 'output': 'val_output'})(),
        ]

        def mock_metric_fn(example, prediction):
            return True  # Always return True for testing

        # Run optimization
        result = optimizer.optimize(
            program=program,
            train_data=train_data,
            val_data=val_data,
            metric_fn=mock_metric_fn,
            improvement_rate=0.5,
            max_improvements=3,
        )

        assert result.optimizer_name == "mock_optimizer"
        assert result.success is True
        assert result.original_prompt == "Test prompt"
        assert result.final_prompt is not None
        assert result.runtime_seconds > 0
        assert "accuracy" in result.metrics

    def test_mock_optimizer_default_config(self):
        """Test mock optimizer default configuration."""
        config = OptimizationConfig()
        optimizer = MockOptimizer(config)
        default_config = optimizer.get_default_config()

        assert "improvement_rate" in default_config
        assert "max_improvements" in default_config
        assert "random_seed" in default_config
        assert default_config["improvement_rate"] == 0.2
        assert default_config["max_improvements"] == 10
        assert default_config["random_seed"] == 42

    def test_mock_optimizer_reproducibility(self):
        """Test mock optimizer reproducibility with same seed."""
        from dspy_bench.dspy_integration.program_builder import DSPyProgramWrapper
        import dspy

        config = OptimizationConfig(random_seed=42)
        optimizer1 = MockOptimizer(config)
        optimizer2 = MockOptimizer(config)

        program = DSPyProgramWrapper(
            program=dspy.Predict(dspy.Signature("input -> output")),
            seed_prompt="Test prompt",
            instruction="Test instruction",
            signature="test_signature",
        )

        train_data = [
            type('Example', (), {'input': 'input1', 'output': 'output1'})(),
        ]
        val_data = [
            type('Example', (), {'input': 'val_input', 'output': 'val_output'})(),
        ]

        def mock_metric_fn(example, prediction):
            return True

        # Run both optimizers
        result1 = optimizer1.optimize(
            program, train_data, val_data, mock_metric_fn,
            improvement_rate=1.0, max_improvements=1
        )
        result2 = optimizer2.optimize(
            program, train_data, val_data, mock_metric_fn,
            improvement_rate=1.0, max_improvements=1
        )

        # With same seed, should get same results
        assert result1.metrics == result2.metrics