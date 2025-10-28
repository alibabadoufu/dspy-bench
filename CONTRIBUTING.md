# Contributing to dspy-bench

Thank you for your interest in contributing to dspy-bench! This document provides guidelines for contributors.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) for package management

### Setup Development Environment

1. **Fork and clone**:
   ```bash
   git clone https://github.com/your-username/dspy-bench.git
   cd dspy-bench
   ```

2. **Install dependencies**:
   ```bash
   uv sync --dev
   ```

3. **Install pre-commit hooks**:
   ```bash
   uv run pre-commit install
   ```

4. **Run tests to verify setup**:
   ```bash
   uv run pytest
   ```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed

### 3. Run Quality Checks

```bash
# Linting
uv run ruff check .

# Formatting
uv run black .
uv run isort .

# Type checking
uv run mypy dspy_bench/

# Run tests
uv run pytest
```

### 4. Commit Changes

Use conventional commit messages:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test changes
- `refactor:` for code refactoring

Example:
```bash
git commit -m "feat: add support for custom optimizer registration"
```

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear description of changes
- Reference any related issues
- Include screenshots if applicable

## Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **Ruff**: Linting and code analysis
- **mypy**: Static type checking

Configuration is in `pyproject.toml`.

### Code Guidelines

- Use type hints everywhere
- Follow PEP 8 style guide
- Write docstrings (Google style preferred)
- Keep functions and classes focused and small
- Add meaningful comments for complex logic

## Testing

### Test Structure

- Unit tests: `tests/test_*.py`
- Integration tests: `tests/test_integration.py`
- Tests should be fast, focused, and independent

### Running Tests

```bash
# All tests
uv run pytest

# Unit tests only
uv run pytest tests/ -m "not integration"

# Integration tests
uv run pytest tests/ -m integration

# With coverage
uv run pytest --cov=dspy_bench --cov-report=html

# Specific test file
uv run pytest tests/test_config.py
```

### Writing Tests

- Test the public API, not implementation details
- Use descriptive test names
- Use fixtures for common setup
- Mock external dependencies

Example:
```python
def test_dataset_validation_with_duplicates():
    """Test dataset validation detects duplicate IDs."""
    examples = [
        DatasetExample(input="test1", label="label1", id="duplicate"),
        DatasetExample(input="test2", label="label2", id="duplicate"),
    ]
    dataset = JSONLDataset(examples)

    result = dataset.validate()

    assert not result.is_valid
    assert any("duplicate" in warning.lower() for warning in result.warnings)
```

## Adding New Optimizers

1. **Create optimizer class**:
   ```python
   from dspy_bench.optimizers.base import BaseOptimizer
   from dspy_bench.optimizers.registry import register_optimizer

   @register_optimizer("my_optimizer")
   class MyOptimizer(BaseOptimizer):
       name = "my_optimizer"
       description = "Description of your optimizer"

       def optimize(self, program, train_data, val_data, metric_fn, **kwargs):
           # Implementation here
           pass

       def get_default_config(self):
           return {"param1": "value1"}
   ```

2. **Add tests**:
   ```python
   def test_my_optimizer_basic_functionality():
       config = OptimizationConfig()
       optimizer = MyOptimizer(config)
       # Test implementation
   ```

3. **Update documentation**:
   - Add to README.md optimizer list
   - Update API docs

## Adding New Metrics

1. **Implement metric function**:
   ```python
   def my_custom_metric(y_true, y_pred):
       """Calculate custom metric."""
       # Implementation
       return score
   ```

2. **Add to standard metrics**:
   ```python
   # In dspy_bench/metrics/standard.py
   from .standard import my_custom_metric
   ```

3. **Add tests**:
   ```python
   def test_my_custom_metric():
       # Test implementation
   ```

## Documentation

### API Documentation

- Use docstrings with Google style
- Include type hints
- Document parameters and return values

Example:
```python
def calculate_metric(y_true: List[str], y_pred: List[str]) -> float:
    """Calculate a custom metric.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Metric score between 0 and 1
    """
    pass
```

### README and User Docs

- Update README.md for user-facing changes
- Add examples for new features
- Keep documentation in sync with code

## Bug Reports

When filing bug reports:

1. Use the GitHub issue template
2. Include:
   - Clear description of the issue
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details
   - Error messages and logs

## Feature Requests

For feature requests:

1. Check existing issues first
2. Describe the use case clearly
3. Explain why it would be valuable
4. Consider if you can contribute it

## Release Process

Releases are managed by maintainers:

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create git tag
4. Publish to PyPI

## Community

- Be respectful and inclusive
- Help others in issues and discussions
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md)

## Getting Help

- Check documentation first
- Search existing issues
- Ask questions in GitHub Discussions
- Tag maintainers for urgent issues

Thank you for contributing to dspy-bench! 🚀