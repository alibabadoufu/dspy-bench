"""Registry for optimizer discovery and management."""

from typing import Dict, List, Type, Union

from rich.console import Console
from rich.table import Table

from dspy_bench.optimizers.base import BaseOptimizer, OptimizationConfig


class OptimizerRegistry:
    """Registry for managing available optimizers."""

    _registry: Dict[str, Type[BaseOptimizer]] = {}
    _descriptions: Dict[str, str] = {}

    @classmethod
    def register(cls, name: str, optimizer_class: Type[BaseOptimizer]) -> None:
        """Register an optimizer class.

        Args:
            name: Optimizer name (unique identifier)
            optimizer_class: Optimizer class to register
        """
        if not issubclass(optimizer_class, BaseOptimizer):
            raise ValueError(f"Optimizer class {optimizer_class} must inherit from BaseOptimizer")

        cls._registry[name.lower()] = optimizer_class
        cls._descriptions[name.lower()] = getattr(optimizer_class, "description", "No description available")

    @classmethod
    def get(cls, name: str, config: OptimizationConfig) -> BaseOptimizer:
        """Get an optimizer instance.

        Args:
            name: Optimizer name
            config: Optimization configuration

        Returns:
            Optimizer instance

        Raises:
            KeyError: If optimizer not found
        """
        name_lower = name.lower()
        if name_lower not in cls._registry:
            available = list(cls._registry.keys())
            raise KeyError(f"Optimizer '{name}' not found. Available optimizers: {available}")

        optimizer_class = cls._registry[name_lower]
        return optimizer_class(config)

    @classmethod
    def list_optimizers(cls) -> Dict[str, Dict[str, Union[str, List[str]]]]:
        """List all registered optimizers with their information.

        Returns:
            Dictionary of optimizer information
        """
        info = {}
        for name, optimizer_class in cls._registry.items():
            # Create a temporary instance to get default config
            temp_config = OptimizationConfig()
            try:
                temp_instance = optimizer_class(temp_config)
                default_config = temp_instance.get_default_config()
                required_keys = getattr(temp_instance, "required_config_keys", [])
                optional_keys = getattr(temp_instance, "optional_config_keys", [])
            except Exception:
                default_config = {}
                required_keys = []
                optional_keys = []

            info[name] = {
                "name": name,
                "description": cls._descriptions.get(name, "No description available"),
                "class": optimizer_class.__name__,
                "default_config": default_config,
                "required_keys": required_keys,
                "optional_keys": optional_keys,
            }

        return info

    @classmethod
    def print_optimizers_table(cls) -> None:
        """Print a formatted table of available optimizers."""
        console = Console()

        table = Table(title="Available Optimizers")
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Description", style="magenta")
        table.add_column("Required Keys", style="green")
        table.add_column("Optional Keys", style="yellow")

        for name, info in cls.list_optimizers().items():
            required = ", ".join(info["required_keys"]) if info["required_keys"] else "None"
            optional = ", ".join(info["optional_keys"]) if info["optional_keys"] else "None"

            table.add_row(
                name,
                info["description"],
                required,
                optional,
            )

        console.print(table)

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if an optimizer is registered.

        Args:
            name: Optimizer name

        Returns:
            True if registered, False otherwise
        """
        return name.lower() in cls._registry

    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister an optimizer.

        Args:
            name: Optimizer name to unregister
        """
        name_lower = name.lower()
        if name_lower in cls._registry:
            del cls._registry[name_lower]
        if name_lower in cls._descriptions:
            del cls._descriptions[name_lower]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered optimizers."""
        cls._registry.clear()
        cls._descriptions.clear()

    @classmethod
    def validate_optimizer_config(cls, name: str, config: Dict[str, Union[str, int, float, bool]]) -> bool:
        """Validate configuration for a specific optimizer.

        Args:
            name: Optimizer name
            config: Configuration to validate

        Returns:
            True if valid, raises ValueError if invalid

        Raises:
            KeyError: If optimizer not found
        """
        if not cls.is_registered(name):
            raise KeyError(f"Optimizer '{name}' not found")

        # Create temporary instance to validate config
        temp_config = OptimizationConfig(**config.get("optimization", {}))
        optimizer = cls.get(name, temp_config)
        return optimizer.validate_config(config.get("optimizer_params", {}))


def register_optimizer(name: str):
    """Decorator for registering optimizers.

    Args:
        name: Optimizer name

    Returns:
        Decorator function
    """
    def decorator(optimizer_class: Type[BaseOptimizer]) -> Type[BaseOptimizer]:
        OptimizerRegistry.register(name, optimizer_class)
        return optimizer_class

    return decorator


# Auto-discovery utility
def auto_register_optimizers() -> None:
    """Auto-register optimizers from the optimizers package."""
    import importlib
    import pkgutil

    # Import all modules in the optimizers package
    package_name = "dspy_bench.optimizers"
    package = importlib.import_module(package_name)

    for importer, modname, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if not ispkg and modname.split(".")[-1] not in ["base", "registry"]:
            try:
                importlib.import_module(modname)
            except ImportError as e:
                # Don't fail the whole process if one optimizer fails to import
                console = Console()
                console.print(f"Warning: Failed to import optimizer module {modname}: {e}", style="yellow")