"""Generic registry for pluggable components."""

from typing import Dict, Type


class Registry:
    """A simple string-to-class mapping registry.

    Usage:
        REGISTRY = Registry("heads")

        @REGISTRY.register("classification")
        class ClassificationHead: ...

        cls = REGISTRY.get("classification")
    """

    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type] = {}

    def register(self, name: str):
        """Decorator that registers a class under the given name."""
        def decorator(cls):
            if name in self._registry:
                raise ValueError(
                    f"[{self.name}] '{name}' is already registered by {self._registry[name]}"
                )
            self._registry[name] = cls
            return cls
        return decorator

    def get(self, name: str) -> Type:
        """Retrieve a registered class by name."""
        if name not in self._registry:
            raise KeyError(
                f"[{self.name}] '{name}' not found. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[name]

    def list(self) -> list:
        """Return all registered names."""
        return list(self._registry.keys())
