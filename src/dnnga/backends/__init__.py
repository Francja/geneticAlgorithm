"""Backends are imported lazily (PEP 562) so that ``import dnnga`` never
requires scikit-learn, TensorFlow or PyTorch to be installed -- only the
backend you actually use."""

from __future__ import annotations

from .base import ModelBackend

__all__ = ["ModelBackend", "SklearnMLPBackend", "KerasMLPBackend"]


def __getattr__(name):
    if name == "SklearnMLPBackend":
        from .sklearn_backend import SklearnMLPBackend

        return SklearnMLPBackend
    if name == "KerasMLPBackend":
        from .keras_backend import KerasMLPBackend

        return KerasMLPBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
