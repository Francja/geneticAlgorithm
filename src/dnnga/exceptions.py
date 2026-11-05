"""Custom exceptions for dnnga.

Everything the library raises on purpose is a subclass of
:class:`DnnGaError`, so calling code can ``except DnnGaError`` to catch what
the library deliberately signals, while unexpected bugs still surface as
their natural exception type.
"""

from __future__ import annotations


class DnnGaError(Exception):
    """Base class for all exceptions raised deliberately by dnnga."""


class InvalidArchitectureError(DnnGaError):
    """An architecture violates its structural constraints (bounds, types, ...)."""


class ConfigError(DnnGaError):
    """The EvolutionConfig (or a related argument) is not usable.

    Raised up front (fail fast) rather than partway through a long run; the
    message lists every problem found, not just the first.
    """


class EvaluationError(DnnGaError):
    """Fitness evaluation (build/train/evaluate) failed.

    Raised when ``on_evaluation_error="raise"``, when too few individuals
    survive under ``"skip"``, or when *no* candidate in the whole run could
    be evaluated. With the default ``"penalize"`` a single failing candidate
    is logged and given the worst possible fitness instead.
    """


class BackendNotAvailableError(DnnGaError):
    """A backend's underlying framework (scikit-learn, TensorFlow, ...) is
    not installed."""
