"""dnnga: evolve a neural network's hidden-layer structure with a genetic algorithm.

A framework-agnostic re-implementation of the GA-driven architecture search
from the thesis "Optymalizacja sieci neuronowej za pomocą algorytmów
genetycznych" and its ``geneticAlgorithm.py`` /
``Genetic_algorithm_for_DNN.ipynb``.

>>> from dnnga import EvolutionConfig, GeneticArchitectureSearch, TrainingData
>>> from dnnga.backends import SklearnMLPBackend
>>> data = TrainingData(X_train, y_train, X_val, y_val, X_test=X_test, y_test=y_test)
>>> config = EvolutionConfig(min_layers=1, max_layers=5, max_neurons=32,
...                          population_size=16)
>>> result = GeneticArchitectureSearch(
...     SklearnMLPBackend(random_state=0), config
... ).run(data, seed=[16, 8])
>>> result.best_architecture, result.best_fitness, result.test_score
"""

from __future__ import annotations

from .architecture import Architecture, architecture_from_any, random_architecture
from .config import EvolutionConfig, bounds_relative_to
from .data import TrainingData
from .engine import EvolutionResult, GeneticArchitectureSearch, evolve_architecture
from .exceptions import (
    BackendNotAvailableError,
    ConfigError,
    DnnGaError,
    EvaluationError,
    InvalidArchitectureError,
)
from .history import GenerationRecord, RunHistory
from .individual import Individual
from .logging_utils import configure_logging

__version__ = "0.2.0"

__all__ = [
    "Architecture",
    "architecture_from_any",
    "random_architecture",
    "EvolutionConfig",
    "bounds_relative_to",
    "TrainingData",
    "GeneticArchitectureSearch",
    "EvolutionResult",
    "evolve_architecture",
    "Individual",
    "GenerationRecord",
    "RunHistory",
    "configure_logging",
    "DnnGaError",
    "ConfigError",
    "InvalidArchitectureError",
    "EvaluationError",
    "BackendNotAvailableError",
]
