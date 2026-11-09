"""Shared fixtures.

``FakeBackend`` does no real machine learning: its "fitness" is a
deterministic function of the architecture. That lets the engine's logic
(elitism, stopping criteria, caching, error policies, reproducibility...)
be tested in milliseconds with exact, predictable outcomes, decoupled from
the noise and cost of training real networks. Real-training behaviour is
tested separately in test_backends_sklearn.py / test_backends_keras.py.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np
import pytest

from dnnga.architecture import Architecture
from dnnga.backends.base import ModelBackend
from dnnga.config import EvolutionConfig
from dnnga.data import TrainingData


class FakeModel:
    def __init__(self, architecture: Architecture):
        self.architecture = architecture
        self.trained_epochs = 0


class FakeBackend(ModelBackend):
    """fitness = -|sum(layer widths) - target_sum| by default: a smooth,
    single-optimum landscape (best possible score 0) a GA should climb."""

    def __init__(
        self,
        target_sum: int = 30,
        fitness_fn: Optional[Callable[[Architecture], float]] = None,
        raise_for: Optional[Callable[[Architecture], bool]] = None,
    ):
        self.target_sum = target_sum
        self.fitness_fn = fitness_fn
        self.raise_for = raise_for
        self.build_calls = 0
        self.train_calls = 0
        self.evaluate_calls = 0
        self.eval_row_counts: list = []
        self.cleaned: list = []

    def build(self, architecture: Architecture, data: TrainingData) -> Any:
        self.build_calls += 1
        if self.raise_for is not None and self.raise_for(architecture):
            raise RuntimeError(f"FakeBackend: injected failure for {architecture}")
        return FakeModel(architecture)

    def train(self, model: Any, data: TrainingData, epochs: int) -> Any:
        self.train_calls += 1
        model.trained_epochs += epochs
        return model

    def evaluate(self, model: Any, data: TrainingData) -> float:
        self.evaluate_calls += 1
        self.eval_row_counts.append(len(data.X_val))
        if self.fitness_fn is not None:
            return self.fitness_fn(model.architecture)
        return -abs(sum(model.architecture.layer_sizes) - self.target_sum)

    def extract_architecture(self, model: Any) -> Architecture:
        return model.architecture

    def cleanup(self, model: Any) -> None:
        self.cleaned.append(model)  # keep references: ids of freed objects can be reused


def make_cfg(**overrides) -> EvolutionConfig:
    """A small, fast config for engine tests."""
    base = dict(
        min_layers=1,
        max_layers=4,
        min_neurons=1,
        max_neurons=20,
        population_size=12,
        n_generations=8,
        epochs_per_candidate=5,
        stagnation_generations=None,
        random_state=0,
    )
    base.update(overrides)
    return EvolutionConfig(**base)


@pytest.fixture(autouse=True)
def _reset_dnnga_logger():
    """configure_logging() mutates global logger state; undo it after every test."""
    yield
    logger = logging.getLogger("dnnga")
    for handler in list(logger.handlers):
        if getattr(handler, "_dnnga_managed", False):
            logger.removeHandler(handler)
            handler.close()
    logger.propagate = True
    logger.setLevel(logging.NOTSET)


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


def _split(X, y, task, with_test=False):
    n = len(X)
    a, b = int(n * 0.5), int(n * 0.8)  # 50 train / 30 val / 20 test of 100
    kw = {}
    if with_test:
        kw = dict(X_test=X[b:], y_test=y[b:])
    return TrainingData(X[:a], y[:a], X[a:b], y[a:b], task=task, **kw)


@pytest.fixture
def tiny_data() -> TrainingData:
    """Tiny, fast, linearly separable 2-class dataset (with a test split)."""
    rng = np.random.default_rng(0)
    n = 100
    X = np.vstack(
        [rng.normal(-2.0, 0.5, (n // 2, 3)), rng.normal(2.0, 0.5, (n // 2, 3))]
    )
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    idx = rng.permutation(n)
    return _split(X[idx], y[idx], "classification", with_test=True)


@pytest.fixture
def tiny_regression_data() -> TrainingData:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = X[:, 0] * 2.0 - X[:, 1] + rng.normal(scale=0.05, size=100)
    return _split(X, y, "regression")
