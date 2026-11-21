"""Tests for Issue #3 Priority 7: scikit-learn backend reproducibility.

Covers mutation gaps in src/dnnga/backends/sklearn_backend.py:
- random_state forwarding to MLPClassifier and MLPRegressor
- random_state=None / removing random_state is observable
- reproducible runs with identical random_state
- constructor kwarg forwarding verification (spy)
"""

from __future__ import annotations

from typing import Any

import pytest

sklearn = pytest.importorskip("sklearn")

from sklearn.neural_network import MLPClassifier, MLPRegressor

from dnnga import (
    Architecture,
    EvolutionConfig,
    GeneticArchitectureSearch,
)
from dnnga.backends.sklearn_backend import SklearnMLPBackend
from tests.conftest import make_cfg


class TestSklearnReproducibilityGaps:
    """P7: Verify random_state forwarding and reproducibility for SklearnMLPBackend."""

    def test_random_state_stored_on_backend(self):
        b1 = SklearnMLPBackend(random_state=123)
        assert b1.random_state == 123
        b2 = SklearnMLPBackend()
        assert b2.random_state is None

    def test_random_state_forwarded_to_classifier(self, tiny_data):
        backend = SklearnMLPBackend(random_state=42)
        model = backend.build(Architecture((8,)), tiny_data)
        assert isinstance(model, MLPClassifier)
        assert model.random_state == 42
        assert model.max_iter == 1

    def test_random_state_forwarded_to_regressor(self, tiny_regression_data):
        backend = SklearnMLPBackend(random_state=84)
        model = backend.build(Architecture((8,)), tiny_regression_data)
        assert isinstance(model, MLPRegressor)
        assert model.random_state == 84
        assert model.max_iter == 1

    def test_random_state_none_forwarded(self, tiny_data, tiny_regression_data):
        backend = SklearnMLPBackend(random_state=None)
        clf = backend.build(Architecture((6,)), tiny_data)
        assert clf.random_state is None
        reg = backend.build(Architecture((6,)), tiny_regression_data)
        assert reg.random_state is None

    def test_constructor_spy_verifies_exact_forwarded_kwargs(self, monkeypatch, tiny_data):
        captured_kwargs: dict[str, Any] = {}
        original_init = MLPClassifier.__init__

        def spy_init(self, *args, **kwargs):
            captured_kwargs.update(kwargs)
            original_init(self, *args, **kwargs)

        monkeypatch.setattr(MLPClassifier, "__init__", spy_init)

        backend = SklearnMLPBackend(random_state=999, alpha=0.05, solver="adam")
        model = backend.build(Architecture((12, 6)), tiny_data)

        assert isinstance(model, MLPClassifier)
        assert captured_kwargs.get("random_state") == 999
        assert captured_kwargs.get("max_iter") == 1
        assert captured_kwargs.get("hidden_layer_sizes") == (12, 6)
        assert captured_kwargs.get("alpha") == 0.05
        assert captured_kwargs.get("solver") == "adam"

    def test_constructor_spy_verifies_regressor_kwargs(self, monkeypatch, tiny_regression_data):
        captured_kwargs: dict[str, Any] = {}
        original_init = MLPRegressor.__init__

        def spy_init(self, *args, **kwargs):
            captured_kwargs.update(kwargs)
            original_init(self, *args, **kwargs)

        monkeypatch.setattr(MLPRegressor, "__init__", spy_init)

        backend = SklearnMLPBackend(random_state=555, alpha=0.1)
        model = backend.build(Architecture((4, 4)), tiny_regression_data)

        assert isinstance(model, MLPRegressor)
        assert captured_kwargs.get("random_state") == 555
        assert captured_kwargs.get("max_iter") == 1
        assert captured_kwargs.get("hidden_layer_sizes") == (4, 4)
        assert captured_kwargs.get("alpha") == 0.1

    def test_two_runs_with_same_random_state_give_identical_results(self, tiny_data):
        cfg = make_cfg(
            population_size=4,
            n_generations=2,
            random_state=42,
            epochs_per_candidate=2,
        )

        def run_once():
            b = SklearnMLPBackend(random_state=42, learning_rate_init=0.01)
            return GeneticArchitectureSearch(b, cfg).run(tiny_data)

        res1 = run_once()
        res2 = run_once()

        assert res1.best_architecture == res2.best_architecture
        assert res1.best_fitness == res2.best_fitness
        assert [r.best_fitness for r in res1.history.records] == [
            r.best_fitness for r in res2.history.records
        ]
        assert [r.best_architecture for r in res1.history.records] == [
            r.best_architecture for r in res2.history.records
        ]

    def test_removing_random_state_is_observable(self, tiny_data):
        # When backend random_state is explicitly set, build() must not produce random_state=None
        backend = SklearnMLPBackend(random_state=77)
        model = backend.build(Architecture((4,)), tiny_data)
        assert model.random_state is not None
        assert model.random_state == 77
