"""Mutation-gap tests for issue #3, priorities 1-2 (``engine.py``).

The 77.5 % mutation baseline left these behaviours unasserted:

* ``evolve_architecture()`` passed its arguments through untested -- the
  ``config`` could be replaced by ``EvolutionConfig()``, the seed and
  ``seed_fitness`` could be dropped, ``resolve_seed`` could be called with
  ``None`` instead of the backend, and the fixed-depth operator disabling
  could fire (or not) at the wrong time;
* ``_finalize_best()`` retrained the winner and nobody checked that the
  retrained score, the direction sign or the metadata keys survived.

Each test asserts a *value* (score, sign, metadata key, exact count) rather
than merely "it did not raise", so a mutant that changes any of them is caught.
"""

from __future__ import annotations

import dnnga.engine as engine_mod
from dnnga import (
    Architecture,
    EvolutionConfig,
    GeneticArchitectureSearch,
    evolve_architecture,
)
from tests.conftest import FakeBackend, FakeModel, make_cfg


def _run(cfg=None, backend=None, data=None, **kw):
    backend = backend or FakeBackend()
    return GeneticArchitectureSearch(backend, cfg or make_cfg()).run(data, **kw), backend


class EpochScoreBackend(FakeBackend):
    """Score == epochs trained, so the search budget and the final budget differ.

    ``FakeBackend`` scores the architecture only, which makes a final retrain
    invisible; this one lets a test tell "scored during the search" (cheap
    budget) from "scored after the final retrain" (full budget).
    """

    def evaluate(self, model, data):
        self.evaluate_calls += 1
        self.eval_row_counts.append(len(data.X_val))
        return float(model.trained_epochs)


def _capture_config(monkeypatch):
    """Record the config the search is actually constructed with."""
    captured = {}
    real_init = engine_mod.GeneticArchitectureSearch.__init__

    def spy(self, backend, config=None):
        real_init(self, backend, config)
        captured["config"] = self.config

    monkeypatch.setattr(engine_mod.GeneticArchitectureSearch, "__init__", spy)
    return captured


# ------------------------------------------------- P1: evolve_architecture
class TestEvolveArchitecturePassThrough:
    def test_a_non_default_config_is_honoured(self, tiny_data):
        # No keyword overrides: if evolve_architecture() ignores ``config`` and
        # falls back to EvolutionConfig(), the defaults are 20 individuals and
        # 20 generations, so both assertions fail.
        cfg = make_cfg(population_size=3, n_generations=2, epochs_per_candidate=2)
        res = evolve_architecture(
            [15, 15], tiny_data, FakeBackend(), config=cfg, return_result=True
        )
        assert len(res.history.records) == 2
        assert res.history.records[0].population_size == 3

    def test_the_seed_reaches_generation_zero_with_origin_seed(
        self, monkeypatch, tiny_data
    ):
        captured = {}
        real = engine_mod.GeneticArchitectureSearch._initialize_population

        def spy(self, seed, seed_fitness, sign, rng, cache):
            population = real(self, seed, seed_fitness, sign, rng, cache)
            captured["population"] = population
            return population

        monkeypatch.setattr(
            engine_mod.GeneticArchitectureSearch, "_initialize_population", spy
        )
        evolve_architecture(
            [15, 15],
            tiny_data,
            FakeBackend(),
            return_result=True,
            population_size=4,
            n_generations=1,
            epochs_per_candidate=2,
            random_state=0,
            stagnation_generations=None,
        )

        seed_individuals = [
            ind
            for ind in captured["population"]
            if ind.metadata.get("origin") == "seed"
        ]
        assert len(seed_individuals) == 1
        assert seed_individuals[0].architecture == Architecture((15, 15))

    def test_seed_fitness_is_passed_through_and_skips_training(self, tiny_data):
        # The seed is pre-scored (worst possible), so it must not be trained:
        # 5 individuals - 1 = 4 trainings. Dropping seed_fitness re-trains it.
        res = evolve_architecture(
            [15, 15],
            tiny_data,
            FakeBackend(),
            seed_fitness=-100.0,
            return_result=True,
            population_size=5,
            n_generations=1,
            epochs_per_candidate=2,
            random_state=0,
            stagnation_generations=None,
        )
        assert res.history.records[0].n_evaluated == 4
        assert res.best_fitness > -100.0  # the pre-scored seed stays the worst

    def test_a_model_object_seed_works_with_relative_bounds(self, tiny_data):
        # resolve_seed(backend, seed) is what extracts the architecture from a
        # *model* seed; resolve_seed(None, seed) would raise AttributeError here.
        res = evolve_architecture(
            FakeModel(Architecture((10, 10))),
            tiny_data,
            FakeBackend(),
            layers_delta=(0, 1),
            neurons_delta=(-2, 2),
            return_result=True,
            population_size=4,
            n_generations=1,
            epochs_per_candidate=1,
            random_state=0,
            stagnation_generations=None,
        )
        assert all(2 <= ind.architecture.n_layers <= 3 for ind in res.final_population)
        assert all(
            8 <= width <= 12
            for ind in res.final_population
            for width in ind.architecture.layer_sizes
        )

    def test_neurons_delta_without_layers_delta_keeps_the_default_weights(
        self, monkeypatch, tiny_data
    ):
        captured = _capture_config(monkeypatch)
        evolve_architecture(
            [10, 10],
            tiny_data,
            FakeBackend(),
            neurons_delta=(-2, 2),
            population_size=4,
            n_generations=1,
            epochs_per_candidate=1,
            random_state=0,
            stagnation_generations=None,
        )
        assert (
            captured["config"].mutation_operator_weights
            == EvolutionConfig().mutation_operator_weights
        )

    def test_a_wider_depth_range_keeps_the_structural_operators(
        self, monkeypatch, tiny_data
    ):
        captured = _capture_config(monkeypatch)
        evolve_architecture(
            [10, 10],
            tiny_data,
            FakeBackend(),
            layers_delta=(-1, 1),
            population_size=4,
            n_generations=1,
            epochs_per_candidate=1,
            random_state=0,
            stagnation_generations=None,
        )
        weights = captured["config"].mutation_operator_weights
        assert weights["add_layer"] > 0 and weights["remove_layer"] > 0

    def test_a_fixed_depth_disables_the_structural_operators(
        self, monkeypatch, tiny_data
    ):
        captured = _capture_config(monkeypatch)
        evolve_architecture(
            [10, 10],
            tiny_data,
            FakeBackend(),
            layers_delta=(0, 0),
            population_size=4,
            n_generations=1,
            epochs_per_candidate=1,
            random_state=0,
            stagnation_generations=None,
        )
        weights = captured["config"].mutation_operator_weights
        assert weights["add_layer"] == 0.0 and weights["remove_layer"] == 0.0
        assert weights["tweak_neurons"] > 0


# --------------------------------------------------------- P2: _finalize_best
class TestFinalizeBestRetrain:
    def test_final_retrain_score_replaces_the_search_score(self, tiny_data):
        res, _ = _run(
            make_cfg(
                final_training_epochs=7,
                epochs_per_candidate=2,
                population_size=5,
                n_generations=1,
            ),
            EpochScoreBackend(),
            tiny_data,
        )
        # The search scored every candidate with 2 epochs...
        assert res.history.records[0].best_fitness == 2.0
        # ...but the returned winner was retrained with 7, and that score wins.
        assert res.best_fitness == 7.0
        assert res.best_individual.metadata["raw_score"] == 7.0
        assert res.best_individual.fitness == 7.0
        assert res.best_individual.metadata["final_retrain"] is True
        assert res.history.extras["final_retrain"] == {
            "architecture": res.best_architecture.as_list(),
            "epochs": 7,
            "score_during_search": 2.0,
            "score_after": 7.0,
        }

    def test_final_retrain_respects_direction_minimize(self, tiny_data):
        res, _ = _run(
            make_cfg(
                direction="minimize",
                final_training_epochs=7,
                epochs_per_candidate=2,
                population_size=5,
                n_generations=1,
            ),
            EpochScoreBackend(),
            tiny_data,
        )
        # Reported score is the metric's own value...
        assert res.best_fitness == 7.0
        assert res.best_individual.metadata["raw_score"] == 7.0
        # ...while the engine's internal fitness is negated for "minimize".
        assert res.best_individual.fitness == -7.0
        assert res.history.extras["final_retrain"]["score_after"] == 7.0
        assert res.history.extras["final_retrain"]["score_during_search"] == 2.0
