"""Issue #3 gap 6: boundary values in ``EvolutionConfig.validate``.

The existing ``test_config.py`` checks values that are clearly invalid (e.g.
``-1``, ``0``, ``1``).  These tests pin the *boundaries* themselves, so an
off-by-one mutation such as ``< 1`` -> ``< 2`` or ``< 0`` -> ``<= 0`` is
caught.
"""

import pytest

from dnnga.config import EvolutionConfig
from dnnga.exceptions import ConfigError


class TestSmallestValidValues:
    """The minimum accepted value of every numeric field must be accepted."""

    @pytest.mark.parametrize(
        "overrides",
        [
            {"population_size": 2, "tournament_size": 2},
            {"n_generations": 1},
            {"epochs_per_candidate": 1},
            {"final_training_epochs": 1},
            {"elitism_count": 0},
            {"tournament_size": 2},
            {"p_crossover": 0.0},
            {"p_crossover": 1.0},
            {"mutation_rate": 0.0},
            {"mutation_rate": 1.0},
            {"neuron_mutation_step": 1},
            {"stagnation_generations": 1},
            {"stagnation_generations": None},
            {"stagnation_min_delta": 0.0},
            {"target_fitness": 0.0},
            {"target_fitness": -1.0},
            {"target_fitness": None},
        ],
    )
    def test_minimum_is_accepted(self, overrides):
        cfg = EvolutionConfig(**overrides)
        for key, value in overrides.items():
            assert getattr(cfg, key) == value

    def test_tournament_may_equal_population_size(self):
        EvolutionConfig(population_size=2, tournament_size=2)

    def test_elitism_just_below_population_is_accepted(self):
        EvolutionConfig(population_size=2, tournament_size=2, elitism_count=1)

    def test_equal_layer_bounds_are_accepted_with_tweak_only(self):
        EvolutionConfig(
            min_layers=1,
            max_layers=1,
            mutation_operator_weights={
                "tweak_neurons": 1.0,
                "add_layer": 0.0,
                "remove_layer": 0.0,
            },
        )

    def test_equal_neuron_bounds_are_accepted(self):
        EvolutionConfig(min_neurons=5, max_neurons=5)

    def test_single_positive_weight_is_enough(self):
        EvolutionConfig(
            mutation_operator_weights={
                "tweak_neurons": 1e-12,
                "add_layer": 0.0,
                "remove_layer": 0.0,
            }
        )

    def test_zero_weight_for_an_operator_is_accepted(self):
        EvolutionConfig(
            mutation_operator_weights={
                "tweak_neurons": 1.0,
                "add_layer": 0.0,
                "remove_layer": 0.0,
            }
        )


class TestJustBelowTheMinimum:
    """One step below every boundary must be rejected, with a useful message."""

    @pytest.mark.parametrize(
        "overrides, field",
        [
            ({"population_size": 1}, "population_size"),
            ({"n_generations": 0}, "n_generations"),
            ({"epochs_per_candidate": 0}, "epochs_per_candidate"),
            ({"final_training_epochs": 0}, "final_training_epochs"),
            ({"elitism_count": -1}, "elitism_count"),
            ({"tournament_size": 1}, "tournament_size"),
            ({"p_crossover": -1e-9}, "p_crossover"),
            ({"p_crossover": 1.0 + 1e-9}, "p_crossover"),
            ({"mutation_rate": -1e-9}, "mutation_rate"),
            ({"mutation_rate": 1.0 + 1e-9}, "mutation_rate"),
            ({"neuron_mutation_step": 0}, "neuron_mutation_step"),
            ({"stagnation_generations": 0}, "stagnation_generations"),
            ({"stagnation_min_delta": -1e-9}, "stagnation_min_delta"),
            ({"min_layers": 0}, "min_layers"),
            ({"min_layers": 4, "max_layers": 3}, "max_layers"),
            ({"min_neurons": 0}, "min_neurons"),
            ({"min_neurons": 10, "max_neurons": 9}, "max_neurons"),
        ],
    )
    def test_one_step_below_is_rejected(self, overrides, field):
        with pytest.raises(ConfigError, match=field):
            EvolutionConfig(**overrides)

    def test_elitism_equal_to_population_is_rejected(self):
        with pytest.raises(ConfigError, match="elitism_count"):
            EvolutionConfig(population_size=2, elitism_count=2)

    def test_tournament_larger_than_population_is_rejected(self):
        with pytest.raises(ConfigError, match="tournament_size"):
            EvolutionConfig(population_size=2, tournament_size=3)


class TestStructuralWeightsAtFixedDepth:
    """When min_layers == max_layers, add/remove can never fire."""

    @pytest.mark.parametrize(
        "weights",
        [
            {"tweak_neurons": 1.0, "add_layer": 1e-12, "remove_layer": 0.0},
            {"tweak_neurons": 1.0, "add_layer": 0.0, "remove_layer": 1e-12},
            {"tweak_neurons": 1.0, "add_layer": 0.2},
            {"add_layer": 0.2, "remove_layer": 0.3},
        ],
    )
    def test_any_positive_structural_weight_is_rejected(self, weights):
        with pytest.raises(ConfigError, match="min_layers == max_layers"):
            EvolutionConfig(
                min_layers=3, max_layers=3, mutation_operator_weights=weights
            )

    def test_zero_structural_weights_are_accepted(self):
        EvolutionConfig(
            min_layers=3,
            max_layers=3,
            mutation_operator_weights={
                "tweak_neurons": 1.0,
                "add_layer": 0.0,
                "remove_layer": 0.0,
            },
        )


class TestWeightValidation:
    @pytest.mark.parametrize(
        "weights",
        [
            {},
            None,
            [1.0, 2.0],
            {"bogus": 1.0},
            {"tweak_neurons": -1e-12},
            {"tweak_neurons": 0.0, "add_layer": 0.0, "remove_layer": 0.0},
        ],
    )
    def test_invalid_weight_maps_are_rejected(self, weights):
        with pytest.raises(ConfigError, match="mutation_operator_weights"):
            EvolutionConfig(mutation_operator_weights=weights)


class TestEnumAndFinitenessValidation:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"direction": "sideways"},
            {"direction": ""},
            {"selection_strategy": "random"},
            {"crossover_strategy": "two_point"},
            {"mutation_boundary_mode": "reflect"},
            {"on_evaluation_error": "ignore"},
        ],
    )
    def test_unknown_enum_values_are_rejected(self, overrides):
        with pytest.raises(ConfigError):
            EvolutionConfig(**overrides)

    @pytest.mark.parametrize(
        "value", [float("nan"), float("inf"), float("-inf")]
    )
    def test_non_finite_target_is_rejected(self, value):
        with pytest.raises(ConfigError, match="target_fitness"):
            EvolutionConfig(target_fitness=value)

    @pytest.mark.parametrize(
        "strategy", ["tournament", "roulette"]
    )
    def test_known_selection_strategies_are_accepted(self, strategy):
        assert EvolutionConfig(selection_strategy=strategy).selection_strategy == strategy

    @pytest.mark.parametrize(
        "strategy", ["single_point", "uniform", "none"]
    )
    def test_known_crossover_strategies_are_accepted(self, strategy):
        assert EvolutionConfig(crossover_strategy=strategy).crossover_strategy == strategy

    @pytest.mark.parametrize("mode", ["clamp", "wrap"])
    def test_known_boundary_modes_are_accepted(self, mode):
        assert EvolutionConfig(mutation_boundary_mode=mode).mutation_boundary_mode == mode

    @pytest.mark.parametrize("policy", ["penalize", "raise", "skip"])
    def test_known_error_policies_are_accepted(self, policy):
        assert EvolutionConfig(on_evaluation_error=policy).on_evaluation_error == policy
