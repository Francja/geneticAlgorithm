"""Issue #3 gaps 4-5: mutation-operator weights and crossover/selection edges.

These complement ``test_operators.py``; they target mutants that survived the
suite at ``f909193`` (mutmut 3.8.1, Python 3.12).
"""

import math
import random

import pytest

from dnnga.architecture import Architecture
from dnnga.config import EvolutionConfig
from dnnga.individual import Individual
from dnnga.operators import (
    _feasible_operators,
    crossover_single_point,
    crossover_uniform,
    maybe_mutate,
    select_roulette,
)

# P4: mutation-operator weights


def _weights_cfg(**overrides):
    base = dict(
        min_layers=1,
        max_layers=5,
        min_neurons=1,
        max_neurons=50,
        mutation_rate=1.0,
        mutation_operator_weights={
            "tweak_neurons": 1.0,
            "add_layer": 2.0,
            "remove_layer": 3.0,
        },
    )
    base.update(overrides)
    return EvolutionConfig(**base)


class TestFeasibleOperatorWeights:
    def test_zero_weight_keys_are_dropped_not_kept_at_zero(self):
        cfg = _weights_cfg(
            mutation_operator_weights={
                "tweak_neurons": 0.0,
                "add_layer": 1.0,
                "remove_layer": 0.0,
            }
        )
        weights = _feasible_operators(Architecture((4, 4)), cfg)
        assert weights == {"add_layer": 1.0}

    def test_weights_are_divided_by_total_not_multiplied(self):
        weights = _feasible_operators(Architecture((4, 4)), _weights_cfg())
        assert weights == pytest.approx(
            {
                "tweak_neurons": 1.0 / 6.0,
                "add_layer": 2.0 / 6.0,
                "remove_layer": 3.0 / 6.0,
            }
        )
        assert sum(weights.values()) == pytest.approx(1.0)
        assert max(weights.values()) <= 1.0

    def test_structural_operator_zeroed_at_upper_bound_and_rest_renormalised(self):
        cfg = _weights_cfg(
            min_layers=2,
            max_layers=4,
            mutation_operator_weights={
                "tweak_neurons": 1.0,
                "add_layer": 1.0,
                "remove_layer": 1.0,
            },
        )
        weights = _feasible_operators(Architecture((5, 5, 5, 5)), cfg)
        assert "add_layer" not in weights
        assert weights == pytest.approx(
            {"tweak_neurons": 0.5, "remove_layer": 0.5}
        )

    def test_structural_operator_zeroed_at_lower_bound_and_rest_renormalised(self):
        cfg = _weights_cfg(
            min_layers=2,
            max_layers=4,
            mutation_operator_weights={
                "tweak_neurons": 1.0,
                "add_layer": 1.0,
                "remove_layer": 1.0,
            },
        )
        weights = _feasible_operators(Architecture((5, 5)), cfg)
        assert "remove_layer" not in weights
        assert weights == pytest.approx({"tweak_neurons": 0.5, "add_layer": 0.5})

    def test_no_feasible_operator_returns_empty_dict(self):
        cfg = _weights_cfg(
            min_layers=1,
            max_layers=3,
            mutation_operator_weights={
                "tweak_neurons": 0.0,
                "add_layer": 1.0,
                "remove_layer": 0.0,
            },
        )
        assert _feasible_operators(Architecture((5, 5, 5)), cfg) == {}

    def test_zero_weight_operator_never_selected(self):
        cfg = _weights_cfg(
            mutation_operator_weights={
                "tweak_neurons": 1.0,
                "add_layer": 0.0,
                "remove_layer": 0.0,
            }
        )
        rng = random.Random(40)
        for _ in range(500):
            child, did = maybe_mutate(Architecture((5, 5)), cfg, rng)
            assert did
            assert child.n_layers == 2, "add_layer has weight 0 and must not fire"

    def test_operator_frequencies_follow_configured_weights(self):
        def add_fraction(tweak_weight, add_weight, seed):
            cfg = _weights_cfg(
                mutation_operator_weights={
                    "tweak_neurons": tweak_weight,
                    "add_layer": add_weight,
                    "remove_layer": 0.0,
                }
            )
            rng = random.Random(seed)
            draws = 1000
            adds = sum(
                maybe_mutate(Architecture((5,)), cfg, rng)[0].n_layers == 2
                for _ in range(draws)
            )
            return adds / draws

        # 10% vs 90%.  If rng.choices() ignored ``weights=`` both would be ~50%.
        assert add_fraction(0.9, 0.1, 41) == pytest.approx(0.1, abs=0.04)
        assert add_fraction(0.1, 0.9, 42) == pytest.approx(0.9, abs=0.04)


# P5: crossover and selection edges


class _SeqRng:
    """Minimal rng stub returning a fixed sequence from ``random()``."""

    def __init__(self, values):
        self.values = list(values)

    def random(self):
        return self.values.pop(0)


class _RecordingRng:
    """Records the weights passed to ``choices`` and selects the first item."""

    def __init__(self):
        self.weight_calls = []

    def choices(self, population, weights=None, k=1):
        self.weight_calls.append(list(weights))
        return [population[0]] * k


def _pop(fitnesses):
    return [
        Individual(architecture=Architecture((i + 1,)), fitness=f)
        for i, f in enumerate(fitnesses)
    ]


class TestCrossoverEdges:
    def test_single_point_two_layer_parents_cut_at_the_only_interior_point(self):
        p1, p2 = Architecture((1, 2)), Architecture((3, 4))
        c1, c2 = crossover_single_point(p1, p2, random.Random(43))
        assert c1.layer_sizes == (1, 4)
        assert c2.layer_sizes == (3, 2)
        assert (c1, c2) != (p2, p1), "two 2-layer parents must mix, not just swap"

    def test_uniform_mixes_genes_position_by_position(self):
        p1, p2 = Architecture((1, 2, 3)), Architecture((10, 20, 30))
        # 0.1 -> swap position 0, 0.9 -> keep position 1, 0.1 -> swap position 2.
        c1, c2 = crossover_uniform(p1, p2, _SeqRng([0.1, 0.9, 0.1]))
        assert c1.layer_sizes == (10, 2, 30)
        assert c2.layer_sizes == (1, 20, 3)
        assert c1 != p1 and c1 != p2
        assert c2 != p1 and c2 != p2

    def test_uniform_at_exactly_half_does_not_swap(self):
        p1, p2 = Architecture((1,)), Architecture((9,))
        c1, c2 = crossover_uniform(p1, p2, _SeqRng([0.5]))
        assert (c1.layer_sizes, c2.layer_sizes) == ((1,), (9,))


class TestRouletteShiftBoundary:
    def test_positive_minimum_gets_only_the_epsilon_shift(self):
        rng = _RecordingRng()
        select_roulette(_pop([0.5, 0.8]), 2, EvolutionConfig(), rng)
        assert rng.weight_calls[0] == pytest.approx([0.5 + 1e-9, 0.8 + 1e-9])

    def test_zero_minimum_gets_only_the_epsilon_shift(self):
        rng = _RecordingRng()
        select_roulette(_pop([0.0, 0.8]), 2, EvolutionConfig(), rng)
        assert rng.weight_calls[0] == pytest.approx([1e-9, 0.8 + 1e-9])

    def test_negative_minimum_shifts_every_finite_fitness(self):
        rng = _RecordingRng()
        select_roulette(_pop([-2.0, -0.5, 1.0]), 2, EvolutionConfig(), rng)
        assert rng.weight_calls[0] == pytest.approx(
            [1e-9, 1.5 + 1e-9, 3.0 + 1e-9]
        )

    def test_minus_infinity_gets_zero_weight_without_poisoning_the_shift(self):
        rng = _RecordingRng()
        select_roulette(
            _pop([float("-inf"), 0.5]), 2, EvolutionConfig(), rng
        )
        weights = rng.weight_calls[0]
        assert weights[0] == 0.0
        assert weights[1] == pytest.approx(0.5 + 1e-9)
        assert all(math.isfinite(w) for w in weights)
