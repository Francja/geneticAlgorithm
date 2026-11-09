import random
from collections import Counter

import pytest

from dnnga.architecture import Architecture
from dnnga.config import EvolutionConfig
from dnnga.exceptions import InvalidArchitectureError
from dnnga.individual import Individual
from dnnga.operators import (
    crossover_single_point,
    crossover_uniform,
    maybe_crossover,
    maybe_mutate,
    mutate_add_layer,
    mutate_remove_layer,
    mutate_tweak_neurons,
    select,
    select_roulette,
    select_tournament,
)


def _pop(fitnesses):
    return [
        Individual(architecture=Architecture((i + 1,)), fitness=f)
        for i, f in enumerate(fitnesses)
    ]


class TestTweakNeurons:
    def test_changes_exactly_one_layer_and_keeps_depth(self):
        cfg = EvolutionConfig(min_neurons=1, max_neurons=100, neuron_mutation_step=3)
        orig = Architecture((10, 10, 10, 10))
        m = mutate_tweak_neurons(orig, cfg, random.Random(1))
        assert sum(a != b for a, b in zip(m.layer_sizes, orig.layer_sizes)) == 1
        assert m.n_layers == 4

    def test_original_untouched(self):
        cfg = EvolutionConfig(min_neurons=1, max_neurons=100)
        orig = Architecture((10, 10))
        mutate_tweak_neurons(orig, cfg, random.Random(1))
        assert orig.layer_sizes == (10, 10)

    def test_step_size_respected(self):
        cfg = EvolutionConfig(min_neurons=1, max_neurons=1000, neuron_mutation_step=5)
        rng = random.Random(2)
        for _ in range(200):
            m = mutate_tweak_neurons(Architecture((500,)), cfg, rng)
            assert 1 <= abs(m.layer_sizes[0] - 500) <= 5

    @pytest.mark.parametrize("mode", ["clamp", "wrap"])
    def test_never_leaves_bounds(self, mode):
        cfg = EvolutionConfig(
            min_neurons=1,
            max_neurons=10,
            neuron_mutation_step=4,
            mutation_boundary_mode=mode,
        )
        rng = random.Random(3)
        for start in (1, 10):
            arch = Architecture((start,))
            for _ in range(300):
                arch = mutate_tweak_neurons(arch, cfg, rng)
                assert 1 <= arch.layer_sizes[0] <= 10

    def test_clamp_stops_at_the_edge_while_wrap_jumps_across(self):
        clamp = EvolutionConfig(
            min_neurons=1,
            max_neurons=10,
            neuron_mutation_step=1,
            mutation_boundary_mode="clamp",
        )
        wrap = EvolutionConfig(
            min_neurons=1,
            max_neurons=10,
            neuron_mutation_step=1,
            mutation_boundary_mode="wrap",
        )
        rng_c, rng_w = random.Random(5), random.Random(5)
        outs_c = {
            mutate_tweak_neurons(Architecture((10,)), clamp, rng_c).layer_sizes[0]
            for _ in range(100)
        }
        outs_w = {
            mutate_tweak_neurons(Architecture((10,)), wrap, rng_w).layer_sizes[0]
            for _ in range(100)
        }
        assert outs_c == {9, 10}
        assert outs_w == {9, 1}


class TestAddRemove:
    def test_add_layer_depth_plus_one_and_width_in_bounds(self):
        cfg = EvolutionConfig(min_neurons=3, max_neurons=7)
        rng = random.Random(6)
        for _ in range(200):
            m = mutate_add_layer(Architecture((4,)), cfg, rng)
            assert m.n_layers == 2
            rest = list(m.layer_sizes)
            rest.remove(4)
            assert 3 <= rest[0] <= 7

    def test_add_layer_can_insert_at_every_position(self):
        cfg = EvolutionConfig(min_neurons=100, max_neurons=100)
        rng = random.Random(7)
        positions = {
            mutate_add_layer(Architecture((1, 2)), cfg, rng).layer_sizes.index(100)
            for _ in range(200)
        }
        assert positions == {0, 1, 2}

    def test_remove_layer(self):
        m = mutate_remove_layer(Architecture((4, 5, 6)), EvolutionConfig(), random.Random(8))
        assert m.n_layers == 2

    def test_remove_from_single_layer_is_invalid(self):
        with pytest.raises(InvalidArchitectureError):
            mutate_remove_layer(Architecture((4,)), EvolutionConfig(), random.Random(8))


class TestMaybeMutate:
    def test_depth_and_width_bounds_hold_over_many_events(self):
        cfg = EvolutionConfig(
            min_layers=1,
            max_layers=4,
            min_neurons=2,
            max_neurons=8,
            mutation_rate=1.0,
            neuron_mutation_step=6,
            mutation_operator_weights={
                "tweak_neurons": 0.2,
                "add_layer": 0.4,
                "remove_layer": 0.4,
            },
        )
        rng = random.Random(9)
        arch = Architecture((2,))
        depths = set()
        for _ in range(1000):
            arch, _ = maybe_mutate(arch, cfg, rng)
            assert 1 <= arch.n_layers <= 4 and all(2 <= w <= 8 for w in arch.layer_sizes)
            depths.add(arch.n_layers)
        assert depths == {1, 2, 3, 4}, "depth should be able to reach every allowed value"

    def test_rate_zero_never_mutates(self):
        cfg = EvolutionConfig(mutation_rate=0.0)
        arch = Architecture((5, 5))
        for _ in range(50):
            new, did = maybe_mutate(arch, cfg, random.Random(11))
            assert not did and new == arch

    def test_rate_one_always_mutates_when_feasible(self):
        cfg = EvolutionConfig(mutation_rate=1.0)
        rng = random.Random(12)
        assert all(
            maybe_mutate(Architecture((5, 5)), cfg, rng)[1] for _ in range(50)
        )

    def test_nothing_feasible_is_a_noop(self):
        cfg = EvolutionConfig(
            min_layers=1,
            max_layers=3,
            mutation_rate=1.0,
            mutation_operator_weights={
                "tweak_neurons": 0.0,
                "add_layer": 1.0,
                "remove_layer": 0.0,
            },
        )
        arch = Architecture((5, 5, 5))
        new, did = maybe_mutate(arch, cfg, random.Random(13))
        assert not did and new == arch

    def test_operator_mix_follows_weights(self):
        cfg = EvolutionConfig(
            min_layers=1,
            max_layers=10,
            mutation_rate=1.0,
            min_neurons=1,
            max_neurons=50,
            mutation_operator_weights={
                "tweak_neurons": 0.0,
                "add_layer": 1.0,
                "remove_layer": 0.0,
            },
        )
        rng = random.Random(14)
        assert all(
            maybe_mutate(Architecture((5, 5)), cfg, rng)[0].n_layers == 3
            for _ in range(30)
        )


class TestCrossover:
    def test_single_point_swaps_depths_like_the_thesis_example(self):
        p1, p2 = Architecture((1, 2, 3)), Architecture((6, 7, 8, 9, 10, 11))
        for seed in range(20):
            c1, c2 = crossover_single_point(p1, p2, random.Random(seed))
            assert (c1.n_layers, c2.n_layers) == (6, 3)

    def test_single_point_prefix_suffix_structure(self):
        p1, p2 = Architecture((1, 2, 3, 4)), Architecture((5, 6, 7, 8, 9))
        c1, c2 = crossover_single_point(p1, p2, random.Random(3))
        k = next(
            i
            for i in range(1, 4)
            if c1.layer_sizes[:i] == p1.layer_sizes[:i]
            and c1.layer_sizes[i:] == p2.layer_sizes[i:]
        )
        assert c2.layer_sizes == p2.layer_sizes[:k] + p1.layer_sizes[k:]

    def test_parents_unaffected_by_editing_children(self):
        # Regression for the original aliasing bug.
        p1, p2 = Architecture((1, 2, 3)), Architecture((4, 5, 6))
        c1, c2 = crossover_single_point(p1, p2, random.Random(21))
        c1.replace_layer(0, 999)
        c2.replace_layer(0, 999)
        assert p1.layer_sizes == (1, 2, 3) and p2.layer_sizes == (4, 5, 6)

    def test_single_layer_parents_swap(self):
        p1, p2 = Architecture((3,)), Architecture((7,))
        assert crossover_single_point(p1, p2, random.Random(22)) == (p2, p1)

    def test_single_point_depths_stay_within_parent_depths(self):
        rng = random.Random(23)
        p1, p2 = Architecture(tuple(range(1, 6))), Architecture((10, 11, 12))
        for _ in range(200):
            c1, c2 = crossover_single_point(p1, p2, rng)
            assert {c1.n_layers, c2.n_layers} == {5, 3}

    def test_uniform_preserves_each_depth_and_genes_come_from_parents(self):
        p1, p2 = Architecture((1, 2, 3)), Architecture((10, 20, 30, 40, 50))
        rng = random.Random(24)
        for _ in range(100):
            c1, c2 = crossover_uniform(p1, p2, rng)
            assert (c1.n_layers, c2.n_layers) == (3, 5), "no drift in depth"
            assert c1.layer_sizes[3:] == () and c2.layer_sizes[3:] == (40, 50)
            for i in range(3):
                assert {c1.layer_sizes[i], c2.layer_sizes[i]} == {
                    p1.layer_sizes[i],
                    p2.layer_sizes[i],
                }

    def test_uniform_mean_depth_does_not_drift(self):
        rng = random.Random(25)
        p1, p2 = Architecture((1,) * 2), Architecture((1,) * 6)
        depths = [
            d
            for _ in range(100)
            for d in map(lambda a: a.n_layers, crossover_uniform(p1, p2, rng))
        ]
        assert sum(depths) / len(depths) == 4.0

    def test_maybe_crossover_probability_zero_and_none(self):
        p1, p2 = Architecture((1, 2)), Architecture((3, 4))
        assert maybe_crossover(
            p1, p2, EvolutionConfig(p_crossover=0.0), random.Random(1)
        ) == (p1, p2)
        assert maybe_crossover(
            p1,
            p2,
            EvolutionConfig(p_crossover=1.0, crossover_strategy="none"),
            random.Random(1),
        ) == (p1, p2)

    def test_maybe_crossover_probability_one_changes_things(self):
        p1, p2 = Architecture((1, 2, 3)), Architecture((4, 5, 6))
        outs = {
            maybe_crossover(p1, p2, EvolutionConfig(p_crossover=1.0), random.Random(s))
            for s in range(20)
        }
        assert any(o != (p1, p2) for o in outs)


class TestTournament:
    def test_count(self):
        assert (
            len(select_tournament(_pop(range(10)), 7, EvolutionConfig(tournament_size=3), random.Random(1)))
            == 7
        )

    def test_tournament_as_big_as_population_always_picks_the_best(self):
        cfg = EvolutionConfig(tournament_size=10, population_size=10)
        sel = select_tournament(
            _pop([0, 1, 2, 3, 4, 5, 6, 7, 8, 100]), 20, cfg, random.Random(31)
        )
        assert all(i.fitness == 100 for i in sel)

    def test_biased_towards_fitter(self):
        pop = _pop([0, 1, 2, 3, 4, 100])
        sel = select_tournament(
            pop, 300, EvolutionConfig(tournament_size=2, population_size=6), random.Random(32)
        )
        assert sum(i.fitness for i in sel) / 300 > sum(i.fitness for i in pop) / 6

    def test_unevaluated_raises(self):
        with pytest.raises(ValueError):
            select_tournament(
                [Individual(Architecture((1,)))], 1, EvolutionConfig(), random.Random(0)
            )

    def test_tolerates_minus_infinity(self):
        pop = _pop([float("-inf"), 1.0, float("-inf"), 2.0])
        sel = select_tournament(
            pop, 50, EvolutionConfig(tournament_size=2, population_size=4), random.Random(2)
        )
        assert len(sel) == 50


class TestRoulette:
    def test_count(self):
        assert (
            len(select_roulette(_pop([1, 2, 3]), 12, EvolutionConfig(), random.Random(33)))
            == 12
        )

    def test_biased_towards_fitter(self):
        pop = _pop([1, 1, 1, 1, 1, 50])
        sel = select_roulette(pop, 500, EvolutionConfig(), random.Random(34))
        assert sum(i.fitness for i in sel) / 500 > sum(i.fitness for i in pop) / 6

    def test_negative_fitness_shifted_and_proportions_correct(self):
        # fitness [-10, -5, -1, 0] -> shifted weights ~[0, 5, 9, 10] -> p ~ [0, .208, .375, .417].
        n = 20000
        sel = select_roulette(_pop([-10, -5, -1, 0]), n, EvolutionConfig(), random.Random(35))
        freq = {
            f: c / n for f, c in Counter(i.fitness for i in sel).items()
        }
        assert freq.get(-10, 0.0) < 0.001
        assert freq[-5] == pytest.approx(5 / 24, abs=0.02)
        assert freq[-1] == pytest.approx(9 / 24, abs=0.02)
        assert freq[0] == pytest.approx(10 / 24, abs=0.02)
        assert freq[0] > freq[-1] > freq[-5]

    def test_survives_penalised_minus_infinity(self):
        # Regression: -inf - (-inf) is NaN, which used to break random.choices for everyone.
        sel = select_roulette(
            _pop([0.5, 0.8, float("-inf"), 0.2]), 200, EvolutionConfig(), random.Random(36)
        )
        assert len(sel) == 200 and all(i.fitness != float("-inf") for i in sel)

    def test_all_failed_raises_clear_error(self):
        with pytest.raises(ValueError, match="non-finite"):
            select_roulette(
                _pop([float("-inf")] * 3), 5, EvolutionConfig(), random.Random(37)
            )

    def test_never_returns_short_list_regression(self):
        # The original could silently return fewer than requested (float rounding
        # of the cumulative table).
        for seed in range(200):
            assert (
                len(select_roulette(_pop([0.1] * 7), 10, EvolutionConfig(), random.Random(seed)))
                == 10
            )


class TestDispatch:
    @pytest.mark.parametrize("strategy", ["tournament", "roulette"])
    def test_select_uses_configured_strategy(self, strategy):
        cfg = EvolutionConfig(selection_strategy=strategy, tournament_size=2)
        assert len(select(_pop([1, 2, 3, 4, 5]), 4, cfg, random.Random(0))) == 4
