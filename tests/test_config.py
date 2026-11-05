import pytest

from dnnga.architecture import Architecture
from dnnga.config import EvolutionConfig, bounds_relative_to
from dnnga.exceptions import ConfigError


class TestValid:
    def test_defaults(self):
        EvolutionConfig()

    def test_custom(self):
        EvolutionConfig(
            min_layers=1,
            max_layers=8,
            min_neurons=2,
            max_neurons=128,
            population_size=30,
            n_generations=50,
            elitism_count=2,
            tournament_size=4,
            target_fitness=0.95,
            stagnation_generations=None,
            direction="minimize",
            selection_strategy="roulette",
        )

    def test_elitism_zero_allowed(self):
        EvolutionConfig(elitism_count=0)

    def test_weights_dict_is_copied(self):
        w = {"tweak_neurons": 1.0, "add_layer": 0.0, "remove_layer": 0.0}
        cfg = EvolutionConfig(mutation_operator_weights=w)
        w["tweak_neurons"] = 99
        assert cfg.mutation_operator_weights["tweak_neurons"] == 1.0


class TestInvalid:
    @pytest.mark.parametrize(
        "kw",
        [
            dict(min_layers=0),
            dict(min_layers=5, max_layers=2),
            dict(min_neurons=0),
            dict(min_neurons=50, max_neurons=10),
            dict(population_size=1),
            dict(n_generations=0),
            dict(epochs_per_candidate=0),
            dict(final_training_epochs=0),
            dict(elitism_count=-1),
            dict(elitism_count=20, population_size=20),
            dict(tournament_size=1),
            dict(tournament_size=25, population_size=20),
            dict(p_crossover=1.5),
            dict(mutation_rate=-0.1),
            dict(neuron_mutation_step=0),
            dict(mutation_operator_weights={"bogus": 1.0}),
            dict(
                mutation_operator_weights={
                    "tweak_neurons": 0,
                    "add_layer": 0,
                    "remove_layer": 0,
                }
            ),
            dict(mutation_operator_weights={"tweak_neurons": -1}),
            dict(mutation_operator_weights={}),
            dict(mutation_operator_weights=None),
            dict(mutation_operator_weights=[1, 2]),
            dict(stagnation_generations=0),
            dict(stagnation_min_delta=-1),
            dict(selection_strategy="x"),
            dict(crossover_strategy="x"),
            dict(on_evaluation_error="x"),
            dict(mutation_boundary_mode="x"),
            dict(direction="sideways"),
            dict(target_fitness=float("nan")),
            dict(target_fitness=float("inf")),
        ],
    )
    def test_raises(self, kw):
        with pytest.raises(ConfigError):
            EvolutionConfig(**kw)

    def test_invalid_direction_caught_even_without_target(self):
        # regression: direction used to be validated only when target_fitness was set
        with pytest.raises(ConfigError):
            EvolutionConfig(direction="up", target_fitness=None)

    def test_min_equals_max_layers_with_structural_weights_raises(self):
        with pytest.raises(ConfigError):
            EvolutionConfig(min_layers=4, max_layers=4)

    def test_min_equals_max_layers_with_only_tweak_is_fine(self):
        EvolutionConfig(
            min_layers=4,
            max_layers=4,
            mutation_operator_weights={
                "tweak_neurons": 1.0,
                "add_layer": 0.0,
                "remove_layer": 0.0,
            },
        )

    def test_all_problems_reported_at_once(self):
        with pytest.raises(ConfigError) as ei:
            EvolutionConfig(min_layers=0, population_size=1, mutation_rate=5.0)
        msg = str(ei.value)
        assert "min_layers" in msg and "population_size" in msg and "mutation_rate" in msg

    def test_validate_catches_later_mutation(self):
        cfg = EvolutionConfig()
        cfg.population_size = 1
        with pytest.raises(ConfigError):
            cfg.validate()


class TestOverrides:
    def test_with_overrides(self):
        cfg = EvolutionConfig().with_overrides(population_size=7, n_generations=3)
        assert (cfg.population_size, cfg.n_generations) == (7, 3)

    def test_unknown_field_raises_and_lists_valid_ones(self):
        with pytest.raises(ConfigError, match="population_size"):
            EvolutionConfig().with_overrides(populaton_size=7)

    def test_overrides_are_revalidated(self):
        with pytest.raises(ConfigError):
            EvolutionConfig().with_overrides(population_size=1)


class TestRelativeBounds:
    def test_layers_and_neurons(self):
        b = bounds_relative_to(
            Architecture((64, 32, 16)), layers_delta=(-1, 2), neurons_delta=(-8, 16)
        )
        assert b == dict(
            min_layers=2, max_layers=5, min_neurons=8, max_neurons=80
        )

    def test_floors_at_one(self):
        b = bounds_relative_to(
            Architecture((2,)), layers_delta=(-5, 0), neurons_delta=(-10, 0)
        )
        assert b["min_layers"] == 1 and b["min_neurons"] == 1

    def test_only_requested_keys_returned(self):
        assert set(bounds_relative_to(Architecture((4, 4)), layers_delta=(0, 1))) == {
            "min_layers",
            "max_layers",
        }

    @pytest.mark.parametrize("bad", [(1, 2), (-1, -1), "ab", (1,), 5])
    def test_bad_deltas(self, bad):
        with pytest.raises(ConfigError):
            bounds_relative_to(Architecture((4,)), layers_delta=bad)

    def test_seed_always_inside_result(self):
        seed = Architecture((10, 3, 7))
        b = bounds_relative_to(seed, layers_delta=(0, 0), neurons_delta=(0, 0))
        assert b["min_layers"] <= seed.n_layers <= b["max_layers"]
        assert b["min_neurons"] <= min(seed.layer_sizes)
        assert max(seed.layer_sizes) <= b["max_neurons"]
