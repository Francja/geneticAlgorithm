"""Real training (scikit-learn MLPs) -- small and fast, but not mocked."""

import warnings

import pytest

sklearn = pytest.importorskip("sklearn")

from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.neural_network import MLPClassifier, MLPRegressor  # noqa: E402

from dnnga import (  # noqa: E402
    Architecture,
    EvolutionConfig,
    GeneticArchitectureSearch,
)
from dnnga.backends import SklearnMLPBackend  # noqa: E402

FAST = dict(learning_rate_init=0.01)  # converge in few epochs on tiny data


class TestSklearnBackend:
    def test_build_returns_the_right_untrained_estimator(
        self, tiny_data, tiny_regression_data
    ):
        b = SklearnMLPBackend(**FAST)
        m = b.build(Architecture((7, 3)), tiny_data)
        assert isinstance(m, MLPClassifier)
        assert m.hidden_layer_sizes == (7, 3)
        assert not hasattr(m, "coefs_")
        assert isinstance(b.build(Architecture((4,)), tiny_regression_data), MLPRegressor)

    def test_train_returns_the_model_and_learns_a_separable_problem(self, tiny_data):
        b = SklearnMLPBackend(random_state=0, **FAST)
        model = b.build(Architecture((8,)), tiny_data)
        out = b.train(model, tiny_data, epochs=100)
        assert out is model
        assert hasattr(model, "coefs_")
        assert b.evaluate(model, tiny_data) >= 0.9

    def test_epochs_are_respected(self, tiny_data):
        b = SklearnMLPBackend(random_state=0, **FAST)
        model = b.train(b.build(Architecture((4,)), tiny_data), tiny_data, epochs=3)
        assert model.n_iter_ == 3

    def test_regression_r2_is_positive_after_training(self, tiny_regression_data):
        b = SklearnMLPBackend(random_state=0, learning_rate_init=0.02)
        model = b.train(
            b.build(Architecture((16,)), tiny_regression_data),
            tiny_regression_data,
            epochs=400,
        )
        assert b.evaluate(model, tiny_regression_data) > 0.5

    def test_convergence_warning_is_suppressed(self, tiny_data):
        b = SklearnMLPBackend(random_state=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            b.train(b.build(Architecture((4,)), tiny_data), tiny_data, epochs=2)
        assert not any(issubclass(w.category, ConvergenceWarning) for w in caught)

    @pytest.mark.parametrize(
        "kw", [{"hidden_layer_sizes": (3,)}, {"max_iter": 5}]
    )
    def test_reserved_kwargs_rejected(self, kw):
        with pytest.raises(ValueError, match="controlled by dnnga"):
            SklearnMLPBackend(**kw)

    def test_extra_kwargs_forwarded(self, tiny_data):
        m = SklearnMLPBackend(activation="tanh", alpha=0.5).build(
            Architecture((3,)), tiny_data
        )
        assert m.activation == "tanh"
        assert m.alpha == 0.5

    def test_extract_architecture_round_trip(self, tiny_data):
        b = SklearnMLPBackend()
        assert b.extract_architecture(
            b.build(Architecture((9, 5, 2)), tiny_data)
        ) == Architecture((9, 5, 2))
        assert b.extract_architecture(
            MLPClassifier(hidden_layer_sizes=6)
        ) == Architecture((6,))

    def test_extract_architecture_from_non_mlp_raises(self):
        with pytest.raises(TypeError):
            SklearnMLPBackend().extract_architecture(object())


class TestEndToEnd:
    CFG = dict(
        min_layers=1,
        max_layers=3,
        min_neurons=2,
        max_neurons=10,
        population_size=6,
        n_generations=4,
        epochs_per_candidate=40,
        final_training_epochs=150,
        stagnation_generations=None,
        random_state=0,
    )

    def test_evolves_a_network_that_solves_the_task_and_reports_a_held_out_score(
        self, tiny_data
    ):
        backend = SklearnMLPBackend(random_state=0, **FAST)
        res = GeneticArchitectureSearch(
            backend, EvolutionConfig(**self.CFG)
        ).run(tiny_data, seed=[4])
        assert isinstance(res.best_model, MLPClassifier)
        assert res.best_model.hidden_layer_sizes == res.best_architecture.as_tuple()
        assert res.best_fitness >= 0.9
        assert res.test_score >= 0.9
        assert res.history.extras["final_retrain"]["epochs"] == 150

    def test_a_real_model_object_can_be_the_seed(self, tiny_data):
        backend = SklearnMLPBackend(random_state=0, **FAST)
        seed_model = backend.train(
            backend.build(Architecture((5, 4)), tiny_data), tiny_data, epochs=60
        )
        score = backend.evaluate(seed_model, tiny_data)
        cfg = EvolutionConfig(**{**self.CFG, "final_training_epochs": None})
        res = GeneticArchitectureSearch(backend, cfg).run(
            tiny_data, seed=seed_model, seed_fitness=score
        )
        assert res.best_fitness >= score  # never worse than the incumbent
        # the seed's known score skips one training in generation 0
        assert res.history.records[0].n_evaluated == cfg.population_size - 1

    def test_regression_task(self, tiny_regression_data):
        backend = SklearnMLPBackend(random_state=0, learning_rate_init=0.02)
        cfg = EvolutionConfig(
            **{
                **self.CFG,
                "epochs_per_candidate": 150,
                "final_training_epochs": None,
            }
        )
        res = GeneticArchitectureSearch(backend, cfg).run(tiny_regression_data)
        assert res.best_fitness > 0.3

    def test_with_fixed_seeds_the_whole_real_run_is_reproducible(self, tiny_data):
        def go():
            b = SklearnMLPBackend(random_state=0, **FAST)
            return GeneticArchitectureSearch(
                b, EvolutionConfig(**self.CFG)
            ).run(tiny_data)

        a, b = go(), go()
        assert a.best_architecture == b.best_architecture
        assert a.best_fitness == b.best_fitness
        assert [r.best_so_far for r in a.history.records] == [
            r.best_so_far for r in b.history.records
        ]
