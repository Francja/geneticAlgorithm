"""Keras backend tests -- skipped automatically when Keras/TensorFlow is not installed."""

import logging

import numpy as np
import pytest

keras = pytest.importorskip("keras")

from dnnga import (  # noqa: E402
    Architecture,
    EvolutionConfig,
    GeneticArchitectureSearch,
    TrainingData,
    evolve_architecture,
)
from dnnga.backends import KerasMLPBackend  # noqa: E402


@pytest.fixture
def backend():
    return KerasMLPBackend(learning_rate=0.02, batch_size=16, random_state=0)


class TestKerasBackend:
    def test_build_creates_dense_stack_with_softmax_head(self, backend, tiny_data):
        m = backend.build(Architecture((6, 3)), tiny_data)
        dense = [layer for layer in m.layers if isinstance(layer, keras.layers.Dense)]
        assert [layer.units for layer in dense] == [6, 3, 2]
        assert dense[-1].activation.__name__ == "softmax"
        assert m.input_shape == (None, 3)

    def test_regression_head_is_linear_single_output(
        self, backend, tiny_regression_data
    ):
        m = backend.build(Architecture((4,)), tiny_regression_data)
        last = m.layers[-1]
        assert last.units == 1
        assert last.activation.__name__ == "linear"

    def test_train_returns_the_MODEL_not_the_History_object(self, backend, tiny_data):
        # The trap documented in ModelBackend.train: keras fit() returns a History.
        model = backend.build(Architecture((4,)), tiny_data)
        out = backend.train(model, tiny_data, epochs=2)
        assert out is model
        assert not isinstance(out, keras.callbacks.History)

    def test_learns_a_separable_problem(self, backend, tiny_data):
        model = backend.train(
            backend.build(Architecture((8,)), tiny_data), tiny_data, epochs=40
        )
        assert backend.evaluate(model, tiny_data) >= 0.9

    def test_regression_r2(self, backend, tiny_regression_data):
        model = backend.train(
            backend.build(Architecture((16,)), tiny_regression_data),
            tiny_regression_data,
            epochs=120,
        )
        assert backend.evaluate(model, tiny_regression_data) > 0.5

    def test_arbitrary_class_labels_are_mapped(self, backend):
        rng = np.random.default_rng(1)
        X = np.vstack(
            [rng.normal(-2, 0.4, (30, 2)), rng.normal(2, 0.4, (30, 2))]
        )
        y = np.array([3] * 30 + [7] * 30)  # labels are not 0..K-1
        idx = rng.permutation(60)
        data = TrainingData(X[idx][:40], y[idx][:40], X[idx][40:], y[idx][40:])
        model = backend.train(
            backend.build(Architecture((6,)), data), data, epochs=40
        )
        assert backend.evaluate(model, data) >= 0.9

    def test_unseen_validation_label_counts_as_wrong_not_a_crash(
        self, backend, tiny_data
    ):
        bad = TrainingData(
            tiny_data.X_train,
            tiny_data.y_train,
            tiny_data.X_val[:4],
            np.array([5, 5, 5, 5]),
        )
        model = backend.train(backend.build(Architecture((3,)), bad), bad, epochs=1)
        assert backend.evaluate(model, bad) == 0.0

    def test_extract_architecture_round_trip(self, backend, tiny_data):
        assert backend.extract_architecture(
            backend.build(Architecture((9, 5, 2)), tiny_data)
        ) == Architecture((9, 5, 2))

    def test_extract_architecture_needs_a_hidden_layer(self, backend):
        m = keras.Sequential(
            [keras.Input(shape=(3,)), keras.layers.Dense(2)]
        )
        with pytest.raises(ValueError, match="hidden"):
            backend.extract_architecture(m)

    def test_unknown_optimizer_rejected(self):
        with pytest.raises(ValueError):
            KerasMLPBackend(optimizer="bogus")

    @pytest.mark.parametrize("opt", ["adam", "sgd", "rmsprop", "adagrad"])
    def test_supported_optimizers_build_and_train(self, opt, tiny_data):
        b = KerasMLPBackend(optimizer=opt, batch_size=16)
        b.train(b.build(Architecture((3,)), tiny_data), tiny_data, epochs=1)


class TestKerasEndToEnd:
    def test_evolves_a_keras_model_and_can_start_from_one(self, backend, tiny_data):
        seed_model = backend.train(
            backend.build(Architecture((5, 4)), tiny_data), tiny_data, epochs=30
        )
        result = evolve_architecture(
            seed_model,
            tiny_data,
            backend,
            return_result=True,
            layers_delta=(-1, 1),
            neurons_delta=(-2, 4),
            population_size=4,
            n_generations=2,
            epochs_per_candidate=15,
            random_state=0,
            stagnation_generations=None,
        )
        assert isinstance(result.best_model, keras.Model)
        got = backend.extract_architecture(result.best_model)
        assert got == result.best_architecture
        assert 1 <= got.n_layers <= 3
        assert all(2 <= w <= 9 for w in got.layer_sizes)
        assert result.best_fitness >= 0.9
        assert result.test_score >= 0.9

    def test_final_retrain_with_keras(self, backend, tiny_data):
        cfg = EvolutionConfig(
            min_layers=1,
            max_layers=2,
            max_neurons=6,
            population_size=4,
            n_generations=2,
            epochs_per_candidate=2,
            final_training_epochs=40,
            random_state=0,
            stagnation_generations=None,
        )
        res = GeneticArchitectureSearch(backend, cfg).run(tiny_data)
        assert res.history.extras["final_retrain"]["score_after"] >= 0.9


class TestKerasEdgeCases:
    def test_regression_r2_on_constant_validation_targets_does_not_divide_by_zero(
        self, backend
    ):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(30, 2))
        data = TrainingData(X, X[:, 0], X[:5], np.zeros(5), task="regression")
        model = backend.train(backend.build(Architecture((3,)), data), data, epochs=1)
        score = backend.evaluate(model, data)
        assert np.isfinite(score)
        assert score in (0.0, 1.0)


class TestKerasExtractGuards:
    def test_convolutional_models_are_refused_not_silently_flattened(self, backend):
        m = keras.Sequential(
            [
                keras.Input(shape=(8, 8, 1)),
                keras.layers.Conv2D(4, 3),
                keras.layers.Flatten(),
                keras.layers.Dense(5),
                keras.layers.Dense(2),
            ]
        )
        with pytest.raises(ValueError, match="Conv2D"):
            backend.extract_architecture(m)

    def test_dropout_is_tolerated_but_the_loss_is_announced(self, backend, caplog):
        m = keras.Sequential(
            [
                keras.Input(shape=(3,)),
                keras.layers.Dense(6),
                keras.layers.Dropout(0.1),
                keras.layers.Dense(4),
                keras.layers.Dense(2),
            ]
        )
        with caplog.at_level(logging.WARNING, logger="dnnga"):
            arch = backend.extract_architecture(m)
        assert arch == Architecture((6, 4))
        assert any("Dropout" in r.getMessage() for r in caplog.records)
