"""Small edge cases that do not fit the other files: reprs, backend defaults,
and robustness against buggy backend implementations."""

import pytest

from dnnga import (
    Architecture,
    EvaluationError,
    GeneticArchitectureSearch,
    Individual,
)
from dnnga.backends.base import ModelBackend
from tests.conftest import FakeBackend, FakeModel, make_cfg


class TestReprs:
    def test_architecture_and_individual(self):
        a = Architecture((4, 2))
        assert repr(a) == "Architecture([4, 2])"
        assert str(a) == "[4, 2]"
        assert "unevaluated" in repr(Individual(a))
        assert "0.5000" in repr(Individual(a, fitness=0.5))

    def test_evolution_result(self, tiny_data):
        res = GeneticArchitectureSearch(FakeBackend(), make_cfg(n_generations=2)).run(
            tiny_data
        )
        text = repr(res)
        assert "best_architecture" in text
        assert "test_score" in text
        assert "max_generations" in text


class TestBackendBaseDefaults:
    def test_abstract_methods_must_be_implemented(self):
        with pytest.raises(TypeError):
            ModelBackend()  # type: ignore[abstract]

    def test_default_extract_architecture_explains_the_workaround(self):
        class Minimal(ModelBackend):
            build = train = evaluate = lambda *a, **k: None  # noqa: E731

        with pytest.raises(NotImplementedError, match="seed as a list"):
            Minimal().extract_architecture(object())

    def test_default_cleanup_is_a_noop(self):
        class Minimal(ModelBackend):
            build = train = evaluate = lambda *a, **k: None  # noqa: E731

        assert Minimal().cleanup(object()) is None

    def test_seed_model_object_with_unsupported_backend_gives_a_clear_error(
        self, tiny_data
    ):
        class NoExtract(FakeBackend):
            extract_architecture = ModelBackend.extract_architecture

        with pytest.raises(NotImplementedError, match="seed as a list"):
            GeneticArchitectureSearch(NoExtract(), make_cfg()).run(
                tiny_data, seed=FakeModel(Architecture((3,)))
            )


class TestRobustnessAgainstBuggyBackends:
    def test_a_cleanup_hook_that_raises_does_not_kill_the_run(self, tiny_data):
        class BadCleanup(FakeBackend):
            def cleanup(self, model):
                raise RuntimeError("boom")

        res = GeneticArchitectureSearch(BadCleanup(), make_cfg(n_generations=4)).run(
            tiny_data
        )
        assert res.best_model is not None

    def test_non_finite_score_after_final_retrain_falls_back_to_the_search_model(
        self, tiny_data
    ):
        class NanOnFinal(FakeBackend):
            def evaluate(self, model, data):
                return (
                    float("nan")
                    if model.trained_epochs == 77
                    else super().evaluate(model, data)
                )

        res = GeneticArchitectureSearch(
            NanOnFinal(), make_cfg(final_training_epochs=77)
        ).run(tiny_data)
        assert res.best_model.trained_epochs == 5

    def test_non_finite_final_score_is_fatal_under_raise_policy(self, tiny_data):
        class NanOnFinal(FakeBackend):
            def evaluate(self, model, data):
                return (
                    float("nan")
                    if model.trained_epochs == 77
                    else super().evaluate(model, data)
                )

        with pytest.raises(EvaluationError):
            GeneticArchitectureSearch(
                NanOnFinal(),
                make_cfg(final_training_epochs=77, on_evaluation_error="raise"),
            ).run(tiny_data)

    def test_train_that_forgets_to_return_the_model_is_diagnosed_by_name(
        self, tiny_data
    ):
        # The documented trap (keras `return model.fit(...)` is the sibling
        # mistake: it returns a History object, not the model).
        class ForgetsToReturn(FakeBackend):
            def train(self, model, data, epochs):
                super().train(model, data, epochs)  # returns None implicitly

        with pytest.raises(EvaluationError, match="must return the trained model"):
            GeneticArchitectureSearch(
                ForgetsToReturn(), make_cfg(on_evaluation_error="raise")
            ).run(tiny_data)

    def test_under_penalize_the_same_mistake_ends_in_a_loud_run_level_error(
        self, tiny_data
    ):
        class ForgetsToReturn(FakeBackend):
            def train(self, model, data, epochs):
                super().train(model, data, epochs)

        with pytest.raises(EvaluationError, match="Every candidate"):
            GeneticArchitectureSearch(
                ForgetsToReturn(), make_cfg(n_generations=2)
            ).run(tiny_data)
