import logging
import random

import pytest

from dnnga import (
    Architecture,
    ConfigError,
    EvaluationError,
    GeneticArchitectureSearch,
    InvalidArchitectureError,
    TrainingData,
    evolve_architecture,
)
from tests.conftest import FakeBackend, FakeModel, make_cfg


def run(cfg=None, backend=None, data=None, **kw):
    backend = backend or FakeBackend()
    return GeneticArchitectureSearch(backend, cfg or make_cfg()).run(data, **kw), backend


# ----------------------------------------------------------------- basics
class TestBasics:
    def test_result_contains_best_architecture_and_trained_model(self, tiny_data):
        res, _ = run(data=tiny_data)
        assert isinstance(res.best_model, FakeModel)
        assert res.best_model.architecture == res.best_architecture
        assert len(res.history.records) == 8
        assert res.history.stop_reason == "max_generations"

    def test_actually_improves_and_gets_close_to_optimum(self, tiny_data):
        res, _ = run(make_cfg(population_size=20, n_generations=15), data=tiny_data)
        first, last = (
            res.history.records[0].best_so_far,
            res.history.records[-1].best_so_far,
        )
        assert last >= first and res.best_fitness >= -2  # optimum is 0

    def test_population_stays_valid_across_strategies(self, tiny_data):
        for xo in ("single_point", "uniform", "none"):
            for sel in ("tournament", "roulette"):
                cfg = make_cfg(
                    crossover_strategy=xo,
                    selection_strategy=sel,
                    mutation_rate=1.0,
                    population_size=10,
                    n_generations=6,
                    elitism_count=2,
                )
                res, _ = run(cfg, data=tiny_data)
                assert len(res.final_population) == 10
                for ind in res.final_population:
                    assert 1 <= ind.architecture.n_layers <= 4
                    assert all(1 <= w <= 20 for w in ind.architecture.layer_sizes)

    def test_config_is_revalidated_at_run_time(self, tiny_data):
        cfg = make_cfg()
        cfg.population_size = 1
        with pytest.raises(ConfigError):
            GeneticArchitectureSearch(FakeBackend(), cfg).run(tiny_data)

    def test_final_population_is_fully_evaluated(self, tiny_data):
        res, _ = run(make_cfg(n_generations=3), data=tiny_data)
        assert all(i.is_evaluated for i in res.final_population)
        fits = [i.fitness for i in res.final_population]
        assert fits == sorted(fits, reverse=True), "final population is returned best-first"


# ------------------------------------------------------- elitism / best-ever
class TestElitismAndBestEver:
    @staticmethod
    def _noisy():
        rng, seen = random.Random(123), []

        def fn(arch):
            v = rng.random()
            seen.append(v)
            return v

        return fn, seen

    def test_generation_best_never_regresses_with_elitism(self, tiny_data):
        fn, _ = self._noisy()
        res, _ = run(make_cfg(elitism_count=1, n_generations=12), FakeBackend(fitness_fn=fn), tiny_data)
        gen_best = [r.best_fitness for r in res.history.records]
        assert all(b >= a for a, b in zip(gen_best, gen_best[1:]))

    def test_without_elitism_best_so_far_cannot_regress(self, tiny_data):
        fn, seen = self._noisy()
        res, _ = run(
            make_cfg(elitism_count=0, n_generations=12, cache_evaluations=False),
            FakeBackend(fitness_fn=fn),
            tiny_data,
        )
        so_far = [r.best_so_far for r in res.history.records]
        assert all(b >= a for a, b in zip(so_far, so_far[1:]))
        # and the returned winner really is the best thing ever evaluated
        assert res.best_fitness == pytest.approx(max(seen))

    def test_elites_are_not_retrained(self, tiny_data):
        res, fb = run(
            make_cfg(elitism_count=3, n_generations=5, cache_evaluations=False),
            data=tiny_data,
        )
        for r in res.history.records[1:]:
            assert r.n_evaluated == 12 - 3


# ------------------------------------------------------------------ stopping
class TestStopping:
    def test_target_fitness_stops_early(self, tiny_data):
        res, _ = run(
            make_cfg(target_fitness=-1, n_generations=30, population_size=20),
            data=tiny_data,
        )
        assert res.history.stop_reason == "target_fitness_reached"
        assert res.best_fitness >= -1 and len(res.history.records) < 30

    def test_unreachable_target_runs_all_generations(self, tiny_data):
        res, _ = run(make_cfg(target_fitness=5), data=tiny_data)
        # max possible score is 0
        assert res.history.stop_reason == "max_generations"
        assert len(res.history.records) == 8

    def test_seed_already_meeting_target_stops_at_generation_zero(self, tiny_data):
        res, _ = run(make_cfg(target_fitness=0), data=tiny_data, seed=[15, 15])
        # sum 30 == target_sum -> score 0
        assert res.history.stop_reason == "target_fitness_reached"
        assert len(res.history.records) == 1

    def test_stagnation_stops_run(self, tiny_data):
        res, _ = run(
            make_cfg(stagnation_generations=3, n_generations=20),
            FakeBackend(fitness_fn=lambda a: 0.5),
            tiny_data,
        )
        assert res.history.stop_reason == "stagnation"
        assert len(res.history.records) == 4

    def test_stagnation_credits_many_tiny_steps_that_add_up(self, tiny_data):
        # Each generation improves by only 3e-5 (< min_delta 1e-4) but every
        # ~4 generations the cumulative gain passes 1e-4. Comparing each
        # generation to the previous best (the old approach) would call that
        # stagnation after 5 generations; comparing to the last *credited*
        # improvement correctly keeps going.
        calls = {"n": 0}

        def creeping(arch):
            calls["n"] += 1
            return calls["n"] * 1e-5

        cfg = make_cfg(
            population_size=4,
            elitism_count=1,
            cache_evaluations=False,
            tournament_size=2,
            stagnation_generations=5,
            stagnation_min_delta=1e-4,
            n_generations=12,
        )
        res, _ = run(cfg, FakeBackend(fitness_fn=creeping), tiny_data)
        assert res.history.stop_reason == "max_generations"
        assert len(res.history.records) == 12

    def test_stagnation_disabled_with_none(self, tiny_data):
        res, _ = run(
            make_cfg(stagnation_generations=None, n_generations=6),
            FakeBackend(fitness_fn=lambda a: 0.5),
            tiny_data,
        )
        assert len(res.history.records) == 6


# ---------------------------------------------------------------------- seed
class TestSeed:
    def test_seed_is_in_generation_zero_and_result_is_never_worse(self, tiny_data):
        res, _ = run(make_cfg(population_size=6, n_generations=3), data=tiny_data, seed=[15, 15])
        assert res.history.records[0].best_so_far == 0
        assert res.best_fitness == 0

    def test_seed_without_fitness_is_trained_like_everyone_else(self, tiny_data):
        res, _ = run(make_cfg(population_size=6), data=tiny_data, seed=[15, 15])
        assert res.history.records[0].n_evaluated == 6

    def test_seed_fitness_skips_training_and_model_is_materialised_if_seed_wins(self, tiny_data):
        res, fb = run(
            make_cfg(population_size=6, epochs_per_candidate=5),
            data=tiny_data,
            seed=[15, 15],
            seed_fitness=0.0,
        )
        assert res.history.records[0].n_evaluated == 5
        assert res.best_architecture == Architecture((15, 15))
        assert res.best_model is not None and res.best_model.trained_epochs == 5
        assert "materialize" in res.history.extras

    def test_seed_model_object_is_used_and_kept_when_it_wins(self, tiny_data):
        seed_model = FakeModel(Architecture((15, 15)))
        res, _ = run(
            make_cfg(population_size=6),
            data=tiny_data,
            seed=seed_model,
            seed_fitness=0.0,
        )
        assert res.best_model is seed_model

    def test_seed_model_object_without_fitness_only_its_structure_is_used(self, tiny_data):
        seed_model = FakeModel(Architecture((15, 15)))
        res, _ = run(make_cfg(population_size=6), data=tiny_data, seed=seed_model)
        assert res.best_model is not seed_model  # retrained from scratch
        assert res.history.records[0].n_evaluated == 6

    def test_seed_outside_bounds_raises_helpful_error(self, tiny_data):
        with pytest.raises(InvalidArchitectureError, match="seed"):
            run(make_cfg(max_neurons=20), data=tiny_data, seed=[1000])

    def test_seed_fitness_without_seed_raises(self, tiny_data):
        with pytest.raises(ConfigError):
            run(data=tiny_data, seed_fitness=0.5)

    def test_non_finite_seed_fitness_raises(self, tiny_data):
        with pytest.raises(ConfigError):
            run(data=tiny_data, seed=[5], seed_fitness=float("nan"))

    def test_seed_fitness_respects_direction(self, tiny_data):
        cfg = make_cfg(direction="minimize", population_size=6)
        res, _ = run(
            cfg,
            FakeBackend(fitness_fn=lambda a: abs(sum(a.layer_sizes) - 30)),
            tiny_data,
            seed=[15, 15],
            seed_fitness=0.0,
        )
        assert res.best_fitness == 0.0


# --------------------------------------------------------------- error policy
class TestErrorPolicies:
    @staticmethod
    def _fail_depth3():
        return FakeBackend(raise_for=lambda a: a.n_layers == 3)

    def test_penalize_continues_and_never_picks_a_failed_architecture(self, tiny_data, caplog):
        with caplog.at_level(logging.WARNING, logger="dnnga"):
            res, _ = run(make_cfg(on_evaluation_error="penalize"), self._fail_depth3(), tiny_data)
        assert sum(r.n_failed for r in res.history.records) > 0
        assert res.best_architecture.n_layers != 3
        assert any("Evaluation failed" in r.getMessage() for r in caplog.records)

    def test_failed_individuals_are_excluded_from_mean_and_worst(self, tiny_data):
        res, _ = run(make_cfg(on_evaluation_error="penalize"), self._fail_depth3(), tiny_data)
        for r in res.history.records:
            assert r.mean_fitness == r.mean_fitness and r.worst_fitness == r.worst_fitness

    def test_raise_policy_raises(self, tiny_data):
        with pytest.raises(EvaluationError, match="Evaluation failed"):
            run(make_cfg(on_evaluation_error="raise"), self._fail_depth3(), tiny_data)

    def test_skip_policy_drops_individuals(self, tiny_data):
        res, _ = run(make_cfg(on_evaluation_error="skip"), self._fail_depth3(), tiny_data)
        assert res.history.records[0].population_size < 12
        assert all(i.architecture.n_layers != 3 for i in res.final_population)

    def test_skip_when_too_few_survive_raises(self, tiny_data):
        with pytest.raises(EvaluationError, match="remain"):
            run(make_cfg(on_evaluation_error="skip"), FakeBackend(raise_for=lambda a: True), tiny_data)

    def test_everything_failing_under_penalize_raises_at_the_end(self, tiny_data):
        with pytest.raises(EvaluationError, match="Every candidate"):
            run(
                make_cfg(on_evaluation_error="penalize", n_generations=2),
                FakeBackend(raise_for=lambda a: True),
                tiny_data,
            )

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_scores_count_as_failures(self, tiny_data, bad):
        fb = FakeBackend(
            fitness_fn=lambda a: bad if a.n_layers == 2 else -abs(sum(a.layer_sizes) - 30)
        )
        res, _ = run(make_cfg(), fb, tiny_data)
        assert sum(r.n_failed for r in res.history.records) > 0
        assert res.best_architecture.n_layers != 2

    def test_penalize_with_roulette_selection_does_not_crash(self, tiny_data):
        res, _ = run(
            make_cfg(selection_strategy="roulette"), self._fail_depth3(), tiny_data
        )
        assert res.best_model is not None

    def test_keyboard_interrupt_is_not_swallowed(self, tiny_data):
        class Boom(FakeBackend):
            def evaluate(self, model, data):
                raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            run(make_cfg(), Boom(), tiny_data)

    def test_failed_training_does_not_leak_the_half_built_model(self, tiny_data):
        class FailsAfterTrain(FakeBackend):
            def evaluate(self, model, data):
                if model.architecture.n_layers == 3:
                    raise RuntimeError("late failure")
                return super().evaluate(model, data)

        fb = FailsAfterTrain()
        run(make_cfg(), fb, tiny_data)
        assert any(m.architecture.n_layers == 3 for m in fb.cleaned)


# ------------------------------------------ caching / cleanup / retraining
class TestCachingCleanupRetrain:
    SMALL_SPACE = dict(
        min_layers=1,
        max_layers=2,
        min_neurons=1,
        max_neurons=3,
        population_size=10,
        n_generations=10,
        tournament_size=2,
    )  # only 12 distinct architectures

    def test_cache_avoids_retraining_duplicates(self, tiny_data):
        on, fb_on = run(make_cfg(cache_evaluations=True, **self.SMALL_SPACE), data=tiny_data)
        off, fb_off = run(make_cfg(cache_evaluations=False, **self.SMALL_SPACE), data=tiny_data)
        assert fb_on.train_calls <= 12 < fb_off.train_calls
        assert on.history.total_cache_hits > 0 and off.history.total_cache_hits == 0

    def test_cached_score_equals_original_score(self, tiny_data):
        res, _ = run(make_cfg(cache_evaluations=True, **self.SMALL_SPACE), data=tiny_data)
        for ind in res.final_population:
            if ind.metadata.get("cache_hit"):
                assert ind.metadata["raw_score"] == -abs(
                    sum(ind.architecture.layer_sizes) - 30
                )

    def test_discarded_models_are_cleaned_up_but_the_winner_never_is(self, tiny_data):
        res, fb = run(make_cfg(n_generations=8), data=tiny_data)
        assert len(fb.cleaned) > 0
        assert not any(m is res.best_model for m in fb.cleaned)

    def test_final_training_epochs_retrains_winner_from_scratch(self, tiny_data):
        res, _ = run(
            make_cfg(final_training_epochs=77, epochs_per_candidate=5), data=tiny_data
        )
        assert res.best_model.trained_epochs == 77  # fresh model, only the final budget
        assert res.history.extras["final_retrain"]["epochs"] == 77

    def test_without_final_epochs_winner_keeps_search_budget(self, tiny_data):
        res, _ = run(make_cfg(epochs_per_candidate=5), data=tiny_data)
        assert res.best_model.trained_epochs == 5
        assert "final_retrain" not in res.history.extras

    def test_failed_final_retrain_falls_back_to_search_model(self, tiny_data):
        class FailOnFinal(FakeBackend):
            def train(self, model, data, epochs):
                if epochs == 77:
                    raise RuntimeError("OOM")
                return super().train(model, data, epochs)

        res, _ = run(make_cfg(final_training_epochs=77), FailOnFinal(), tiny_data)
        assert res.best_model.trained_epochs == 5

    def test_failed_final_retrain_raises_under_raise_policy(self, tiny_data):
        class FailOnFinal(FakeBackend):
            def train(self, model, data, epochs):
                if epochs == 77:
                    raise RuntimeError("OOM")
                return super().train(model, data, epochs)

        with pytest.raises(EvaluationError):
            run(
                make_cfg(final_training_epochs=77, on_evaluation_error="raise"),
                FailOnFinal(),
                tiny_data,
            )


# ------------------------------- direction / test split / reproducibility
class TestDirectionTestSplitReproducibility:
    def test_minimize_reports_raw_units_and_respects_direction(self, tiny_data):
        loss = lambda a: float(abs(sum(a.layer_sizes) - 30))  # noqa: E731
        res, _ = run(
            make_cfg(direction="minimize", target_fitness=1.0, n_generations=30, population_size=20),
            FakeBackend(fitness_fn=loss),
            tiny_data,
        )
        assert res.history.stop_reason == "target_fitness_reached"
        assert 0 <= res.best_fitness <= 1.0
        so_far = [r.best_so_far for r in res.history.records]
        assert all(b <= a for a, b in zip(so_far, so_far[1:]))
        assert all(r.worst_fitness >= r.best_fitness for r in res.history.records)

    def test_test_split_is_scored_once_at_the_end(self, tiny_data):
        res, fb = run(make_cfg(), data=tiny_data)
        assert res.test_score is not None and res.history.extras["test_score"] == res.test_score
        assert fb.eval_row_counts[-1] == len(tiny_data.X_test) == 20
        assert set(fb.eval_row_counts[:-1]) == {30}, "search must only see validation"

    def test_no_test_split_means_no_test_score(self, tiny_data):
        no_test = TrainingData(
            tiny_data.X_train, tiny_data.y_train, tiny_data.X_val, tiny_data.y_val
        )
        res, _ = run(make_cfg(), data=no_test)
        assert res.test_score is None and "test_score" not in res.history.extras

    def test_same_random_state_reproduces_the_run(self, tiny_data):
        a, _ = run(make_cfg(random_state=5), data=tiny_data)
        b, _ = run(make_cfg(random_state=5), data=tiny_data)
        trace = lambda r: [  # noqa: E731
            (x.best_architecture, x.best_so_far, x.mean_fitness) for x in r.history.records
        ]
        assert trace(a) == trace(b) and a.best_architecture == b.best_architecture

    def test_different_random_state_explores_differently(self, tiny_data):
        a, _ = run(make_cfg(random_state=1), data=tiny_data)
        b, _ = run(make_cfg(random_state=2), data=tiny_data)
        trace = lambda r: [x.mean_fitness for x in r.history.records]  # noqa: E731
        assert trace(a) != trace(b)

    def test_engine_does_not_touch_global_random_state(self, tiny_data):
        random.seed(999)
        expected = random.random()
        random.seed(999)
        run(make_cfg(), data=tiny_data)
        assert random.random() == expected

    @pytest.mark.parametrize(
        "pop,elite,tour", [(2, 1, 2), (3, 1, 2), (5, 2, 3), (7, 2, 3), (13, 0, 4), (21, 5, 3)]
    )
    def test_any_population_size_works_not_just_multiples_of_four(
        self, tiny_data, pop, elite, tour
    ):
        res, _ = run(
            make_cfg(population_size=pop, elitism_count=elite, tournament_size=tour, n_generations=4),
            data=tiny_data,
        )
        assert [r.population_size for r in res.history.records] == [pop] * 4


# ---------------------------------------------------- depth actually evolves
class TestDepthEvolves:
    """The original could never produce a depth absent from its *initial*
    random population (crossover only redistributes depths; its mutation only
    nudged a width by +/-1). Force every initial individual to depth 2 and
    compare."""

    @staticmethod
    def _all_depth_two(monkeypatch):
        import dnnga.engine as eng

        monkeypatch.setattr(eng, "random_architecture", lambda **kw: Architecture((5, 5)))

    def _cfg(self, **kw):
        return make_cfg(
            min_layers=1,
            max_layers=6,
            population_size=10,
            n_generations=15,
            mutation_rate=0.8,
            stagnation_generations=None,
            **kw,
        )

    def test_original_behaviour_depth_is_frozen_without_depth_mutations(self, monkeypatch, tiny_data):
        self._all_depth_two(monkeypatch)
        cfg = self._cfg(
            mutation_operator_weights={
                "tweak_neurons": 1.0,
                "add_layer": 0.0,
                "remove_layer": 0.0,
            }
        )
        res, _ = run(cfg, FakeBackend(fitness_fn=lambda a: float(a.n_layers)), tiny_data)
        assert {i.architecture.n_layers for i in res.final_population} == {2}

    def test_with_add_remove_layer_depth_grows_towards_what_the_fitness_rewards(self, monkeypatch, tiny_data):
        self._all_depth_two(monkeypatch)
        res, _ = run(
            self._cfg(), FakeBackend(fitness_fn=lambda a: float(a.n_layers)), tiny_data
        )
        assert res.best_architecture.n_layers >= 4

    def test_and_shrinks_when_the_fitness_rewards_shallow_networks(self, monkeypatch, tiny_data):
        self._all_depth_two(monkeypatch)
        res, _ = run(
            self._cfg(), FakeBackend(fitness_fn=lambda a: -float(a.n_layers)), tiny_data
        )
        assert res.best_architecture.n_layers == 1


# ------------------------------------------------- logging + history export
class TestLoggingAndHistory:
    def test_generation_summaries_are_logged(self, tiny_data, caplog):
        with caplog.at_level(logging.INFO, logger="dnnga"):
            run(make_cfg(n_generations=3), data=tiny_data)
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "Generation 1/3" in text and "Generation 3/3" in text and "Stopped:" in text

    def test_debug_level_logs_each_evaluation(self, tiny_data, caplog):
        with caplog.at_level(logging.DEBUG, logger="dnnga"):
            run(make_cfg(n_generations=1), data=tiny_data)
        assert any("Evaluated" in r.getMessage() for r in caplog.records)

    def test_history_json_and_csv_exports(self, tiny_data, tmp_path):
        import csv, json

        res, _ = run(make_cfg(n_generations=4), data=tiny_data)
        data = json.loads(res.history.to_json(str(tmp_path / "h.json")))
        assert data["stop_reason"] == "max_generations" and len(data["generations"]) == 4
        assert json.loads((tmp_path / "h.json").read_text())["total_evaluations"] == res.history.total_evaluations
        res.history.to_csv(str(tmp_path / "h.csv"))
        rows = list(csv.DictReader((tmp_path / "h.csv").open()))
        assert len(rows) == 4 and {"generation", "best_so_far", "n_cache_hits"} <= set(rows[0])

    def test_json_export_is_strict_json_even_with_nan(self, tiny_data):
        import json

        res, _ = run(make_cfg(n_generations=2), FakeBackend(raise_for=lambda a: a.n_layers == 3), tiny_data)
        res.history.records[0].worst_fitness = float("nan")
        json.loads(
            res.history.to_json(),
            parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)),
        )

    def test_summary_text(self, tiny_data):
        res, _ = run(make_cfg(n_generations=3), data=tiny_data)
        text = res.history.summary_text()
        assert "Generation 0" in text and "Stopped: max_generations" in text and "Total trainings" in text

    def test_total_evaluations_matches_backend_calls(self, tiny_data):
        res, fb = run(make_cfg(cache_evaluations=False), data=tiny_data)
        assert res.history.total_evaluations == fb.train_calls


# -------------------------------------------------- evolve_architecture()
class TestEvolveArchitectureFunction:
    def test_returns_a_model_by_default(self, tiny_data):
        m = evolve_architecture(
            [15, 15],
            tiny_data,
            FakeBackend(),
            population_size=6,
            n_generations=3,
            epochs_per_candidate=3,
            random_state=1,
            stagnation_generations=None,
        )
        assert isinstance(m, FakeModel)

    def test_return_result(self, tiny_data):
        r = evolve_architecture(
            [15, 15],
            tiny_data,
            FakeBackend(),
            return_result=True,
            population_size=6,
            n_generations=3,
            epochs_per_candidate=3,
            random_state=1,
        )
        assert r.best_model is not None and r.history is not None

    def test_relative_bounds_constrain_the_search(self, tiny_data):
        r = evolve_architecture(
            [10, 10, 10],
            tiny_data,
            FakeBackend(),
            layers_delta=(-1, 1),
            neurons_delta=(-5, 5),
            return_result=True,
            population_size=10,
            n_generations=6,
            mutation_rate=1.0,
            epochs_per_candidate=2,
            random_state=3,
            stagnation_generations=None,
        )
        for ind in r.final_population:
            assert 2 <= ind.architecture.n_layers <= 4
            assert all(5 <= w <= 15 for w in ind.architecture.layer_sizes)

    def test_seed_may_be_a_model_object(self, tiny_data):
        m = evolve_architecture(
            FakeModel(Architecture((15, 15))),
            tiny_data,
            FakeBackend(),
            population_size=6,
            n_generations=2,
            epochs_per_candidate=2,
            random_state=1,
        )
        assert isinstance(m, FakeModel)

    def test_target_fitness_keyword(self, tiny_data):
        r = evolve_architecture(
            [15, 15],
            tiny_data,
            FakeBackend(),
            target_fitness=0,
            return_result=True,
            population_size=6,
            epochs_per_candidate=2,
            random_state=1,
        )
        assert r.history.stop_reason == "target_fitness_reached"

    def test_explicit_config_plus_overrides(self, tiny_data):
        base = make_cfg(population_size=6, n_generations=2)
        r = evolve_architecture(
            [15, 15],
            tiny_data,
            FakeBackend(),
            config=base,
            n_generations=3,
            return_result=True,
        )
        assert len(r.history.records) == 3 and base.n_generations == 2

    def test_unknown_keyword_raises(self, tiny_data):
        with pytest.raises(ConfigError, match="Unknown"):
            evolve_architecture([5], tiny_data, FakeBackend(), populaton_size=3)

    def test_fixed_depth_relative_search_disables_structural_mutations(self, tiny_data):
        # layers_delta=(0, 0) means "evolve widths, keep the depth"; the user
        # should not also have to zero add_layer/remove_layer by hand.
        r = evolve_architecture(
            [5, 5],
            tiny_data,
            FakeBackend(),
            layers_delta=(0, 0),
            return_result=True,
            population_size=6,
            n_generations=3,
            epochs_per_candidate=1,
            mutation_rate=1.0,
            random_state=0,
        )
        assert all(i.architecture.n_layers == 2 for i in r.final_population)

    def test_relative_and_absolute_bounds_conflict(self, tiny_data):
        with pytest.raises(ConfigError, match="max_layers"):
            evolve_architecture(
                [5, 5], tiny_data, FakeBackend(), layers_delta=(0, 1), max_layers=9
            )

    def test_deltas_require_a_seed(self, tiny_data):
        with pytest.raises(ConfigError):
            evolve_architecture(None, tiny_data, FakeBackend(), layers_delta=(0, 1))
