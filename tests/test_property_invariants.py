"""Seeded random-configuration test: invariants that must hold for *every* valid run.

Hand-written tests cover the cases someone thought of; this draws 80 random but valid
configurations (population/elitism/tournament sizes, depth and width bounds, selection and
crossover strategies, mutation weights, error policies, optional seeds -- some of them real
model objects -- and injected build failures) and checks the properties no configuration may
violate. It found no engine bug when written, and it is what would have caught the original
code's silent corruption (a stored fitness no longer matching its architecture).
"""

import random

import numpy as np
import pytest

from dnnga import Architecture, EvaluationError, EvolutionConfig, GeneticArchitectureSearch, TrainingData
from tests.conftest import FakeBackend, FakeModel

DATA = TrainingData(np.zeros((6, 2)), [0, 1] * 3, np.zeros((4, 2)), [0, 1] * 2,
                    X_test=np.zeros((3, 2)), y_test=[0, 1, 0])


class RecordingBackend(FakeBackend):
    """FakeBackend + noisy scores + optional injected build failures + bookkeeping."""

    def __init__(self, noise_rng, fail_p, **kw):
        super().__init__(**kw)
        self.noise_rng, self.fail_p = noise_rng, fail_p
        self.raw_seen, self.build_failures = [], 0

    def build(self, architecture, data):
        if self.fail_p and self.noise_rng.random() < self.fail_p:
            self.build_failures += 1
            raise RuntimeError("injected build failure")
        return super().build(architecture, data)

    def evaluate(self, model, data):
        score = super().evaluate(model, data) + self.noise_rng.uniform(-3, 3)
        if len(data.X_val) == len(DATA.X_val):          # validation-split scores only (not the final test score)
            self.raw_seen.append(score)
        return score


def _random_case(case):
    r = random.Random(case)
    min_l = r.randint(1, 3)
    max_l = min_l + r.randint(0, 3)
    min_n = r.randint(1, 5)
    max_n = min_n + r.randint(0, 25)
    pop = r.randint(2, 25)
    weights = {"tweak_neurons": r.choice([0.0, 0.5, 1.0]), "add_layer": r.choice([0.0, 0.3]),
               "remove_layer": r.choice([0.0, 0.3])}
    if min_l == max_l:                                  # structural mutation is impossible at a fixed depth
        weights["add_layer"] = weights["remove_layer"] = 0.0
    if sum(weights.values()) == 0:
        weights["tweak_neurons"] = 1.0
    cfg = EvolutionConfig(
        min_layers=min_l, max_layers=max_l, min_neurons=min_n, max_neurons=max_n, population_size=pop,
        n_generations=r.randint(1, 8), epochs_per_candidate=r.randint(1, 5),
        elitism_count=r.randint(0, min(3, pop - 1)), tournament_size=r.randint(2, min(5, pop)),
        selection_strategy=r.choice(["tournament", "roulette"]),
        crossover_strategy=r.choice(["single_point", "uniform", "none"]), p_crossover=r.random(),
        mutation_rate=r.random(), neuron_mutation_step=r.randint(1, 6),
        mutation_boundary_mode=r.choice(["clamp", "wrap"]), mutation_operator_weights=weights,
        cache_evaluations=r.random() < 0.7, on_evaluation_error=r.choice(["penalize", "penalize", "skip"]),
        direction=r.choice(["maximize", "minimize"]), random_state=case,
        target_fitness=r.choice([None, None, 5.0, -5.0]), stagnation_generations=r.choice([None, 2, 4]),
        final_training_epochs=r.choice([None, None, 9]),
    )
    seed_arch = None
    if r.random() < 0.5:
        seed_arch = Architecture(tuple(r.randint(min_n, max_n) for _ in range(r.randint(min_l, max_l))))
    use_model_seed = seed_arch is not None and r.random() < 0.5
    backend = RecordingBackend(random.Random(case + 1), r.choice([0.0, 0.0, 0.15]), target_sum=r.randint(5, 60))
    return cfg, seed_arch, use_model_seed, backend


@pytest.mark.parametrize("case", range(80))
def test_invariants_hold_for_random_valid_configurations(case):
    cfg, seed_arch, use_model_seed, backend = _random_case(case)
    sign = 1 if cfg.direction == "maximize" else -1
    seed, seed_fitness, users_model = None, None, None
    if seed_arch is not None and use_model_seed:
        users_model = FakeModel(seed_arch)
        seed_fitness = float(-abs(sum(seed_arch.layer_sizes) - backend.target_sum))
        backend.raw_seen.append(seed_fitness)
        seed = users_model
    elif seed_arch is not None:
        seed = seed_arch

    try:
        res = GeneticArchitectureSearch(backend, cfg).run(DATA, seed=seed, seed_fitness=seed_fitness)
    except EvaluationError:
        assert backend.fail_p > 0, "EvaluationError is only legitimate when failures were injected"
        return

    hist = res.history
    for ind in res.final_population:                                   # structure stays inside the bounds
        a = ind.architecture
        assert cfg.min_layers <= a.n_layers <= cfg.max_layers and all(
            cfg.min_neurons <= w <= cfg.max_neurons for w in a.layer_sizes), a
    if cfg.on_evaluation_error != "skip":
        assert len(res.final_population) == cfg.population_size
    so_far = [rec.best_so_far for rec in hist.records]                 # never regresses, in either direction
    assert all((b - a) * sign >= -1e-12 for a, b in zip(so_far, so_far[1:]))
    assert 1 <= len(hist.records) <= cfg.n_generations
    if not cfg.cache_evaluations:
        assert hist.total_cache_hits == 0
    for rec in hist.records[1:]:                                       # elites are never re-trained
        assert rec.n_evaluated <= cfg.population_size - cfg.elitism_count
    if cfg.final_training_epochs is None:                              # compute accounting is exact
        assert hist.total_evaluations == backend.train_calls + backend.build_failures
        best_seen = max(backend.raw_seen) if sign > 0 else min(backend.raw_seen)
        assert res.best_fitness == pytest.approx(best_seen), "the winner must be the best score ever seen"
        if hist.stop_reason == "target_fitness_reached":
            assert (res.best_fitness - cfg.target_fitness) * sign >= -1e-9
    cleaned = [id(m) for m in backend.cleaned]
    assert len(cleaned) == len(set(cleaned)), "a model was passed to cleanup() twice"
    assert not any(m is res.best_model for m in backend.cleaned), "cleanup() was called on the returned model"
    if users_model is not None:
        assert not any(m is users_model for m in backend.cleaned), "cleanup() was called on the caller's own model"
    assert res.best_model is not None and res.test_score is not None
