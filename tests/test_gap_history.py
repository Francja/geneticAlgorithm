"""Mutation-gap tests for issue #3, priority 3 (``RunHistory`` counters).

The baseline suite only ever saw 0 or 1 failures / cache hits per generation,
so ``n_failed += 2``, ``n_failed = 1``, ``n_cached += 2``, ``n_cached = 1`` and
``total_cache_hits = ...`` (instead of ``+=``) all survived. These tests assert
exact counts for a generation with two failures and for a second pass over the
same population with four cache hits, plus the run-level sums.
"""

from __future__ import annotations

from dnnga import Architecture, GeneticArchitectureSearch, Individual
from dnnga.history import GenerationRecord, RunHistory
from tests.conftest import FakeBackend, make_cfg


def _record(n_evaluated=0, n_cache_hits=0, n_failed=0):
    return GenerationRecord(
        generation=0,
        best_fitness=0.0,
        mean_fitness=0.0,
        worst_fitness=0.0,
        best_architecture=[1],
        best_so_far=0.0,
        population_size=6,
        n_evaluated=n_evaluated,
        n_cache_hits=n_cache_hits,
        n_failed=n_failed,
    )


class TestRunHistoryTotals:
    def test_cache_hits_and_trainings_are_summed_across_generations(self):
        history = RunHistory()
        history.add(_record(n_evaluated=3, n_cache_hits=2))
        history.add(_record(n_evaluated=4, n_cache_hits=3))
        assert history.total_cache_hits == 5
        assert history.total_evaluations == 7

    def test_a_record_with_two_failures_and_two_cache_hits_keeps_its_counts(self):
        history = RunHistory()
        history.add(_record(n_evaluated=3, n_cache_hits=2, n_failed=2))
        assert history.total_cache_hits == 2
        assert history.total_evaluations == 3
        assert history.records[0].n_failed == 2


class TestGenerationCounters:
    ARCHITECTURES = [Architecture((width,)) for width in (2, 3, 4, 5, 6, 7)]
    FAILING = {Architecture((2,)), Architecture((3,))}

    def test_two_failures_and_four_cache_hits_are_counted_exactly(self, tiny_data):
        backend = FakeBackend(raise_for=lambda architecture: architecture in self.FAILING)
        config = make_cfg(
            population_size=6, epochs_per_candidate=2, cache_evaluations=True
        )
        search = GeneticArchitectureSearch(backend, config)
        cache = {}
        population = [
            Individual(architecture, metadata={"origin": "random"})
            for architecture in self.ARCHITECTURES
        ]

        survivors, n_trained, n_cached, n_failed = search._evaluate_population(
            population, tiny_data, 1.0, cache
        )
        # n_trained counts attempts (the two failed builds are still compute),
        # n_failed counts only the failures, n_cached is zero on a cold cache.
        assert (len(survivors), n_trained, n_cached, n_failed) == (6, 6, 0, 2)

        # A second, freshly-built generation with the four architectures that
        # were scored successfully: all four are served from the cache.
        cached_population = [
            Individual(architecture, metadata={"origin": "random"})
            for architecture in self.ARCHITECTURES
            if architecture not in self.FAILING
        ]
        survivors, n_trained, n_cached, n_failed = search._evaluate_population(
            cached_population, tiny_data, 1.0, cache
        )
        assert (len(survivors), n_trained, n_cached, n_failed) == (4, 0, 4, 0)
        assert all(ind.metadata.get("cache_hit") for ind in survivors)

    def test_history_totals_equal_the_sum_of_the_generation_records(self, tiny_data):
        # A small architecture space fills the cache quickly, so at least one
        # generation must contain several cache hits.
        config = make_cfg(
            min_layers=1,
            max_layers=2,
            min_neurons=1,
            max_neurons=3,
            population_size=10,
            n_generations=10,
            tournament_size=2,
        )
        result = GeneticArchitectureSearch(FakeBackend(), config).run(tiny_data)
        records = result.history.records

        assert sum(r.n_cache_hits for r in records) == result.history.total_cache_hits
        assert sum(r.n_evaluated for r in records) == result.history.total_evaluations
        assert result.history.total_cache_hits >= 2
        assert max(r.n_cache_hits for r in records) >= 2
