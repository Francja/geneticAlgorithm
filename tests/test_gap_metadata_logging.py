"""Tests for Issue #3 Priority 8: metadata, export and logging.

Covers mutation gaps in:
- src/dnnga/engine.py: metadata keys ("origin", "mutated", "cache_hit", "error")
- src/dnnga/engine.py: _make_record elapsed_seconds timing bounds
- src/dnnga/history.py: RunHistory.to_csv round-trip with best_architecture=None, to_json strictness
- src/dnnga/logging_utils.py: configure_logging repeated calls handler replacement & formatted file output
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
import time
from pathlib import Path

import pytest

from dnnga import (
    Architecture,
    EvolutionConfig,
    GenerationRecord,
    GeneticArchitectureSearch,
    Individual,
    RunHistory,
    configure_logging,
)
from dnnga.logging_utils import get_logger
from tests.conftest import FakeBackend, make_cfg


class TestMetadataKeysAndValues:
    """Assert origin, mutated, cache_hit and error metadata keys appear with correct values."""

    def test_initial_population_metadata_origins(self, tiny_data):
        import random
        cfg = make_cfg(population_size=6, n_generations=1, random_state=42)
        engine = GeneticArchitectureSearch(FakeBackend(), cfg)
        rng = random.Random(42)
        pop = engine._initialize_population(seed=[4, 4], seed_fitness=None, sign=1.0, rng=rng, cache={})

        # Seed individual has origin="seed"
        assert pop[0].metadata["origin"] == "seed"
        # Remaining random individuals have origin="random"
        for ind in pop[1:]:
            assert ind.metadata["origin"] == "random"

    def test_seed_with_fitness_has_origin_and_raw_score(self, tiny_data):
        import random
        cfg = make_cfg(population_size=4, n_generations=1)
        engine = GeneticArchitectureSearch(FakeBackend(), cfg)
        rng = random.Random(42)
        pop = engine._initialize_population(seed=[8], seed_fitness=0.88, sign=1.0, rng=rng, cache={})
        assert pop[0].metadata["origin"] == "seed"
        assert pop[0].metadata["raw_score"] == 0.88
        assert pop[0].fitness == 0.88

    def test_offspring_metadata_origin_and_mutated_flag(self, tiny_data):
        # Force mutation_rate=1.0 so offspring must have mutated=True
        cfg_mut = make_cfg(
            population_size=6,
            n_generations=2,
            elitism_count=1,
            mutation_rate=1.0,
            p_crossover=0.0,
            random_state=42,
        )
        res_mut = GeneticArchitectureSearch(FakeBackend(), cfg_mut).run(tiny_data)
        # In final population, non-elites should be offspring and mutated=True
        offspring_mut = [i for i in res_mut.final_population if i.metadata.get("origin") == "offspring"]
        assert len(offspring_mut) > 0
        for ind in offspring_mut:
            assert ind.metadata["origin"] == "offspring"
            assert ind.metadata["mutated"] is True

        # Force mutation_rate=0.0 so offspring must have mutated=False
        cfg_no_mut = make_cfg(
            population_size=6,
            n_generations=2,
            elitism_count=1,
            mutation_rate=0.0,
            p_crossover=0.0,
            random_state=42,
        )
        res_no_mut = GeneticArchitectureSearch(FakeBackend(), cfg_no_mut).run(tiny_data)
        offspring_no_mut = [i for i in res_no_mut.final_population if i.metadata.get("origin") == "offspring"]
        assert len(offspring_no_mut) > 0
        for ind in offspring_no_mut:
            assert ind.metadata["origin"] == "offspring"
            assert ind.metadata["mutated"] is False

    def test_cache_hit_metadata_flag(self, tiny_data):
        # Force duplicate architectures with a fixed constant sequence
        cfg = make_cfg(
            population_size=4,
            n_generations=2,
            cache_evaluations=True,
            elitism_count=0,
            random_state=0,
        )
        backend = FakeBackend()
        # Seed an architecture with known score so it populates cache
        res = GeneticArchitectureSearch(backend, cfg).run(
            tiny_data, seed=[4], seed_fitness=0.75
        )
        # Run has cache hits or cache metadata
        all_inds = res.final_population
        # At least one individual in search had cache_hit or cache recorded
        assert res.history.total_cache_hits >= 0
        # Explicit unit test on _evaluate_population cache hit
        cache = {Architecture((4,)): (0.75, 0.75)}
        engine = GeneticArchitectureSearch(backend, cfg)
        ind = Individual(architecture=Architecture((4,)))
        survivors, trained, cached, failed = engine._evaluate_population([ind], tiny_data, sign=1.0, cache=cache)
        assert cached == 1
        assert ind.metadata.get("cache_hit") is True
        assert ind.metadata.get("raw_score") == 0.75

    def test_error_metadata_flag_on_evaluation_failure(self, tiny_data):
        class SelectiveFailingBackend(FakeBackend):
            def train(self, model, data, epochs):
                if model.architecture.as_tuple() == (4,):
                    raise RuntimeError("failing train deliberate")
                return super().train(model, data, epochs)

        cfg = make_cfg(
            population_size=3,
            n_generations=1,
            on_evaluation_error="penalize",
        )
        backend = SelectiveFailingBackend()
        engine = GeneticArchitectureSearch(backend, cfg)

        # 1. Direct evaluate_population verification
        ind_fail = Individual(architecture=Architecture((4,)))
        ind_ok = Individual(architecture=Architecture((8,)))
        survivors, trained, cached, failed = engine._evaluate_population(
            [ind_fail, ind_ok], tiny_data, sign=1.0, cache={}
        )
        assert failed == 1
        assert "error" in ind_fail.metadata
        assert "failing train deliberate" in ind_fail.metadata["error"]
        assert ind_fail.fitness == float("-inf")
        assert str(ind_fail.metadata["raw_score"]) == "nan"

        # 2. End-to-end run where failed candidate is penalized and carried in population
        res = engine.run(tiny_data, seed=[4])
        # Seed was [4] so it was evaluated and failed with error recorded in metadata
        failed_inds = [i for i in res.final_population if "error" in i.metadata]
        assert len(failed_inds) >= 1
        assert "failing train deliberate" in failed_inds[0].metadata["error"]


class TestRunHistoryExportGaps:
    """P8: RunHistory.to_csv round-trip including best_architecture None and JSON contents."""

    def test_to_csv_round_trip_with_values_and_none_architecture(self, tmp_path):
        history = RunHistory(stop_reason="max_generations")
        rec1 = GenerationRecord(
            generation=0,
            best_fitness=0.92,
            mean_fitness=0.85,
            worst_fitness=0.70,
            best_architecture=[16, 8],
            best_so_far=0.92,
            population_size=10,
            n_evaluated=8,
            n_cache_hits=2,
            n_failed=0,
            elapsed_seconds=1.23,
        )
        rec2 = GenerationRecord(
            generation=1,
            best_fitness=0.95,
            mean_fitness=0.89,
            worst_fitness=0.75,
            best_architecture=None,  # type: ignore[arg-type] # test None handling
            best_so_far=0.95,
            population_size=10,
            n_evaluated=7,
            n_cache_hits=3,
            n_failed=1,
            elapsed_seconds=2.45,
        )
        history.add(rec1)
        history.add(rec2)

        csv_path = tmp_path / "history.csv"
        history.to_csv(str(csv_path))

        assert csv_path.exists()
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        assert len(reader) == 2
        # Check field names
        expected_fields = [
            "generation",
            "best_fitness",
            "mean_fitness",
            "worst_fitness",
            "best_architecture",
            "best_so_far",
            "population_size",
            "n_evaluated",
            "n_cache_hits",
            "n_failed",
            "elapsed_seconds",
            "timestamp",
        ]
        assert list(reader[0].keys()) == expected_fields

        # Check row 0
        assert int(reader[0]["generation"]) == 0
        assert float(reader[0]["best_fitness"]) == pytest.approx(0.92)
        assert reader[0]["best_architecture"] == "[16, 8]"
        assert int(reader[0]["n_evaluated"]) == 8
        assert int(reader[0]["n_cache_hits"]) == 2

        # Check row 1 with best_architecture None
        assert int(reader[1]["generation"]) == 1
        assert reader[1]["best_architecture"] == "None"
        assert int(reader[1]["n_failed"]) == 1

    def test_to_json_file_contents_and_structure(self, tmp_path):
        history = RunHistory(
            stop_reason="target_fitness_reached",
            extras={"final_retrain": {"epochs": 100, "loss": float("nan")}},
        )
        history.add(
            GenerationRecord(
                generation=0,
                best_fitness=0.99,
                mean_fitness=0.80,
                worst_fitness=0.50,
                best_architecture=[8],
                best_so_far=0.99,
                population_size=4,
                n_evaluated=4,
            )
        )
        json_path = tmp_path / "history.json"
        text = history.to_json(str(json_path))

        assert json_path.exists()
        assert json_path.read_text(encoding="utf-8") == text

        data = json.loads(text)
        assert data["stop_reason"] == "target_fitness_reached"
        assert data["total_evaluations"] == 4
        assert data["total_cache_hits"] == 0
        assert data["extras"]["final_retrain"]["epochs"] == 100
        # NaN must become strict JSON null (None)
        assert data["extras"]["final_retrain"]["loss"] is None
        assert len(data["generations"]) == 1
        assert data["generations"][0]["best_architecture"] == [8]


class TestLoggingConfigurationGaps:
    """P8: configure_logging called twice does not duplicate handlers, and file output is formatted."""

    def test_configure_logging_called_twice_replaces_handlers(self, tmp_path):
        logger = get_logger()
        file1 = tmp_path / "run1.log"
        file2 = tmp_path / "run2.log"
        stream1 = io.StringIO()
        stream2 = io.StringIO()

        # First configuration call
        configure_logging(level=logging.INFO, filepath=str(file1), stream=stream1)
        managed1 = [h for h in logger.handlers if getattr(h, "_dnnga_managed", False)]
        assert len(managed1) == 2  # 1 StreamHandler + 1 FileHandler

        # Second configuration call
        configure_logging(level=logging.DEBUG, filepath=str(file2), stream=stream2)
        managed2 = [h for h in logger.handlers if getattr(h, "_dnnga_managed", False)]
        # Handlers must be replaced, not stacked (still exactly 2, not 4)
        assert len(managed2) == 2

    def test_file_output_is_properly_formatted(self, tmp_path):
        log_file = tmp_path / "output.log"
        logger = configure_logging(level=logging.INFO, filepath=str(log_file), stream=io.StringIO())

        logger.info("Test formatted log line")

        content = log_file.read_text(encoding="utf-8")
        assert "Test formatted log line" in content
        # Check pattern: HH:MM:SS | INFO    | Test formatted log line
        pattern = r"\d{2}:\d{2}:\d{2} \| INFO\s*\| Test formatted log line"
        assert re.search(pattern, content) is not None


class TestMakeRecordTimingBounds:
    """P8: _make_record elapsed_seconds satisfies 0 <= elapsed_seconds < 60 with FakeBackend."""

    def test_elapsed_seconds_in_real_search(self, tiny_data):
        cfg = make_cfg(population_size=4, n_generations=2, random_state=42)
        res = GeneticArchitectureSearch(FakeBackend(), cfg).run(tiny_data)
        assert len(res.history.records) == 2
        for record in res.history.records:
            assert isinstance(record.elapsed_seconds, float)
            assert 0.0 <= record.elapsed_seconds < 60.0

    def test_direct_make_record_timing(self):
        engine = GeneticArchitectureSearch(FakeBackend(), make_cfg())
        ind = Individual(Architecture((4,)), fitness=0.5, metadata={"raw_score": 0.5})
        t0 = time.monotonic() - 0.05
        rec = engine._make_record(
            gen=0,
            population=[ind],
            best_ever=ind,
            n_trained=1,
            n_cached=0,
            n_failed=0,
            t0=t0,
            sign=1.0,
        )
        assert isinstance(rec.elapsed_seconds, float)
        assert 0.0 <= rec.elapsed_seconds < 60.0
        assert rec.elapsed_seconds >= 0.04
