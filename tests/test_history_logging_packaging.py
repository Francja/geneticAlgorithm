import io
import json
import logging
import subprocess
import sys

import pytest

from dnnga import (
    GenerationRecord,
    GeneticArchitectureSearch,
    RunHistory,
    configure_logging,
)
from tests.conftest import FakeBackend, make_cfg


def _rec(g, best, so_far, trained=3, cached=0):
    return GenerationRecord(
        generation=g,
        best_fitness=best,
        mean_fitness=best / 2,
        worst_fitness=0.0,
        best_architecture=[1, 2],
        best_so_far=so_far,
        population_size=5,
        n_evaluated=trained,
        n_cache_hits=cached,
    )


class TestRunHistory:
    def test_totals_accumulate(self):
        h = RunHistory()
        h.add(_rec(0, 0.5, 0.5, trained=5, cached=0))
        h.add(_rec(1, 0.6, 0.6, trained=3, cached=2))
        assert (h.total_evaluations, h.total_cache_hits) == (8, 2)

    def test_best_so_far_is_taken_from_the_last_record_not_max_of_best(self):
        # For a *minimised* loss the "best" values go down; max() of them (the
        # old implementation) would have returned the WORST value.
        h = RunHistory()
        h.add(_rec(0, 0.9, 0.9))
        h.add(_rec(1, 0.4, 0.4))
        h.add(_rec(2, 0.1, 0.1))
        assert h.best_so_far() == 0.1

    def test_empty_history(self):
        assert RunHistory().best_so_far() is None
        assert "Total trainings: 0" in RunHistory().summary_text()

    def test_summary_mentions_cache_and_failures(self):
        h = RunHistory()
        r = _rec(0, 0.5, 0.5, cached=2)
        r.n_failed = 1
        h.add(r)
        assert "2 cached" in h.summary_text() and "1 failed" in h.summary_text()

    def test_to_json_replaces_nan_with_null(self):
        h = RunHistory()
        r = _rec(0, 0.5, 0.5)
        r.mean_fitness = float("nan")
        h.add(r)
        assert json.loads(h.to_json())["generations"][0]["mean_fitness"] is None


class TestConfigureLogging:
    def test_library_is_silent_by_default(self):
        proc = subprocess.run(
            [sys.executable, "-c", "import dnnga"], capture_output=True, text=True
        )
        assert proc.returncode == 0 and proc.stdout == "" and proc.stderr == ""

    def test_output_goes_to_the_given_stream(self, tiny_data):
        buf = io.StringIO()
        configure_logging(level=logging.INFO, stream=buf)
        GeneticArchitectureSearch(FakeBackend(), make_cfg(n_generations=2)).run(tiny_data)
        assert "Generation 1/2" in buf.getvalue()

    def test_level_filters_debug(self, tiny_data):
        buf = io.StringIO()
        configure_logging(level=logging.INFO, stream=buf)
        GeneticArchitectureSearch(FakeBackend(), make_cfg(n_generations=1)).run(tiny_data)
        assert "Evaluated" not in buf.getvalue()

    def test_calling_twice_does_not_duplicate_lines(self, tiny_data):
        buf = io.StringIO()
        configure_logging(stream=buf)
        configure_logging(stream=buf)
        GeneticArchitectureSearch(FakeBackend(), make_cfg(n_generations=1)).run(tiny_data)
        assert buf.getvalue().count("Generation 1/1") == 1

    def test_file_handler(self, tiny_data, tmp_path):
        path = tmp_path / "run.log"
        configure_logging(filepath=str(path), stream=io.StringIO())
        GeneticArchitectureSearch(FakeBackend(), make_cfg(n_generations=2)).run(tiny_data)
        assert "Generation 2/2" in path.read_text(encoding="utf-8")


class TestOptionalDependencies:
    def test_import_dnnga_does_not_require_scikit_learn(self):
        code = "import sys; sys.modules['sklearn'] = None; import dnnga; print('ok')"
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert proc.returncode == 0, proc.stderr
        assert "ok" in proc.stdout

    def test_missing_scikit_learn_gives_a_clear_error_only_when_the_backend_is_used(self):
        code = (
            "import sys; sys.modules['sklearn'] = None\n"
            "from dnnga import BackendNotAvailableError\n"
            "try:\n"
            "    from dnnga.backends import SklearnMLPBackend\n"
            "except BackendNotAvailableError as e:\n"
            "    print('clear-error', 'scikit-learn' in str(e))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert "clear-error True" in proc.stdout, proc.stderr

    def test_unknown_backend_attribute_raises_attribute_error(self):
        import dnnga.backends as b

        with pytest.raises(AttributeError):
            b.DoesNotExist
