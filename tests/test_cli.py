"""Tests for the ``dnnga`` console entry point (``dnnga/cli.py``).

The CLI is a thin wrapper, so most of these check wiring and the CLI contract
(exit codes, what is logged, which split is scored) rather than the engine,
which has its own tests. ``synthetic`` mode needs no framework; ``sklearn``
mode is skipped when scikit-learn is not installed.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import dnnga.cli as cli
from dnnga import Architecture, __version__
from tests.conftest import FakeBackend


# --------------------------------------------------------------- unit helpers
class TestToyFitness:
    def test_matches_the_original_formula(self):
        # geneticAlgorithm.py: first * second / second_last + last * 8
        assert cli.toy_fitness(Architecture((6, 6, 5, 6))) == pytest.approx(6 * 6 / 5 + 6 * 8)

    def test_single_layer_is_defined_instead_of_raising_indexerror(self):
        # The original would raise IndexError for a 1-layer genome; here the
        # same formula degenerates to 9 * width.
        assert cli.toy_fitness(Architecture((4,))) == pytest.approx(36.0)


class TestParseSeedLayers:
    def test_parses_a_comma_separated_list(self):
        assert cli.parse_seed_layers("16,8") == [16, 8]

    def test_none_means_no_seed(self):
        assert cli.parse_seed_layers(None) is None

    def test_malformed_input_raises_a_clear_error(self):
        with pytest.raises(ValueError, match="comma-separated"):
            cli.parse_seed_layers("16,x")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="empty"):
            cli.parse_seed_layers("")


# ------------------------------------------------------------ end-to-end runs
def test_synthetic_mode_runs_and_logs_the_winner(tmp_path, capsys):
    log_file = tmp_path / "run.log"
    exit_code = cli.main(
        [
            "--mode",
            "synthetic",
            "--population",
            "8",
            "--generations",
            "3",
            "--seed",
            "1",
            "--log-file",
            str(log_file),
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "Generation 3/3" in stdout  # per-generation logging reaches the console

    contents = log_file.read_text(encoding="utf-8")
    assert "Stopped after 3 generation(s): max_generations" in contents
    assert "Best architecture: [" in contents
    assert "Validation score: " in contents
    assert "Held-out test score: n/a (no test split in this mode)" in contents
    assert "Trainings: " in contents


def test_population_need_not_be_divisible_by_4(capsys):
    """The original script required POPULATION % 4 == 0; the engine does not."""
    exit_code = cli.main(
        [
            "--mode",
            "synthetic",
            "--population",
            "3",
            "--tournament-size",
            "2",
            "--generations",
            "2",
            "--seed",
            "0",
        ]
    )
    assert exit_code == 0
    assert "Stopped after 2 generation(s)" in capsys.readouterr().out


def test_target_score_stops_the_run_early(capsys):
    # The toy fitness is unbounded above, so any finite target is reachable.
    exit_code = cli.main(
        [
            "--mode",
            "synthetic",
            "--population",
            "6",
            "--generations",
            "20",
            "--target-score",
            "0",
            "--seed",
            "0",
        ]
    )
    assert exit_code == 0
    assert "target_fitness_reached" in capsys.readouterr().out


def test_seed_layers_are_passed_to_the_engine(monkeypatch, tiny_data):
    captured = {}
    real_run = cli.GeneticArchitectureSearch.run

    def spy(self, data, seed=None, seed_fitness=None):
        captured["seed"] = seed
        return real_run(self, data, seed=seed, seed_fitness=seed_fitness)

    monkeypatch.setattr(cli.GeneticArchitectureSearch, "run", spy)
    monkeypatch.setattr(cli, "make_backend", lambda mode, random_state: FakeBackend())
    monkeypatch.setattr(cli, "load_data", lambda mode, random_state: tiny_data)

    assert cli.main(["--mode", "synthetic", "--seed-layers", "4,4", "--population", "6"]) == 0
    assert captured["seed"] == [4, 4]


def test_candidates_are_scored_on_validation_and_only_the_winner_on_test(
    monkeypatch, tiny_data, capsys
):
    """The dnn-ga CLI scored candidates on the test split; this one must not."""
    backend = FakeBackend()
    monkeypatch.setattr(cli, "make_backend", lambda mode, random_state: backend)
    monkeypatch.setattr(cli, "load_data", lambda mode, random_state: tiny_data)

    assert cli.main(["--mode", "synthetic", "--population", "6", "--generations", "2"]) == 0

    val_rows, test_rows = len(tiny_data.X_val), len(tiny_data.X_test)
    # Every search evaluation saw the validation rows; the final test score is
    # the single evaluation that saw the test rows.
    assert backend.eval_row_counts[:-1] and all(n == val_rows for n in backend.eval_row_counts[:-1])
    assert backend.eval_row_counts[-1] == test_rows
    assert "Held-out test score: " in capsys.readouterr().out


def test_verbose_enables_per_candidate_debug_logging(capsys):
    assert cli.main(["--mode", "synthetic", "--population", "4", "--generations", "1", "--verbose"]) == 0
    assert "Evaluated" in capsys.readouterr().out


# --------------------------------------------------------------- error paths
def test_invalid_config_exits_2_with_a_message_on_stderr(capsys):
    assert cli.main(["--mode", "synthetic", "--population", "1"]) == 2
    captured = capsys.readouterr()
    assert "dnnga: error:" in captured.err
    assert "population_size must be >= 2" in captured.err
    assert "Generation" not in captured.out  # nothing was computed


def test_bad_seed_layers_exits_2(capsys):
    assert cli.main(["--mode", "synthetic", "--seed-layers", "16,x"]) == 2
    assert "comma-separated" in capsys.readouterr().err


def test_seed_outside_the_bounds_exits_2(capsys):
    assert cli.main(["--mode", "synthetic", "--seed-layers", "100", "--max-neurons", "8"]) == 2
    assert "seed does not fit the search bounds" in capsys.readouterr().err


# ------------------------------------------------------- optional frameworks
def test_sklearn_mode_scores_on_iris():
    pytest.importorskip("sklearn")
    assert (
        cli.main(
            [
                "--mode",
                "sklearn",
                "--population",
                "4",
                "--generations",
                "1",
                "--epochs-per-candidate",
                "2",
                "--max-layers",
                "2",
                "--max-neurons",
                "8",
                "--seed",
                "0",
            ]
        )
        == 0
    )


def test_keras_mode_builds_the_keras_backend():
    pytest.importorskip("tensorflow")
    from dnnga.backends import KerasMLPBackend

    assert isinstance(cli.make_backend("keras", 0), KerasMLPBackend)


def test_synthetic_mode_needs_no_framework():
    code = (
        "import sys; sys.modules['sklearn'] = None; sys.modules['tensorflow'] = None\n"
        "import dnnga.cli as c\n"
        "raise SystemExit(c.main(['--mode', 'synthetic', '--population', '4',"
        " '--generations', '1']))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_missing_sklearn_gives_a_clear_error():
    code = (
        "import sys; sys.modules['sklearn'] = None\n"
        "import dnnga.cli as c\n"
        "raise SystemExit(c.main(['--mode', 'sklearn', '--population', '4',"
        " '--generations', '1']))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 2
    assert "scikit-learn" in proc.stderr


def test_python_dash_m_dnnga_runs_the_cli():
    proc = subprocess.run(
        [sys.executable, "-m", "dnnga", "--mode", "synthetic", "--population", "4",
         "--generations", "1", "--seed", "0"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Best architecture: [" in proc.stdout


def test_version_flag_prints_the_version(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0
    assert f"dnnga {__version__}" in capsys.readouterr().out
