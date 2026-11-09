"""Command-line entry point: run one architecture search without writing code.

    dnnga --mode sklearn --population 12 --generations 8 --target-score 0.97

A thin wrapper over the library -- it builds an :class:`EvolutionConfig`, picks
a backend, loads a dataset and calls :class:`GeneticArchitectureSearch`. It is
the ``dnnga`` equivalent of the earlier ``dnn-ga`` CLI; anything it can do,
``evolve_architecture()`` can do with more control (a model object as the seed,
relative bounds, your own data).

Two deliberate differences from the ``dnn-ga`` CLI it mirrors:

* Candidates are scored on a **validation** split. ``dnn-ga --mode dnn``
  trained every candidate and scored it on the *test* split, so the GA selected
  on the very data its result was reported on. Here the test split is touched
  exactly once, at the end, on the winner; that number is printed as the
  held-out score.
* ``--population`` / ``--generations`` accept any value. The original script
  required the population to be divisible by 4 (an artefact of its
  elitism/crossover bookkeeping); the engine has no such rule.

Modes:

``synthetic``
    The toy fitness from the original ``geneticAlgorithm.py``. No dataset, no
    framework and no training: useful as a smoke test and for watching the GA
    loop itself.
``sklearn``
    The iris dataset, scored by an sklearn MLP backend (needs the ``sklearn``
    extra). Candidates are scored on validation, the winner on test.
``keras``
    The same data and protocol, scored by a Keras MLP backend (needs the
    ``keras`` extra).
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional, Sequence

import numpy as np

from . import __version__
from .architecture import Architecture
from .backends.base import ModelBackend
from .config import EvolutionConfig
from .data import TrainingData
from .engine import GeneticArchitectureSearch
from .exceptions import DnnGaError
from .logging_utils import configure_logging

MODES = ("synthetic", "sklearn", "keras")


def toy_fitness(architecture: Architecture) -> float:
    """The original script's toy cost function, for ``--mode synthetic``.

    ``first * second / second_last + last * 8`` over the layer widths, as in
    ``geneticAlgorithm.py``. A single-layer architecture has no distinct
    "second"/"second-last" width, so it is scored by the same formula with all
    four positions set to that one width (``9 * width``); the original would
    have raised ``IndexError`` there.

    It ignores the data entirely, so this is a pure function of the genotype --
    it exercises the GA, not any learning.
    """
    sizes = architecture.layer_sizes
    if len(sizes) >= 2:
        first, second, second_last, last = sizes[0], sizes[1], sizes[-2], sizes[-1]
    else:
        first = second = second_last = last = sizes[0]
    return first * second / second_last + last * 8.0


class ToyBackend(ModelBackend):
    """Scores architectures with :func:`toy_fitness`; no framework, no training."""

    def build(self, architecture: Architecture, data: TrainingData) -> Architecture:
        return architecture

    def train(self, model: Architecture, data: TrainingData, epochs: int) -> Architecture:
        return model

    def evaluate(self, model: Architecture, data: TrainingData) -> float:
        return toy_fitness(model)


def _synthetic_data() -> TrainingData:
    """A placeholder split: the toy fitness never looks at the data, but the
    engine requires a valid :class:`TrainingData`. No test split -- there is no
    held-out number to report for a formula that does not learn."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    y = np.array([0, 1] * 20)
    return TrainingData(X[:24], y[:24], X[24:], y[24:])


def _iris_data(random_state: int) -> TrainingData:
    """Iris with a train/val/test split; candidates are scored on val only.

    The scaler is fitted on the training rows alone (no leakage), and the same
    ``random_state`` drives the split, so ``--seed`` makes a run reproducible.
    """
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = load_iris(return_X_y=True)
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=0.25, stratify=y_rest, random_state=random_state
    )
    scaler = StandardScaler().fit(X_train)
    return TrainingData(
        scaler.transform(X_train),
        y_train,
        scaler.transform(X_val),
        y_val,
        X_test=scaler.transform(X_test),
        y_test=y_test,
    )


def make_backend(mode: str, random_state: Optional[int]) -> ModelBackend:
    """The backend for ``mode``; framework imports stay lazy so that
    ``--mode synthetic`` needs nothing beyond NumPy."""
    if mode == "sklearn":
        from .backends import SklearnMLPBackend

        return SklearnMLPBackend(random_state=random_state)
    if mode == "keras":
        from .backends import KerasMLPBackend

        return KerasMLPBackend(random_state=random_state)
    return ToyBackend()


def load_data(mode: str, random_state: Optional[int]) -> TrainingData:
    if mode == "synthetic":
        return _synthetic_data()
    return _iris_data(0 if random_state is None else random_state)


def parse_seed_layers(text: Optional[str]) -> Optional[List[int]]:
    """``"16,8"`` -> ``[16, 8]``; ``None`` -> ``None`` (no seed)."""
    if text is None:
        return None
    if not text.strip():
        raise ValueError("--seed-layers was empty; omit the flag to start unseeded.")
    try:
        widths = [int(part) for part in text.split(",")]
    except ValueError:
        raise ValueError(
            f"--seed-layers must be a comma-separated list of integers, got {text!r}."
        ) from None
    return widths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dnnga",
        description="Evolve a neural network's hidden layers with a genetic algorithm.",
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="synthetic",
        help=(
            "synthetic: the original script's toy fitness -- no data, no training, "
            "no framework (default). sklearn: iris, scored by an sklearn MLP. "
            "keras: iris, scored by a Keras MLP."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "seed for the GA, the train/val/test split and the backend's weight "
            "initialisation, for reproducible runs"
        ),
    )
    parser.add_argument(
        "--log-file", default=None, metavar="PATH", help="also write the log to this file"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="DEBUG logging: one line per candidate"
    )
    parser.add_argument("--population", type=int, default=12, help="individuals per generation")
    parser.add_argument("--generations", type=int, default=8, help="generation cap")
    parser.add_argument(
        "--target-score",
        type=float,
        default=None,
        help="stop as soon as the best score reaches this (in the metric's own units)",
    )
    parser.add_argument("--min-layers", type=int, default=1)
    parser.add_argument("--max-layers", type=int, default=4)
    parser.add_argument("--min-neurons", type=int, default=2)
    parser.add_argument("--max-neurons", type=int, default=32)
    parser.add_argument(
        "--epochs-per-candidate",
        type=int,
        default=20,
        help="cheap proxy training budget used to rank architectures",
    )
    parser.add_argument(
        "--final-training-epochs",
        type=int,
        default=None,
        help="retrain the winner from scratch with this budget",
    )
    parser.add_argument(
        "--elitism-count",
        type=int,
        default=1,
        help="top-k individuals carried over unchanged (never re-trained)",
    )
    parser.add_argument(
        "--tournament-size",
        type=int,
        default=3,
        help="contestants per tournament selection (must be <= --population)",
    )
    parser.add_argument(
        "--direction",
        choices=["maximize", "minimize"],
        default="maximize",
        help="maximize accuracy-like metrics (default), minimize loss-like ones",
    )
    parser.add_argument(
        "--seed-layers",
        default=None,
        metavar="WIDTHS",
        help='hidden-layer widths to start the search from, e.g. "16,8"',
    )
    parser.add_argument("--version", action="version", version=f"dnnga {__version__}")
    return parser


def _config_from_args(args: argparse.Namespace) -> EvolutionConfig:
    # Constructing the config validates every value up front, before any compute.
    return EvolutionConfig(
        min_layers=args.min_layers,
        max_layers=args.max_layers,
        min_neurons=args.min_neurons,
        max_neurons=args.max_neurons,
        population_size=args.population,
        n_generations=args.generations,
        epochs_per_candidate=args.epochs_per_candidate,
        final_training_epochs=args.final_training_epochs,
        target_fitness=args.target_score,
        elitism_count=args.elitism_count,
        tournament_size=args.tournament_size,
        direction=args.direction,
        random_state=args.seed,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logger = configure_logging(
        level=logging.DEBUG if args.verbose else logging.INFO, filepath=args.log_file
    )

    try:
        seed = parse_seed_layers(args.seed_layers)
        config = _config_from_args(args)
        backend = make_backend(args.mode, args.seed)
        data = load_data(args.mode, args.seed)
        result = GeneticArchitectureSearch(backend, config).run(data, seed=seed)
    except (DnnGaError, ValueError) as exc:
        # Everything dnnga raises deliberately is a DnnGaError; a bad CLI value
        # surfaces as ValueError. Both mean "nothing was computed, fix the input".
        print(f"dnnga: error: {exc}", file=sys.stderr)
        return 2

    logger.info(
        "Stopped after %d generation(s): %s",
        len(result.history.records),
        result.history.stop_reason,
    )
    logger.info("Best architecture: %s", list(result.best_architecture.layer_sizes))
    logger.info(
        "Validation score: %.4f (optimistically biased: the best of many candidates)",
        result.best_fitness,
    )
    if result.test_score is None:
        logger.info("Held-out test score: n/a (no test split in this mode)")
    else:
        logger.info("Held-out test score: %.4f (the honest number)", result.test_score)
    logger.info(
        "Trainings: %d (+%d reused from cache)",
        result.history.total_evaluations,
        result.history.total_cache_hits,
    )
    return 0
