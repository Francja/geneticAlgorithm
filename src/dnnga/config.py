"""All tunable knobs for a run, in one place, validated up front.

The original scripts spread these over module-level globals (``MIN_LAYER``,
``MAX_LAYER``, ``MAX_NEURONS``, ``POPULATION``, ``GENERATIONS``,
``TARGET_SCORE``, ``MUTATION_CHANCE``, ``NO_EVOLVE_GENS`` and, later,
``STEPS``). Two runs in one process shared that state, nothing checked the
values before compute was spent, and it was not obvious which globals a
function depended on.

``EvolutionConfig`` collects the same knobs (plus those needed for depth
evolution, elitism, caching and error handling) into one object that
validates itself and is passed explicitly.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

from .architecture import Architecture, validate_bounds
from .exceptions import ConfigError

Direction = Literal["maximize", "minimize"]
SelectionStrategy = Literal["tournament", "roulette"]
CrossoverStrategy = Literal["single_point", "uniform", "none"]
ErrorPolicy = Literal["penalize", "raise", "skip"]
BoundaryMode = Literal["clamp", "wrap"]

_OPERATOR_KEYS = {"tweak_neurons", "add_layer", "remove_layer"}


@dataclass
class EvolutionConfig:
    # --- structural search space ---------------------------------------
    min_layers: int = 1
    max_layers: int = 6
    min_neurons: int = 1
    max_neurons: int = 64

    # --- population / budget -------------------------------------------
    population_size: int = 20
    n_generations: int = 20
    epochs_per_candidate: int = 20
    """Training epochs each candidate gets before it is scored (the analogue
    of ``STEPS``; note STEPS were estimator gradient *steps*, not epochs)."""
    final_training_epochs: Optional[int] = None
    """If set, the winning architecture is rebuilt from scratch and retrained
    for this many epochs before being returned -- the standard NAS pattern of
    searching with a cheap, short proxy budget and spending a full budget
    once, on the winner. ``None`` returns the model exactly as it finished
    the search."""

    # --- stopping criteria (all are checked, as in the original) -------
    target_fitness: Optional[float] = None
    """Stop as soon as the best score reaches this (in the metric's own units,
    respecting ``direction``). ``None`` = evolve for ``n_generations``. This
    one field *is* the "train until a desired accuracy" mode -- not a
    separate path."""
    stagnation_generations: Optional[int] = 10
    """Stop when the best score has not improved by more than
    ``stagnation_min_delta`` for this many generations (``None`` disables)."""
    stagnation_min_delta: float = 1e-4

    # --- fitness convention --------------------------------------------
    direction: Direction = "maximize"
    """"maximize" for accuracy/F1/R^2, "minimize" for loss/error. Internally
    the engine always maximizes; this controls whether the raw metric is
    negated."""

    # --- selection ------------------------------------------------------
    selection_strategy: SelectionStrategy = "tournament"
    tournament_size: int = 3
    elitism_count: int = 1
    """The top-k individuals pass into the next generation unchanged (and are
    not re-trained). The original had no such guarantee, so the best
    individual could be lost and results could regress from one generation
    to the next."""

    # --- crossover ------------------------------------------------------
    crossover_strategy: CrossoverStrategy = "single_point"
    p_crossover: float = 0.8

    # --- mutation -------------------------------------------------------
    mutation_rate: float = 0.2
    """Probability that an individual is mutated at all in a generation (the
    analogue of ``MUTATION_CHANCE``). When it fires, exactly one operator
    from ``mutation_operator_weights`` is applied."""
    neuron_mutation_step: int = 4
    """Max size of one neuron-count tweak (the original always used +/-1)."""
    mutation_boundary_mode: BoundaryMode = "clamp"
    """What a neuron tweak does at the [min_neurons, max_neurons] edge.
    "clamp" stops at the edge. "wrap" reproduces the original (above max ->
    1, below 1 -> max), which turns a *small* perturbation into a huge jump;
    kept only for run-for-run comparison with the original."""
    mutation_operator_weights: dict = field(
        default_factory=lambda: {
            "tweak_neurons": 0.7,
            "add_layer": 0.15,
            "remove_layer": 0.15,
        }
    )
    """Relative weights of the three mutation operators, renormalised over
    those currently feasible (``add_layer`` is skipped at ``max_layers``,
    etc.). ``add_layer``/``remove_layer`` are what the original lacked: its
    only mutation nudged one layer's width by 1, so depth could change only
    by crossing over two *initial* random individuals of different depth --
    structural diversity could shrink over a run but never grow."""

    # --- efficiency / robustness ----------------------------------------
    cache_evaluations: bool = True
    """Remember each architecture's score and reuse it if the same
    architecture reappears (duplicates become common as the population
    converges -- the thesis's own Table 3 shows 8 clones of one network).
    Saves a full training run per duplicate. Trade-off: with a *noisy*
    backend (unseeded weight init) an architecture gets one noisy draw, kept
    for the run."""
    on_evaluation_error: ErrorPolicy = "penalize"
    """If build/train/evaluate raises for a candidate: "penalize" = log a
    warning, give it the worst possible fitness, carry on (default);
    "raise" = stop with EvaluationError; "skip" = drop it from the
    population."""

    # --- reproducibility ------------------------------------------------
    random_state: Optional[int] = None

    def __post_init__(self) -> None:
        if isinstance(self.mutation_operator_weights, dict):
            # defensive copy: later edits to the caller's dict must not leak in
            self.mutation_operator_weights = dict(self.mutation_operator_weights)
        self.validate()

    def with_overrides(self, **overrides) -> "EvolutionConfig":
        """A validated copy with some fields replaced; unknown names raise ConfigError."""
        valid = {f.name for f in dataclasses.fields(self)}
        unknown = sorted(set(overrides) - valid)
        if unknown:
            raise ConfigError(
                f"Unknown EvolutionConfig field(s) {unknown}. "
                f"Valid fields: {sorted(valid)}"
            )
        return dataclasses.replace(self, **overrides)

    def validate(self) -> None:
        problems: list = []

        try:
            validate_bounds(
                min_layers=self.min_layers,
                max_layers=self.max_layers,
                min_neurons=self.min_neurons,
                max_neurons=self.max_neurons,
            )
        except ConfigError as exc:
            problems.append(str(exc))

        if self.population_size < 2:
            problems.append(
                f"population_size must be >= 2, got {self.population_size}"
            )
        if self.n_generations < 1:
            problems.append(
                f"n_generations must be >= 1, got {self.n_generations}"
            )
        if self.epochs_per_candidate < 1:
            problems.append(
                f"epochs_per_candidate must be >= 1, got {self.epochs_per_candidate}"
            )
        if self.final_training_epochs is not None and self.final_training_epochs < 1:
            problems.append(
                f"final_training_epochs must be >= 1 or None, got "
                f"{self.final_training_epochs}"
            )
        if self.elitism_count < 0:
            problems.append(
                f"elitism_count must be >= 0, got {self.elitism_count}"
            )
        if self.elitism_count >= self.population_size:
            problems.append(
                f"elitism_count ({self.elitism_count}) must be < population_size "
                f"({self.population_size}), or no offspring could ever be produced"
            )
        if self.tournament_size < 2:
            problems.append(
                f"tournament_size must be >= 2, got {self.tournament_size}"
            )
        if self.tournament_size > self.population_size:
            problems.append(
                f"tournament_size ({self.tournament_size}) cannot exceed "
                f"population_size ({self.population_size})"
            )

        for name, value in (
            ("p_crossover", self.p_crossover),
            ("mutation_rate", self.mutation_rate),
        ):
            if not (0.0 <= value <= 1.0):
                problems.append(f"{name} must be within [0, 1], got {value}")

        if self.neuron_mutation_step < 1:
            problems.append(
                f"neuron_mutation_step must be >= 1, got "
                f"{self.neuron_mutation_step}"
            )

        weights = self.mutation_operator_weights
        if not isinstance(weights, dict) or not weights or set(weights) - _OPERATOR_KEYS:
            problems.append(
                f"mutation_operator_weights must be a non-empty dict with keys from "
                f"{sorted(_OPERATOR_KEYS)}, got {weights!r}"
            )
        else:
            if any(w < 0 for w in weights.values()):
                problems.append(
                    "mutation_operator_weights cannot contain negative weights"
                )
            elif all(w == 0 for w in weights.values()):
                problems.append(
                    "mutation_operator_weights must have at least one positive weight"
                )
            if self.min_layers == self.max_layers and (
                weights.get("add_layer", 0) + weights.get("remove_layer", 0) > 0
            ):
                problems.append(
                    "min_layers == max_layers, so add_layer/remove_layer can never "
                    "apply; set them to 0 in mutation_operator_weights (or widen the "
                    "layer range) so a mutation operator does not silently never fire"
                )

        if self.direction not in ("maximize", "minimize"):
            problems.append(
                f"direction must be 'maximize' or 'minimize', got {self.direction!r}"
            )
        if self.target_fitness is not None and not math.isfinite(self.target_fitness):
            problems.append(
                f"target_fitness must be a finite number or None, got "
                f"{self.target_fitness}"
            )
        if self.stagnation_generations is not None and self.stagnation_generations < 1:
            problems.append(
                f"stagnation_generations must be >= 1 or None, got "
                f"{self.stagnation_generations}"
            )
        if self.stagnation_min_delta < 0:
            problems.append(
                f"stagnation_min_delta must be >= 0, got {self.stagnation_min_delta}"
            )
        if self.selection_strategy not in ("tournament", "roulette"):
            problems.append(
                f"unknown selection_strategy {self.selection_strategy!r}"
            )
        if self.crossover_strategy not in ("single_point", "uniform", "none"):
            problems.append(
                f"unknown crossover_strategy {self.crossover_strategy!r}"
            )
        if self.mutation_boundary_mode not in ("clamp", "wrap"):
            problems.append(
                f"unknown mutation_boundary_mode {self.mutation_boundary_mode!r}"
            )
        if self.on_evaluation_error not in ("penalize", "raise", "skip"):
            problems.append(
                f"unknown on_evaluation_error {self.on_evaluation_error!r}"
            )

        if problems:
            bullets = "\n - ".join(problems)
            raise ConfigError(
                f"EvolutionConfig has {len(problems)} problem(s):\n - {bullets}"
            )


def _check_delta(name: str, delta: Sequence[int]):
    try:
        lo, hi = delta
        lo, hi = int(lo), int(hi)
    except (TypeError, ValueError):
        raise ConfigError(
            f"{name} must be a (remove, add) pair like (-1, 2), got {delta!r}"
        ) from None
    if lo > 0 or hi < 0:
        raise ConfigError(
            f"{name} must satisfy remove <= 0 <= add, got ({lo}, {hi})"
        )
    return lo, hi


def bounds_relative_to(
    seed: Architecture,
    *,
    layers_delta: Optional[Sequence[int]] = None,
    neurons_delta: Optional[Sequence[int]] = None,
) -> dict:
    """Turn "may add/remove up to N layers / M neurons relative to my network"
    into the absolute bounds the engine uses.

    ``layers_delta=(-1, 2)`` on a 3-layer seed -> ``min_layers=2,
    max_layers=5``. ``neurons_delta=(-8, 16)`` gives one *global* width range
    for every layer: ``[min(seed widths) - 8, max(seed widths) + 16]``
    (floored at 1), e.g. a seed of ``[64, 32, 16]`` -> ``[8, 80]``.

    This is one interpretation of "how many neurons it may add or subtract";
    if you meant a per-layer range, pass absolute ``min_neurons`` /
    ``max_neurons`` instead. The seed always lies inside the result.
    """
    out: dict = {}
    if layers_delta is not None:
        lo, hi = _check_delta("layers_delta", layers_delta)
        out["min_layers"] = max(1, seed.n_layers + lo)
        out["max_layers"] = seed.n_layers + hi
    if neurons_delta is not None:
        lo, hi = _check_delta("neurons_delta", neurons_delta)
        out["min_neurons"] = max(1, min(seed.layer_sizes) + lo)
        out["max_neurons"] = max(seed.layer_sizes) + hi
    return out
