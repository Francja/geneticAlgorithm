"""The evolutionary loop.

``GeneticArchitectureSearch.run()`` replaces the
``if __name__ == "__main__":`` block of the original
``geneticAlgorithm.py`` / notebook -- but it is a method you call with any
backend and dataset, not a script you hand-edit, and it fixes the bugs found
in the original (each fix is noted where it happens; the full list is in
docs/CODE_REVIEW.md).
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .architecture import (
    Architecture,
    architecture_from_any,
    check_within_bounds,
    random_architecture,
)
from .backends.base import ModelBackend
from .config import EvolutionConfig, bounds_relative_to
from .data import TrainingData
from .exceptions import ConfigError, EvaluationError, InvalidArchitectureError
from .history import GenerationRecord, RunHistory
from .individual import Individual
from .logging_utils import get_logger
from .operators import maybe_crossover, maybe_mutate, select

logger = get_logger()

SeedType = Union[Architecture, Sequence[int], Any, None]
_ARCHITECTURE_LIKE = (Architecture, list, tuple, range, np.ndarray)


def resolve_seed(backend: ModelBackend, seed: Any) -> Tuple[Architecture, Optional[Any]]:
    """``(architecture, model_object_or_None)`` for a seed.

    A list/tuple/array/Architecture is taken as a structure; anything else is
    treated as a framework model object and handed to
    ``backend.extract_architecture``.
    """
    if isinstance(seed, _ARCHITECTURE_LIKE):
        return architecture_from_any(seed), None
    return backend.extract_architecture(seed), seed


@dataclass
class EvolutionResult:
    """Everything a run returns.

    The winning architecture *and its trained model* are both here. The
    original never returned either -- it only printed a number, so finding
    out which architecture produced the best score meant reading the printed
    log by eye.
    """

    best_individual: Individual
    history: RunHistory
    final_population: List[Individual]
    test_score: Optional[float] = None
    """Score of the winning model on the held-out test split, if one was
    given. Prefer this over ``best_fitness`` when reporting results:
    ``best_fitness`` was the maximum of many validation scores, so it is
    optimistically biased."""

    @property
    def best_architecture(self) -> Architecture:
        return self.best_individual.architecture

    @property
    def best_model(self) -> Any:
        return self.best_individual.model

    @property
    def best_fitness(self) -> float:
        """Winning *validation* score in the metric's own units (any direction)."""
        return self.best_individual.metadata.get(
            "raw_score", self.best_individual.fitness
        )

    def __repr__(self) -> str:
        test = (
            ""
            if self.test_score is None
            else f", test_score={self.test_score:.4f}"
        )
        return (
            f"EvolutionResult(best_architecture={self.best_architecture}, "
            f"best_fitness={self.best_fitness:.4f}{test}, "
            f"stop_reason={self.history.stop_reason!r}, "
            f"generations_run={len(self.history.records)})"
        )


class GeneticArchitectureSearch:
    """One evolutionary architecture search against a backend.

    Holds only the backend and config, so it is stateless between
    :meth:`run` calls and can be reused (e.g. for a parameter sweep).
    """

    def __init__(
        self,
        backend: ModelBackend,
        config: Optional[EvolutionConfig] = None,
    ):
        self.backend = backend
        self.config = config if config is not None else EvolutionConfig()

    # ------------------------------------------------------------------ run
    def run(
        self,
        data: TrainingData,
        seed: SeedType = None,
        seed_fitness: Optional[float] = None,
    ) -> EvolutionResult:
        """Evolve architectures against `data`; return the best one.

        seed:
            Optional starting point -- a model object (if the backend
            implements ``extract_architecture``), an :class:`Architecture`, or
            a plain list such as ``[64, 32]``. It becomes one member of
            generation 0, so the search explores *around* your network; with
            best-ever tracking the result can never score below the seed.
            Only the seed's *structure* is used: its weights are not reused
            (it is retrained like any candidate) unless you also pass
            ``seed_fitness``.
        seed_fitness:
            The seed's known score on this data. Skips re-training it and
            uses this number instead -- the "I already know my current
            accuracy" mode. If the seed was a model object it is kept as that
            individual's model.
        """
        cfg = self.config
        cfg.validate()  # the config is mutable; re-check before spending compute

        if seed_fitness is not None:
            if seed is None:
                raise ConfigError("seed_fitness was given without a seed.")
            if not math.isfinite(float(seed_fitness)):
                raise ConfigError(
                    f"seed_fitness must be a finite number, got {seed_fitness!r}"
                )

        rng = random.Random(cfg.random_state)
        history = RunHistory()
        sign = 1.0 if cfg.direction == "maximize" else -1.0
        cache: Dict[Architecture, Tuple[float, float]] = {}
        population = self._initialize_population(seed, seed_fitness, sign, rng, cache)

        best_ever: Optional[Individual] = None
        reference: Optional[float] = None  # last fitness credited as an improvement
        stagnation_counter = 0
        stop_reason = "max_generations"

        for gen in range(cfg.n_generations):
            t0 = time.monotonic()
            population, n_trained, n_cached, n_failed = self._evaluate_population(
                population, data, sign, cache
            )

            if len(population) < max(2, cfg.elitism_count):
                raise EvaluationError(
                    f"Only {len(population)} individual(s) remain after dropping "
                    "failed evaluations (on_evaluation_error='skip'); cannot "
                    "continue. Use 'penalize', or investigate why candidates fail "
                    "(see the logged warnings)."
                )

            population.sort(key=lambda ind: ind.fitness, reverse=True)
            if best_ever is None or population[0].fitness > best_ever.fitness:
                best_ever = population[0]

            record = self._make_record(
                gen, population, best_ever, n_trained, n_cached, n_failed, t0, sign
            )
            history.add(record)
            logger.info(
                "Generation %d/%d | gen_best=%.4f mean=%.4f best_so_far=%.4f | "
                "trained %d, cached %d, failed %d | %.1fs",
                gen + 1,
                cfg.n_generations,
                record.best_fitness,
                record.mean_fitness,
                record.best_so_far,
                n_trained,
                n_cached,
                n_failed,
                record.elapsed_seconds,
            )

            # Stopping criteria: target score, then stagnation; max_generations
            # ends the loop naturally.
            if cfg.target_fitness is not None and best_ever.fitness >= cfg.target_fitness * sign:
                stop_reason = "target_fitness_reached"
                break

            # Stagnation compares against the last *credited* improvement
            # (Keras EarlyStopping semantics) so many tiny steps that add up to
            # a real gain still count. The original compared floats for exact
            # equality, which essentially never holds for a noisy accuracy, so
            # its stop never fired.
            if reference is None or best_ever.fitness > reference + cfg.stagnation_min_delta:
                reference = best_ever.fitness
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if (
                cfg.stagnation_generations is not None
                and stagnation_counter >= cfg.stagnation_generations
            ):
                stop_reason = "stagnation"
                break

            if gen == cfg.n_generations - 1:
                break  # nothing will evaluate another generation; don't breed one

            population = self._breed_next_generation(population, best_ever, rng)

        history.stop_reason = stop_reason
        assert best_ever is not None

        if not math.isfinite(best_ever.fitness):
            raise EvaluationError(
                "Every candidate evaluation in this run failed, so there is no "
                "usable model. See the logged warnings for the underlying errors."
            )

        logger.info(
            "Stopped: %s | best=%.4f %s | %d trainings (+%d cache hits)",
            stop_reason,
            best_ever.metadata.get("raw_score", best_ever.fitness),
            best_ever.architecture,
            history.total_evaluations,
            history.total_cache_hits,
        )

        best = self._finalize_best(best_ever, data, sign, history)

        test_score = None
        if data.has_test:
            test_score = float(self.backend.evaluate(best.model, data.test_view()))
            history.extras["test_score"] = test_score
            logger.info("Held-out test score of the winner: %.4f", test_score)

        return EvolutionResult(
            best_individual=best,
            history=history,
            final_population=population,
            test_score=test_score,
        )

    # ------------------------------------------------------------ internals
    def _initialize_population(
        self,
        seed: SeedType,
        seed_fitness: Optional[float],
        sign: float,
        rng: random.Random,
        cache: Dict[Architecture, Tuple[float, float]],
    ) -> List[Individual]:
        cfg = self.config
        population: List[Individual] = []

        if seed is not None:
            seed_arch, seed_model = resolve_seed(self.backend, seed)
            try:
                check_within_bounds(
                    seed_arch,
                    min_layers=cfg.min_layers,
                    max_layers=cfg.max_layers,
                    min_neurons=cfg.min_neurons,
                    max_neurons=cfg.max_neurons,
                )
            except InvalidArchitectureError as exc:
                raise InvalidArchitectureError(
                    f"The seed does not fit the search bounds: {exc} Widen "
                    f"min_layers/max_layers/min_neurons/max_neurons to cover "
                    f"{seed_arch}, or use layers_delta/neurons_delta "
                    "(evolve_architecture) to derive bounds from the seed."
                ) from exc

            ind = Individual(architecture=seed_arch, metadata={"origin": "seed"})
            if seed_fitness is not None:
                raw = float(seed_fitness)
                ind.fitness = raw * sign
                ind.metadata["raw_score"] = raw
                ind.model = seed_model
                if seed_model is not None:
                    # owned by the caller, not by the engine: never pass it to backend.cleanup()
                    ind.metadata["external_model"] = True
                if cfg.cache_evaluations:
                    cache[seed_arch] = (raw, ind.fitness)
            population.append(ind)

        while len(population) < cfg.population_size:
            arch = random_architecture(
                min_layers=cfg.min_layers,
                max_layers=cfg.max_layers,
                min_neurons=cfg.min_neurons,
                max_neurons=cfg.max_neurons,
                rng=rng,
            )
            population.append(
                Individual(architecture=arch, metadata={"origin": "random"})
            )
        return population

    def _evaluate_population(
        self,
        population: List[Individual],
        data: TrainingData,
        sign: float,
        cache: Dict[Architecture, Tuple[float, float]],
    ):
        cfg = self.config
        n_trained = n_cached = n_failed = 0
        survivors: List[Individual] = []

        for ind in population:
            if ind.is_evaluated:
                # elite carry-over or a seed with a known score
                survivors.append(ind)
                continue

            if cfg.cache_evaluations and ind.architecture in cache:
                raw, fit = cache[ind.architecture]
                ind.fitness = fit
                ind.metadata["raw_score"] = raw
                ind.metadata["cache_hit"] = True
                n_cached += 1
                survivors.append(ind)
                continue

            n_trained += 1
            model = None
            try:
                model = self.backend.build(ind.architecture, data)
                model = self._train(model, data, cfg.epochs_per_candidate)
                raw = float(self.backend.evaluate(model, data))
                if not math.isfinite(raw):
                    raise EvaluationError(
                        f"backend.evaluate returned a non-finite score ({raw})"
                    )
            except Exception as exc:
                # noqa: BLE001 -- deliberately broad; policy decides
                # (KeyboardInterrupt is a BaseException, so it propagates)
                n_failed += 1
                if model is not None:
                    self._safe_cleanup(model)
                handled = self._handle_evaluation_error(ind, exc)
                if handled is not None:
                    survivors.append(handled)
                continue

            ind.model = model
            ind.fitness = raw * sign
            ind.metadata["raw_score"] = raw
            if cfg.cache_evaluations:
                cache[ind.architecture] = (raw, ind.fitness)
            logger.debug("Evaluated %s -> %.4f", ind.architecture, raw)
            survivors.append(ind)

        return survivors, n_trained, n_cached, n_failed

    def _handle_evaluation_error(
        self, ind: Individual, exc: Exception
    ) -> Optional[Individual]:
        policy = self.config.on_evaluation_error
        message = f"Evaluation failed for {ind.architecture}: {exc!r}"

        if policy == "raise":
            raise EvaluationError(message) from exc
        if policy == "skip":
            logger.warning(
                "%s -- dropping this individual (on_evaluation_error='skip').",
                message,
            )
            return None

        logger.warning(
            "%s -- assigning worst possible fitness and continuing.", message
        )
        ind.fitness = float("-inf")
        ind.metadata["raw_score"] = float("nan")
        ind.metadata["error"] = repr(exc)
        return ind

    def _make_record(
        self,
        gen: int,
        population: List[Individual],
        best_ever: Individual,
        n_trained: int,
        n_cached: int,
        n_failed: int,
        t0: float,
        sign: float,
    ) -> GenerationRecord:
        nan = float("nan")
        raw_scores = [
            ind.metadata["raw_score"]
            for ind in population
            if math.isfinite(ind.metadata.get("raw_score", nan))
        ]
        mean_fitness = sum(raw_scores) / len(raw_scores) if raw_scores else nan
        worst = (min(raw_scores) if sign > 0 else max(raw_scores)) if raw_scores else nan

        gen_best = population[0]  # population is sorted best-first
        return GenerationRecord(
            generation=gen,
            best_fitness=gen_best.metadata.get("raw_score", nan),
            mean_fitness=mean_fitness,
            worst_fitness=worst,
            best_architecture=gen_best.architecture.as_list(),
            best_so_far=best_ever.metadata.get("raw_score", nan),
            population_size=len(population),
            n_evaluated=n_trained,
            n_cache_hits=n_cached,
            n_failed=n_failed,
            elapsed_seconds=time.monotonic() - t0,
        )

    def _breed_next_generation(
        self,
        population: List[Individual],
        best_ever: Individual,
        rng: random.Random,
    ) -> List[Individual]:
        cfg = self.config

        # Elitism: top performers pass through unchanged, with fitness and
        # model intact -- no re-training, and the best can never be lost. The
        # original had no equivalent: a generation's best survived only if
        # re-picked by roulette *and* not mutated, so results could (and, in
        # the thesis's own plots, did) go backwards between generations.
        elites = [
            Individual(
                architecture=ind.architecture,
                fitness=ind.fitness,
                model=ind.model,
                metadata=dict(ind.metadata),
            )
            for ind in population[: cfg.elitism_count]
        ]

        n_children = cfg.population_size - len(elites)
        n_parents = n_children + (n_children % 2)  # children come in pairs
        parents = select(population, n_parents, cfg, rng)

        children: List[Individual] = []
        for i in range(0, n_parents, 2):
            a1, a2 = maybe_crossover(
                parents[i].architecture, parents[i + 1].architecture, cfg, rng
            )
            a1, m1 = maybe_mutate(a1, cfg, rng)
            a2, m2 = maybe_mutate(a2, cfg, rng)
            children.append(
                Individual(
                    architecture=a1,
                    metadata={"origin": "offspring", "mutated": m1},
                )
            )
            children.append(
                Individual(
                    architecture=a2,
                    metadata={"origin": "offspring", "mutated": m2},
                )
            )

        next_population = elites + children[:n_children]
        self._release_models(population, next_population, best_ever)
        return next_population

    def _release_models(
        self,
        old: List[Individual],
        new: List[Individual],
        best_ever: Individual,
    ) -> None:
        """Give the backend a chance to free models that are about to be dropped."""
        keep = {id(ind.model) for ind in new if ind.model is not None}
        if best_ever.model is not None:
            keep.add(id(best_ever.model))
        for ind in old:
            if ind.model is not None and id(ind.model) not in keep:
                # Dropping our reference to a caller-supplied model is fine; running a backend's
                # resource-freeing hook on it would damage an object the engine does not own.
                if not ind.metadata.get("external_model"):
                    self._safe_cleanup(ind.model)
                ind.model = None

    def _safe_cleanup(self, model: Any) -> None:
        try:
            self.backend.cleanup(model)
        except Exception as exc:  # noqa: BLE001
            logger.debug("backend.cleanup raised %r (ignored)", exc)

    def _train(self, model: Any, data: TrainingData, epochs: int) -> Any:
        trained = self.backend.train(model, data, epochs)
        if trained is None:
            raise EvaluationError(
                f"{type(self.backend).__name__}.train() returned None; it must "
                "return the trained model (see ModelBackend.train)."
            )
        return trained

    def _finalize_best(
        self,
        best: Individual,
        data: TrainingData,
        sign: float,
        history: RunHistory,
    ) -> Individual:
        """Optionally retrain the winner with the full budget; make sure it
        has a model."""
        cfg = self.config

        if cfg.final_training_epochs is not None:
            epochs, why = cfg.final_training_epochs, "final_retrain"
        elif best.model is None:
            # e.g. the seed won on a user-supplied seed_fitness and was never
            # trained here
            epochs, why = cfg.epochs_per_candidate, "materialize"
        else:
            return best

        before = best.metadata.get("raw_score", float("nan"))
        logger.info(
            "%s: building %s from scratch and training %d epochs (search "
            "budget was %d)",
            why,
            best.architecture,
            epochs,
            cfg.epochs_per_candidate,
        )

        try:
            model = self.backend.build(best.architecture, data)
            model = self._train(model, data, epochs)
            raw = float(self.backend.evaluate(model, data))
            if not math.isfinite(raw):
                raise EvaluationError(f"non-finite score ({raw}) after {why}")
        except Exception as exc:
            # noqa: BLE001
            if best.model is not None and cfg.on_evaluation_error != "raise":
                logger.warning(
                    "%s failed (%r); returning the search-trained model instead.",
                    why,
                    exc,
                )
                return best
            raise EvaluationError(
                f"{why} failed for {best.architecture}: {exc!r}"
            ) from exc

        history.extras[why] = {
            "architecture": best.architecture.as_list(),
            "epochs": epochs,
            "score_during_search": before,
            "score_after": raw,
        }
        logger.info("%s result: %.4f (was %.4f)", why, raw, before)

        return Individual(
            architecture=best.architecture,
            fitness=raw * sign,
            model=model,
            metadata={**best.metadata, "raw_score": raw, why: True},
        )


# ---------------------------------------------------------------------------
# The function-style entry point from the original request:
# "a function which takes a DNN as input ... and outputs a new DNN"
# ---------------------------------------------------------------------------
def evolve_architecture(
    seed: SeedType,
    data: TrainingData,
    backend: ModelBackend,
    config: Optional[EvolutionConfig] = None,
    *,
    seed_fitness: Optional[float] = None,
    layers_delta: Optional[Sequence[int]] = None,
    neurons_delta: Optional[Sequence[int]] = None,
    return_result: bool = False,
    **config_overrides,
) -> Any:
    """Take a DNN (or a list of layer widths), return a new trained DNN.

    Search-space limits can be given directly as keyword arguments instead of
    building an :class:`EvolutionConfig`::

        new_model = evolve_architecture(
            my_model, data, KerasMLPBackend(),
            layers_delta=(-1, 2),      # may remove 1 / add 2 layers
            neurons_delta=(-8, 16),    # widths between (smallest - 8) and (largest + 16)
            epochs_per_candidate=30,
            population_size=12,
            n_generations=15,
            target_fitness=0.97,       # stop early at this accuracy
        )

    ``layers_delta``/``neurons_delta`` are relative to the seed (see
    :func:`~dnnga.config.bounds_relative_to`) and may not be combined with the
    matching absolute ``min_/max_`` keywords. Any other keyword must be an
    :class:`EvolutionConfig` field.

    Returns the best *model*; pass ``return_result=True`` for the full
    :class:`EvolutionResult` (history, architecture, scores) -- you usually
    want it.
    """
    base = config if config is not None else EvolutionConfig()
    overrides = dict(config_overrides)

    if layers_delta is not None or neurons_delta is not None:
        if seed is None:
            raise ConfigError(
                "layers_delta/neurons_delta are relative to a seed; pass a seed."
            )
        seed_arch, _ = resolve_seed(backend, seed)
        relative = bounds_relative_to(
            seed_arch, layers_delta=layers_delta, neurons_delta=neurons_delta
        )
        clash = sorted(set(relative) & set(overrides))
        if clash:
            raise ConfigError(
                f"{clash} were given explicitly and also derived from "
                "layers_delta/neurons_delta; use one or the other."
            )
        overrides.update(relative)

        # ``layers_delta=(0, 0)`` (or a delta clamped to the seed's depth) means
        # "keep the depth fixed". The default mutation weights include
        # add_layer/remove_layer, which EvolutionConfig rightly rejects as
        # impossible when min_layers == max_layers. Since the user expressed
        # the fixed depth through a relative bound, disable those two operators
        # for them instead of making them spell out the weights themselves.
        if (
            "min_layers" in relative
            and relative["min_layers"] == relative["max_layers"]
            and "mutation_operator_weights" not in overrides
        ):
            weights = dict(base.mutation_operator_weights)
            weights["add_layer"] = 0.0
            weights["remove_layer"] = 0.0
            overrides["mutation_operator_weights"] = weights

    final_config = (
        base.with_overrides(**overrides) if overrides else base
    )
    result = GeneticArchitectureSearch(backend, final_config).run(
        data, seed=seed, seed_fitness=seed_fitness
    )
    return result if return_result else result.best_model
