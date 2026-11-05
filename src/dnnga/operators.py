"""Mutation, crossover and selection.

Every function is pure with respect to its ``Architecture``/``Individual``
arguments: nothing is mutated in place, new objects are returned. With a
frozen ``Architecture`` this makes the original code's aliasing bug (see
``architecture.py``) structurally impossible.

All randomness comes from an explicit ``random.Random`` passed by the
caller, never from global state, so a whole run is reproducible from
``EvolutionConfig.random_state``.
"""

from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple

from .architecture import Architecture
from .config import EvolutionConfig
from .individual import Individual

# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------


def _feasible_operators(architecture: Architecture, config: EvolutionConfig) -> dict:
    """Operator weights with structurally infeasible operators zeroed out and
    the rest renormalised, so a mutation event never "does nothing" just
    because it rolled an operator that cannot apply.

    Returns ``{}`` if nothing is feasible.
    """
    weights = dict(config.mutation_operator_weights)
    if architecture.n_layers >= config.max_layers:
        weights["add_layer"] = 0.0
    if architecture.n_layers <= config.min_layers:
        weights["remove_layer"] = 0.0

    total = sum(weights.values())
    if total <= 0:
        # Every operator the user asked for is inapplicable to this architecture
        # right now (e.g. they zeroed tweak_neurons and it sits at a depth bound).
        # Rather than forcing a mutation the config did not ask for, the event is
        # a no-op -- see maybe_mutate.
        return {}
    return {k: v / total for k, v in weights.items() if v > 0}


def mutate_tweak_neurons(
    architecture: Architecture,
    config: EvolutionConfig,
    rng: random.Random,
) -> Architecture:
    """Change one random layer's width by a random non-zero amount in
    ``[-neuron_mutation_step, +neuron_mutation_step]``.

    Generalises the original's fixed +/-1. By default values *clamp* at the
    bounds; the original *wrapped* (above max -> 1, below 1 -> max), which
    makes a "small" mutation a huge jump -- see
    ``EvolutionConfig.mutation_boundary_mode``.
    """
    index = rng.randrange(architecture.n_layers)
    step = rng.randint(1, config.neuron_mutation_step)
    if rng.random() < 0.5:
        step = -step
    new_value = architecture.layer_sizes[index] + step

    if config.mutation_boundary_mode == "wrap":
        span = config.max_neurons - config.min_neurons + 1
        new_value = config.min_neurons + (new_value - config.min_neurons) % span
    else:
        new_value = max(config.min_neurons, min(config.max_neurons, new_value))

    return architecture.replace_layer(index, new_value)


def mutate_add_layer(
    architecture: Architecture,
    config: EvolutionConfig,
    rng: random.Random,
) -> Architecture:
    """Insert one new layer of random width at a random position."""
    index = rng.randint(0, architecture.n_layers)  # both ends allowed
    width = rng.randint(config.min_neurons, config.max_neurons)
    return architecture.insert_layer(index, width)


def mutate_remove_layer(
    architecture: Architecture,
    config: EvolutionConfig,
    rng: random.Random,
) -> Architecture:
    """Delete one random layer (InvalidArchitectureError if it was the only one)."""
    return architecture.remove_layer(rng.randrange(architecture.n_layers))


_MUTATORS = {
    "tweak_neurons": mutate_tweak_neurons,
    "add_layer": mutate_add_layer,
    "remove_layer": mutate_remove_layer,
}


def maybe_mutate(
    architecture: Architecture,
    config: EvolutionConfig,
    rng: random.Random,
) -> Tuple[Architecture, bool]:
    """With probability ``mutation_rate`` apply exactly one feasible operator.

    Returns ``(architecture, did_mutate)``.
    """
    if rng.random() >= config.mutation_rate:
        return architecture, False

    weights = _feasible_operators(architecture, config)
    if not weights:
        return architecture, False

    name = rng.choices(list(weights), weights=list(weights.values()), k=1)[0]
    return _MUTATORS[name](architecture, config, rng), True


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------


def crossover_single_point(
    parent1: Architecture,
    parent2: Architecture,
    rng: random.Random,
) -> Tuple[Architecture, Architecture]:
    """Single-point crossover, identical in behaviour to the original ``crossover``.

    One cut ``k`` is drawn uniformly from ``[1, min(len1, len2) - 1]``;
    ``child1 = p1[:k] + p2[k:]`` and ``child2 = p2[:k] + p1[k:]``.
    Consequently ``child1`` has ``p2``'s depth and ``child2`` has ``p1``'s:
    crossover can *redistribute* depths between individuals but never create
    a depth that is not already in the pair -- which is why depth-changing
    mutations matter. Two single-layer parents have no interior cut point and
    simply swap.
    """
    max_cut = min(parent1.n_layers, parent2.n_layers) - 1
    if max_cut < 1:
        return parent2, parent1
    cut = rng.randint(1, max_cut)
    return (
        Architecture(parent1.layer_sizes[:cut] + parent2.layer_sizes[cut:]),
        Architecture(parent2.layer_sizes[:cut] + parent1.layer_sizes[cut:]),
    )


def crossover_uniform(
    parent1: Architecture,
    parent2: Architecture,
    rng: random.Random,
) -> Tuple[Architecture, Architecture]:
    """Position-wise uniform crossover over the overlapping depth.

    Each child keeps its own parent's depth and any tail beyond the shorter
    parent's length; every overlapping position is swapped between the
    children with probability 0.5. Depth is therefore preserved exactly (no
    drift towards deeper or shallower networks).
    """
    a, b = parent1.layer_sizes, parent2.layer_sizes
    c1, c2 = list(a), list(b)
    for i in range(min(len(a), len(b))):
        if rng.random() < 0.5:
            c1[i], c2[i] = b[i], a[i]
    return Architecture(tuple(c1)), Architecture(tuple(c2))


_CROSSOVERS = {
    "single_point": crossover_single_point,
    "uniform": crossover_uniform,
}


def maybe_crossover(
    parent1: Architecture,
    parent2: Architecture,
    config: EvolutionConfig,
    rng: random.Random,
) -> Tuple[Architecture, Architecture]:
    """Crossover with probability ``p_crossover``; otherwise parents pass through."""
    if config.crossover_strategy == "none" or rng.random() >= config.p_crossover:
        return parent1, parent2
    return _CROSSOVERS[config.crossover_strategy](parent1, parent2, rng)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _fitness_key(individual: Individual) -> float:
    if individual.fitness is None:
        raise ValueError(
            f"{individual} has not been evaluated; selection needs every "
            "individual to have a fitness."
        )
    return individual.fitness


def select_tournament(
    population: Sequence[Individual],
    n: int,
    config: EvolutionConfig,
    rng: random.Random,
) -> List[Individual]:
    """Repeat n times: sample ``tournament_size`` individuals (without
    replacement within one tournament) and keep the fittest.

    The default strategy because it needs no special-casing for
    zero/negative/equal/-inf fitness values.
    """
    k = min(config.tournament_size, len(population))
    pool = list(population)
    return [max(rng.sample(pool, k), key=_fitness_key) for _ in range(n)]


def select_roulette(
    population: Sequence[Individual],
    n: int,
    config: EvolutionConfig,
    rng: random.Random,
) -> List[Individual]:
    """Fitness-proportionate selection with replacement.

    The original hand-rolled this (cumulative table + linear scan) with two
    edge-case bugs: it assumed strictly positive costs without checking, and
    floating-point error can leave the last cumulative probability just under
    1.0, so a draw of e.g. 0.9999999999 selects nobody and the next
    generation silently comes up one short. ``random.choices`` implements
    weighted sampling correctly.

    Weights are shifted to be non-negative: if the lowest *finite* fitness is
    negative everything is shifted up by ``|min| + 1e-9``. Individuals with
    non-finite fitness (``-inf`` from a penalised failure) get weight 0 --
    they are excluded from the shift arithmetic because ``-inf - (-inf)`` is
    NaN and one NaN weight would make ``random.choices`` fail for the whole
    population.

    Caveat (why tournament is the default): roulette is sensitive to the
    *scale and offset* of the metric. Accuracies of 0.90/0.95/0.99 get almost
    equal weights (weak selection pressure), while the same ranking expressed
    as a loss or with a negative offset gets very different pressure.
    """
    fitnesses = [ind.fitness for ind in population]
    finite = [f for f in fitnesses if f is not None and math.isfinite(f)]
    if not finite:
        raise ValueError(
            "select_roulette: every individual has non-finite fitness (all "
            "evaluations failed?); cannot form a selection distribution."
        )

    min_fit = min(finite)
    shift = (-min_fit + 1e-9) if min_fit < 0 else 1e-9
    weights = [
        (f + shift) if f is not None and math.isfinite(f) else 0.0
        for f in fitnesses
    ]
    return rng.choices(list(population), weights=weights, k=n)


_SELECTORS = {"tournament": select_tournament, "roulette": select_roulette}


def select(
    population: Sequence[Individual],
    n: int,
    config: EvolutionConfig,
    rng: random.Random,
) -> List[Individual]:
    return _SELECTORS[config.selection_strategy](population, n, config, rng)
