"""The genotype: a neural network's hidden-layer structure.

This is the direct analogue of the "chromosome" in the original thesis
code -- there, an individual was a plain Python ``list`` of ints such as
``[6, 6, 5, 6]``, meaning "4 hidden layers with 6, 6, 5 and 6 neurons".
Here it is the same idea wrapped in a small immutable value type instead of
a bare, mutable ``list``.

That is not decoration: the original code mutated those lists in place
(``mutate()`` in ``geneticAlgorithm.py``), and the roulette-selection step
can hand out the *same list object* to more than one slot of the next
generation (fitness-proportionate selection samples with replacement, so an
above-average individual is *expected* to be picked more than once). An
in-place mutation of one slot then silently changed every other slot
aliasing the same object, leaving each with a now-stale fitness value. A
frozen dataclass over a ``tuple`` makes that class of bug structurally
impossible: every operator returns a *new* ``Architecture``, so there is
nothing to alias.
"""

from __future__ import annotations

import numbers
import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Tuple

from .exceptions import ConfigError, InvalidArchitectureError


@dataclass(frozen=True)
class Architecture:
    """An immutable, hashable hidden-layer specification, e.g. ``(64, 32, 16)``.

    Only *hidden* layers are represented; input/output sizes come from the
    dataset at build time (as in the original, where ``hidden_units`` never
    included them). Widths may be any integer type (Python ``int``, NumPy
    integers) and are normalised to plain ``int``. ``bool`` and floats --
    even ``4.0`` -- are rejected on purpose rather than silently truncated.
    """

    layer_sizes: Tuple[int, ...]

    def __post_init__(self) -> None:
        raw = self.layer_sizes
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Iterable):
            raise InvalidArchitectureError(
                f"layer_sizes must be an iterable of positive ints, got {raw!r}."
            )

        sizes = tuple(raw)
        if len(sizes) == 0:
            raise InvalidArchitectureError(
                "An Architecture must have at least one hidden layer; got an "
                "empty layer_sizes tuple."
            )

        normalised = []
        for i, width in enumerate(sizes):
            if isinstance(width, bool) or not isinstance(width, numbers.Integral):
                raise InvalidArchitectureError(
                    f"layer_sizes[{i}] = {width!r} is not an int "
                    f"(got {type(width).__name__})."
                )
            width = int(width)
            if width < 1:
                raise InvalidArchitectureError(
                    f"layer_sizes[{i}] = {width} must be >= 1: a layer with 0 "
                    "or fewer neurons cannot pass any signal forward."
                )
            normalised.append(width)

        object.__setattr__(self, "layer_sizes", tuple(normalised))

    # -- convenience views -------------------------------------------------
    @property
    def n_layers(self) -> int:
        return len(self.layer_sizes)

    def as_list(self) -> list:
        return list(self.layer_sizes)

    def as_tuple(self) -> Tuple[int, ...]:
        return self.layer_sizes

    # -- "edits" always return a NEW Architecture --------------------------
    def replace_layer(self, index: int, new_width: int) -> "Architecture":
        sizes = list(self.layer_sizes)
        sizes[index] = new_width
        return Architecture(tuple(sizes))

    def insert_layer(self, index: int, width: int) -> "Architecture":
        sizes = list(self.layer_sizes)
        sizes.insert(index, width)
        return Architecture(tuple(sizes))

    def remove_layer(self, index: int) -> "Architecture":
        sizes = list(self.layer_sizes)
        del sizes[index]
        return Architecture(tuple(sizes))

    def __repr__(self) -> str:
        return f"Architecture({list(self.layer_sizes)})"

    def __str__(self) -> str:
        return "[" + ", ".join(str(w) for w in self.layer_sizes) + "]"


def validate_bounds(
    *,
    min_layers: int,
    max_layers: int,
    min_neurons: int,
    max_neurons: int,
) -> None:
    """Validate a set of structural bounds in isolation (raises ConfigError)."""
    problems = []
    if min_layers < 1:
        problems.append(f"min_layers must be >= 1, got {min_layers}")
    if max_layers < min_layers:
        problems.append(
            f"max_layers ({max_layers}) must be >= min_layers ({min_layers})"
        )
    if min_neurons < 1:
        problems.append(f"min_neurons must be >= 1, got {min_neurons}")
    if max_neurons < min_neurons:
        problems.append(
            f"max_neurons ({max_neurons}) must be >= min_neurons ({min_neurons})"
        )
    if problems:
        raise ConfigError("Invalid architecture bounds: " + "; ".join(problems))


def check_within_bounds(
    architecture: Architecture,
    *,
    min_layers: int,
    max_layers: int,
    min_neurons: int,
    max_neurons: int,
) -> None:
    """Raise InvalidArchitectureError if `architecture` violates the bounds."""
    if not (min_layers <= architecture.n_layers <= max_layers):
        raise InvalidArchitectureError(
            f"{architecture} has {architecture.n_layers} layers, outside "
            f"[{min_layers}, {max_layers}]."
        )
    for width in architecture.layer_sizes:
        if not (min_neurons <= width <= max_neurons):
            raise InvalidArchitectureError(
                f"{architecture} has a layer of width {width}, outside "
                f"[{min_neurons}, {max_neurons}]."
            )


def random_architecture(
    *,
    min_layers: int,
    max_layers: int,
    min_neurons: int,
    max_neurons: int,
    rng: random.Random,
) -> Architecture:
    """Uniformly random depth, then uniformly random width per layer.

    Analogue of ``generate_individual()`` in the original code, but taking an
    explicit :class:`random.Random` instead of the module-global ``random``
    functions, so a whole run is reproducible from one seed.
    """
    validate_bounds(
        min_layers=min_layers,
        max_layers=max_layers,
        min_neurons=min_neurons,
        max_neurons=max_neurons,
    )
    n_layers = rng.randint(min_layers, max_layers)
    return Architecture(
        tuple(rng.randint(min_neurons, max_neurons) for _ in range(n_layers))
    )


def architecture_from_any(value) -> Architecture:
    """Coerce an Architecture, or a list/tuple/array of positive ints, into one."""
    if isinstance(value, Architecture):
        return value
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise InvalidArchitectureError(
            f"Cannot interpret {value!r} as an Architecture: expected an "
            "Architecture, or a list/tuple of positive ints."
        )
    return Architecture(tuple(value))
