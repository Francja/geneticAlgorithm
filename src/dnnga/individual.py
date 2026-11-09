"""A single member of the population: an architecture plus what we know about it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .architecture import Architecture


@dataclass
class Individual:
    """One candidate network.

    ``fitness`` is always "higher is better" internally, whatever the user's
    metric is (see ``EvolutionConfig.direction``); the metric's original,
    un-negated value lives in ``metadata["raw_score"]`` so logs and reports
    show the number the user recognises.

    Unlike the original ``[architecture_list, cost]`` pairs (which used
    ``isinstance(x[0], list)`` as an improvised "has this been scored yet?"
    tag), ``fitness`` is simply ``None`` until scored.
    """

    architecture: Architecture
    fitness: Optional[float] = None
    model: Optional[Any] = None
    metadata: dict = field(default_factory=dict)

    @property
    def is_evaluated(self) -> bool:
        return self.fitness is not None

    def __repr__(self) -> str:
        fitness_str = (
            "unevaluated" if self.fitness is None else f"{self.fitness:.4f}"
        )
        return f"Individual({self.architecture}, fitness={fitness_str})"
