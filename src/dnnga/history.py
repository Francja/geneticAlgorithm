"""Structured records of a run, exportable to JSON/CSV.

The original's only record of a run was printed text (and, in the later
notebook, a hand-written .txt on Drive): readable once, not machine-readable,
and it never recorded *which individual* achieved the best score.
``RunHistory`` is structured data first; text/JSON/CSV are just views of it.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class GenerationRecord:
    generation: int
    best_fitness: float
    """Best score *in this generation's population*, in the metric's own units."""
    mean_fitness: float
    worst_fitness: float
    best_architecture: List[int]
    """Architecture that achieved ``best_fitness`` this generation."""
    best_so_far: float
    """Best score seen in any generation up to and including this one."""
    population_size: int
    n_evaluated: int
    """Candidates actually built + trained + scored this generation (the real
    compute cost). Elite carry-overs and cache hits are not counted."""
    n_cache_hits: int = 0
    n_failed: int = 0
    """Evaluations that raised; excluded from best/mean/worst."""
    elapsed_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def to_dict(self) -> dict:
        return asdict(self)


def _json_safe(obj):
    """NaN/inf are not valid strict JSON; emit null instead."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


@dataclass
class RunHistory:
    records: List[GenerationRecord] = field(default_factory=list)
    stop_reason: Optional[str] = None
    total_evaluations: int = 0
    total_cache_hits: int = 0
    extras: dict = field(default_factory=dict)
    """Run-level facts: ``final_retrain`` details, ``test_score``."""

    def add(self, record: GenerationRecord) -> None:
        self.records.append(record)
        self.total_evaluations += record.n_evaluated
        self.total_cache_hits += record.n_cache_hits

    def best_so_far(self) -> Optional[float]:
        """Best score seen so far, in the metric's own units (direction-aware,
        because the engine records it from the winning individual)."""
        return self.records[-1].best_so_far if self.records else None

    def to_dict(self) -> dict:
        return {
            "stop_reason": self.stop_reason,
            "total_evaluations": self.total_evaluations,
            "total_cache_hits": self.total_cache_hits,
            "extras": self.extras,
            "generations": [r.to_dict() for r in self.records],
        }

    def to_json(self, path: Optional[str] = None) -> str:
        text = json.dumps(_json_safe(self.to_dict()), indent=2)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_csv(self, path: str) -> None:
        names = [
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
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=names)
            writer.writeheader()
            for record in self.records:
                row = record.to_dict()
                row["best_architecture"] = str(row["best_architecture"])
                writer.writerow(row)

    def summary_text(self) -> str:
        lines = []
        for r in self.records:
            notes = []
            if r.n_cache_hits:
                notes.append(f"{r.n_cache_hits} cached")
            if r.n_failed:
                notes.append(f"{r.n_failed} failed")
            note = (", " + ", ".join(notes)) if notes else ""
            lines.append(
                f"Generation {r.generation}: best={r.best_fitness:.4f} "
                f"mean={r.mean_fitness:.4f} worst={r.worst_fitness:.4f} "
                f"best_so_far={r.best_so_far:.4f} "
                f"best_arch={r.best_architecture} "
                f"(trained {r.n_evaluated}/{r.population_size}{note}, "
                f"{r.elapsed_seconds:.1f}s)"
            )
        if self.stop_reason:
            lines.append(f"Stopped: {self.stop_reason}")
        lines.append(
            f"Total trainings: {self.total_evaluations} "
            f"(+{self.total_cache_hits} reused from cache)"
        )
        return "\n".join(lines)
