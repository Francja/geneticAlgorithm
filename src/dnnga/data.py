"""A small, framework-agnostic container for the data a backend needs.

Three roles, deliberately kept separate:

* **train** -- the weights are fitted on this.
* **val** -- candidates are *scored and selected* on this (it is the GA's
  fitness signal).
* **test** -- optional; touched exactly once, at the very end, on the
  winning model only.

Because the GA selects "the best of many" on the validation set, the
winner's validation score is optimistically biased; the test score is the
honest number. (The original notebook used the test CSV *as* the fitness
data, so its reported accuracies were selected on the data they were
reported on -- see docs/CODE_REVIEW.md.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TrainingData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    task: str = "classification"  # or "regression"
    X_test: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.X_train = np.asarray(self.X_train)
        self.y_train = np.asarray(self.y_train)
        self.X_val = np.asarray(self.X_val)
        self.y_val = np.asarray(self.y_val)

        if self.task not in ("classification", "regression"):
            raise ValueError(
                f"task must be 'classification' or 'regression', got {self.task!r}"
            )

        for name, X in (("X_train", self.X_train), ("X_val", self.X_val)):
            if X.ndim != 2:
                raise ValueError(
                    f"{name} must be 2-D (n_samples, n_features), got shape {X.shape}"
                )

        if self.X_train.shape[1] != self.X_val.shape[1]:
            raise ValueError(
                f"X_train has {self.X_train.shape[1]} features but X_val has "
                f"{self.X_val.shape[1]}; they must match."
            )
        if len(self.X_train) != len(self.y_train):
            raise ValueError(
                f"X_train has {len(self.X_train)} rows but y_train has "
                f"{len(self.y_train)}."
            )
        if len(self.X_val) != len(self.y_val):
            raise ValueError(
                f"X_val has {len(self.X_val)} rows but y_val has {len(self.y_val)}."
            )
        if len(self.X_train) == 0:
            raise ValueError("X_train is empty.")
        if len(self.X_val) == 0:
            raise ValueError(
                "X_val is empty -- fitness cannot be computed without validation data."
            )

        if (self.X_test is None) != (self.y_test is None):
            raise ValueError("Provide both X_test and y_test, or neither.")
        if self.X_test is not None:
            self.X_test = np.asarray(self.X_test)
            self.y_test = np.asarray(self.y_test)
            if self.X_test.ndim != 2 or self.X_test.shape[1] != self.X_train.shape[1]:
                raise ValueError(
                    f"X_test must be 2-D with {self.X_train.shape[1]} features, "
                    f"got shape {self.X_test.shape}"
                )
            if len(self.X_test) != len(self.y_test):
                raise ValueError(
                    f"X_test has {len(self.X_test)} rows but y_test has "
                    f"{len(self.y_test)}."
                )
            if len(self.X_test) == 0:
                raise ValueError("X_test is empty.")

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    @property
    def n_outputs(self) -> int:
        """Number of classes (classification) or regression targets."""
        if self.task == "classification":
            return int(len(np.unique(self.y_train)))
        return 1 if self.y_train.ndim == 1 else self.y_train.shape[1]

    @property
    def has_test(self) -> bool:
        return self.X_test is not None

    def test_view(self) -> "TrainingData":
        """A copy whose *validation* slot holds the test split.

        Backends only ever score on ``data.X_val``; passing them this view
        lets the same ``backend.evaluate`` compute the final held-out score
        without every backend needing a separate "evaluate on test" method.
        """
        if not self.has_test:
            raise ValueError("No test split was provided.")
        return TrainingData(
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_test,
            y_val=self.y_test,
            task=self.task,
        )
