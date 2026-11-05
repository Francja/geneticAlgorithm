"""The seam between the (framework-agnostic) genetic algorithm and a real
deep-learning framework.

The engine never imports TensorFlow, PyTorch or scikit-learn. It only calls
``build``/``train``/``evaluate`` on the ``ModelBackend`` it was given. This
is the fix for the original design, where CSV loading,
``tf.estimator.DNNClassifier`` construction and the training loop were wired
into one module-level ``DNN()`` function: pointing the GA at another dataset
or framework meant editing that function's body.

To add a framework, subclass ``ModelBackend`` and implement the three
abstract methods. ``sklearn_backend.py`` and ``keras_backend.py`` are
complete examples.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..architecture import Architecture
from ..data import TrainingData


class ModelBackend(ABC):
    """Strategy interface a genetic run delegates all framework-specific work to."""

    @abstractmethod
    def build(self, architecture: Architecture, data: TrainingData) -> Any:
        """Construct an *untrained* model for this architecture.

        `data` is passed (not just two ints) so a backend can look at
        ``data.task``, ``data.n_features`` and ``data.n_outputs`` to choose
        input/output sizes, activation and loss.
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, model: Any, data: TrainingData, epochs: int) -> Any:
        """Train `model` and **return the trained model**.

        The return value is what the engine keeps -- always return the model
        itself, even if you also trained it in place. This is an easy trap for
        Keras: ``keras_model.fit(...)`` returns a ``History`` object, not the
        model, so ``return model.fit(...)`` would silently hand the engine the
        wrong object. Write ``model.fit(...); return model``.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, model: Any, data: TrainingData) -> float:
        """Return one float score on ``data.X_val`` / ``data.y_val``.

        Return your metric in its natural units (accuracy, R^2, *or* a loss);
        set ``EvolutionConfig.direction`` to ``"minimize"`` for loss-like
        metrics and the engine flips the sign internally. Must be finite --
        NaN/inf is treated as a failed evaluation.
        """
        raise NotImplementedError

    def extract_architecture(self, model: Any) -> Architecture:
        """Optional: read the hidden-layer sizes out of an existing model, so a
        real model object can be used as the search's seed."""
        raise NotImplementedError(
            f"{type(self).__name__} cannot extract an Architecture from a model "
            "object. Pass the seed as a list of ints instead, e.g. seed=[64, 32]."
        )

    def cleanup(self, model: Any) -> None:
        """Optional hook called for models that lost the competition and are
        about to be dropped (e.g. to free GPU memory). Default: no-op."""
        return None
