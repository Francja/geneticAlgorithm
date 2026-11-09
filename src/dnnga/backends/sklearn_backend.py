"""Reference backend on scikit-learn's MLPClassifier / MLPRegressor.

Light, no GPU, and ``hidden_layer_sizes`` is exactly the "list of hidden
layer widths" the original used for
``tf.estimator.DNNClassifier(hidden_units=...)``, so this is close to a
drop-in modern replacement for the original ``DNN()`` function without the
deprecated Estimator API. It is also the template for other backends (see
``base.py``).
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

from ..architecture import Architecture
from ..data import TrainingData
from ..exceptions import BackendNotAvailableError
from .base import ModelBackend

try:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPClassifier, MLPRegressor
except ImportError as exc:  # pragma: no cover - exercised via a subprocess test
    raise BackendNotAvailableError(
        "SklearnMLPBackend requires scikit-learn: `pip install scikit-learn`."
    ) from exc


class SklearnMLPBackend(ModelBackend):
    """Builds/trains/scores ``sklearn.neural_network`` MLPs.

    ``random_state`` is forwarded for reproducible initialisation; other
    keyword arguments (``activation``, ``solver``, ``alpha``,
    ``learning_rate_init``...) go to the MLP. ``hidden_layer_sizes`` and
    ``max_iter`` are controlled by dnnga (from the Architecture and
    ``epochs_per_candidate``) and cannot be passed.

    For ``adam``/``sgd``, ``max_iter`` counts epochs.
    """

    def __init__(self, random_state: Optional[int] = None, **mlp_kwargs: Any):
        clash = {"hidden_layer_sizes", "max_iter"} & set(mlp_kwargs)
        if clash:
            raise ValueError(
                f"{sorted(clash)} are controlled by dnnga (from the Architecture "
                "and EvolutionConfig.epochs_per_candidate) and cannot be passed "
                "in mlp_kwargs."
            )
        self.random_state = random_state
        self.mlp_kwargs = mlp_kwargs

    def build(self, architecture: Architecture, data: TrainingData) -> Any:
        cls = MLPClassifier if data.task == "classification" else MLPRegressor
        return cls(
            hidden_layer_sizes=architecture.as_tuple(),
            max_iter=1,
            random_state=self.random_state,
            **self.mlp_kwargs,
        )

    def train(self, model: Any, data: TrainingData, epochs: int) -> Any:
        model.max_iter = epochs
        with warnings.catch_warnings():
            # A candidate not converging within its epoch budget is expected
            # during a search and not an error; suppress this one well-known
            # warning only.
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(data.X_train, data.y_train)
        return model

    def evaluate(self, model: Any, data: TrainingData) -> float:
        # accuracy for the classifier, R^2 for the regressor: both "higher is better"
        return float(model.score(data.X_val, data.y_val))

    def extract_architecture(self, model: Any) -> Architecture:
        if not hasattr(model, "hidden_layer_sizes"):
            raise TypeError(
                f"{model!r} does not look like an MLPClassifier/MLPRegressor."
            )
        sizes = model.hidden_layer_sizes
        return Architecture((sizes,) if isinstance(sizes, int) else tuple(sizes))
