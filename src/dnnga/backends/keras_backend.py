"""Keras 3 backend (TensorFlow, JAX or PyTorch Keras backend).

The modern successor to the original ``tf.estimator.DNNClassifier`` call,
which is deprecated. Builds a ``Sequential`` stack of ``Dense`` hidden layers
from the Architecture, with a softmax head (classification) or a linear head
(regression).

Install with ``pip install tensorflow`` (or any Keras 3 backend).

Notes
-----
* ``random_state`` calls ``keras.utils.set_random_seed`` in ``build`` -- that
  seeds Python, NumPy and the backend *globally*, i.e. it also affects your
  own code.
* Class labels need not be 0..K-1: they are mapped through
  ``np.unique(y_train)``. Validation labels never seen in training simply
  count as wrong.
* ``cleanup`` is a no-op: Keras 3 models are freed by garbage collection once
  the engine drops its reference. For very long GPU runs consider calling
  ``keras.backend.clear_session()`` yourself between searches.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..architecture import Architecture
from ..data import TrainingData
from ..exceptions import BackendNotAvailableError
from ..logging_utils import get_logger
from .base import ModelBackend

try:
    import keras
except ImportError as exc:  # pragma: no cover
    raise BackendNotAvailableError(
        "KerasMLPBackend requires Keras 3 with a backend: `pip install tensorflow`."
    ) from exc

_OPTIMIZERS = {"adam": "Adam", "sgd": "SGD", "rmsprop": "RMSprop", "adagrad": "Adagrad"}
_IGNORED_LAYERS = (
    keras.layers.Dropout,
    keras.layers.Activation,
    keras.layers.ReLU,
    keras.layers.BatchNormalization,
    keras.layers.Flatten,
    keras.layers.InputLayer,
)
logger = get_logger()


class KerasMLPBackend(ModelBackend):
    def __init__(
        self,
        activation: str = "relu",
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        if optimizer.lower() not in _OPTIMIZERS:
            raise ValueError(
                f"optimizer must be one of {sorted(_OPTIMIZERS)}, got {optimizer!r}"
            )
        self.activation = activation
        self.optimizer = optimizer.lower()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose

    # -- helpers --------------------------------------------------------
    @staticmethod
    def _classes(data: TrainingData) -> np.ndarray:
        return np.unique(data.y_train)

    def _targets(self, data: TrainingData) -> np.ndarray:
        if data.task == "classification":
            return np.searchsorted(self._classes(data), data.y_train).astype("int32")
        y = np.asarray(data.y_train, dtype="float32")
        return y.reshape(len(y), -1)

    # -- ModelBackend ---------------------------------------------------
    def build(self, architecture: Architecture, data: TrainingData) -> Any:
        if self.random_state is not None:
            keras.utils.set_random_seed(self.random_state)

        layers = [keras.Input(shape=(data.n_features,))]
        layers += [
            keras.layers.Dense(w, activation=self.activation)
            for w in architecture.layer_sizes
        ]
        if data.task == "classification":
            layers.append(keras.layers.Dense(data.n_outputs, activation="softmax"))
            loss = "sparse_categorical_crossentropy"
        else:
            layers.append(keras.layers.Dense(data.n_outputs))
            loss = "mse"

        model = keras.Sequential(layers)
        opt = getattr(keras.optimizers, _OPTIMIZERS[self.optimizer])(
            learning_rate=self.learning_rate
        )
        model.compile(optimizer=opt, loss=loss)
        return model

    def train(self, model: Any, data: TrainingData, epochs: int) -> Any:
        model.fit(
            np.asarray(data.X_train, dtype="float32"),
            self._targets(data),
            epochs=epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            shuffle=True,
        )
        return model  # NOT the History object that fit() returns

    def evaluate(self, model: Any, data: TrainingData) -> float:
        X = np.asarray(data.X_val, dtype="float32")
        preds = model.predict(X, batch_size=max(256, self.batch_size), verbose=0)
        if data.task == "classification":
            labels = self._classes(data)[np.argmax(preds, axis=1)]
            return float(np.mean(labels == data.y_val))

        y = np.asarray(data.y_val, dtype="float64").reshape(len(preds), -1)
        p = np.asarray(preds, dtype="float64").reshape(len(preds), -1)
        ss_res = float(np.sum((y - p) ** 2))
        ss_tot = float(np.sum((y - y.mean(axis=0)) ** 2))
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return 1.0 - ss_res / ss_tot  # R^2, higher is better

    def extract_architecture(self, model: Any) -> Architecture:
        """Hidden-layer widths of a plain ``Dense`` stack.

        Only Dense stacks are understood: this backend rebuilds every
        candidate as ``Dense`` layers, so a convolutional/recurrent/attention
        model cannot be represented -- that raises instead of silently
        flattening it into an MLP. Dropout/BatchNorm/Activation/Flatten layers
        are tolerated but *not* carried over to rebuilt candidates (a warning
        says so).
        """
        unsupported = sorted(
            {
                type(layer).__name__
                for layer in model.layers
                if not isinstance(layer, (keras.layers.Dense,) + _IGNORED_LAYERS)
            }
        )
        if unsupported:
            raise ValueError(
                f"KerasMLPBackend only supports plain Dense stacks; this model "
                f"contains {unsupported}. Pass the seed as a list of hidden-layer "
                "widths instead, or write a backend for your model family."
            )

        dropped = sorted(
            {
                type(layer).__name__
                for layer in model.layers
                if isinstance(layer, _IGNORED_LAYERS)
                and not isinstance(layer, keras.layers.InputLayer)
            }
        )
        if dropped:
            logger.warning(
                "extract_architecture ignores %s layers; evolved candidates are "
                "plain Dense stacks.",
                dropped,
            )

        dense = [layer for layer in model.layers if isinstance(layer, keras.layers.Dense)]
        if len(dense) < 2:
            raise ValueError(
                "Need at least one hidden Dense layer plus the output layer to "
                f"extract an Architecture; found {len(dense)} Dense layer(s)."
            )
        return Architecture(tuple(int(layer.units) for layer in dense[:-1]))
