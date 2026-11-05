"""Same idea with a real Keras (TensorFlow) model -- the modern replacement for
the deprecated ``tf.estimator.DNNClassifier`` used by the original notebook.

The task is two concentric circles: a 1-neuron hidden layer cannot separate
them, so network *capacity* genuinely matters here (unlike Iris, where almost
any architecture scores ~97-100%). The baseline gets the SAME final training
budget as the evolved winner, so any gain is due to the architecture, not to
training longer.

Needs: pip install tensorflow
Run:   python examples/keras_example.py
"""

import keras
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

from dnnga import Architecture, TrainingData, evolve_architecture
from dnnga.backends import KerasMLPBackend

X, y = make_circles(n_samples=800, noise=0.12, factor=0.45, random_state=0)
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=0
)
data = TrainingData(X_train, y_train, X_val, y_val, X_test=X_test, y_test=y_test)

backend = KerasMLPBackend(learning_rate=0.01, batch_size=32, random_state=0)
FINAL_EPOCHS = 80

# the DNN you already have: one hidden neuron, trained for the same budget the
# winner will get
my_model = backend.train(
    backend.build(Architecture((1,)), data), data, epochs=FINAL_EPOCHS
)
baseline_val = backend.evaluate(my_model, data)
baseline_test = backend.evaluate(my_model, data.test_view())
print(
    f"your model {backend.extract_architecture(my_model).as_list()}: "
    f"val {baseline_val:.3f} | test {baseline_test:.3f} ({FINAL_EPOCHS} epochs)"
)

result = evolve_architecture(
    my_model,
    data,
    backend,
    return_result=True,
    layers_delta=(0, 2),
    neurons_delta=(0, 15),      # up to 2 more layers, widths up to 16
    epochs_per_candidate=25,
    final_training_epochs=FINAL_EPOCHS,
    population_size=8,
    n_generations=5,
    random_state=3,
    stagnation_generations=None,
)

print(
    f"\nevolved {result.best_architecture} ({result.history.stop_reason}): "
    f"val {result.best_fitness:.3f} | test {result.test_score:.3f} "
    f"[{result.history.total_evaluations} trainings, "
    f"{result.history.total_cache_hits} cache hits]"
)
assert isinstance(result.best_model, keras.Model)
print(f"returned a trained keras.Model with {result.best_model.count_params()} parameters")
