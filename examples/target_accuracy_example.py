"""'Train until a desired accuracy' using the function-style API.

``evolve_architecture(model_or_layers, data, backend, **limits)`` -> trained model

Search-space limits are keyword arguments, relative to the network you pass in:

* ``layers_delta=(-1, 2)`` may remove 1 / add 2 layers
* ``neurons_delta=(-4, 12)`` widths may range from (smallest - 4) to (largest + 12)

Search stops as soon as ``target_fitness`` is reached (or the generation /
stagnation limits hit).

Run: python examples/target_accuracy_example.py
"""

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dnnga import Architecture, TrainingData, evolve_architecture
from dnnga.backends import SklearnMLPBackend

X, y = load_wine(return_X_y=True)
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=1
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=1
)
sc = StandardScaler().fit(X_train)

data = TrainingData(
    sc.transform(X_train),
    y_train,
    sc.transform(X_val),
    y_val,
    X_test=sc.transform(X_test),
    y_test=y_test,
)
backend = SklearnMLPBackend(random_state=0, learning_rate_init=0.01)

# --- 1. the network we start from (deliberately tiny), and what it scores ------
seed = [2]
model = backend.train(
    backend.build(Architecture(tuple(seed)), data), data, epochs=15
)
current = backend.evaluate(model, data)
print(f"starting network {seed}: validation accuracy {current:.3f}")

# --- 2. evolve until 97% validation accuracy -----------------------------------
result = evolve_architecture(
    model,
    data,
    backend,                  # a real model object works as the seed
    seed_fitness=current,     # "we input the current accuracy" -> not re-trained
    layers_delta=(0, 2),
    neurons_delta=(0, 14),
    target_fitness=0.97,
    epochs_per_candidate=15,
    population_size=10,
    n_generations=12,
    random_state=7,
    return_result=True,
)

print(f"\nstopped because: {result.history.stop_reason}")
print(
    f"new architecture: {result.best_architecture} "
    f"(validation {result.best_fitness:.3f}, "
    f"held-out test {result.test_score:.3f})"
)
print(
    f"cost: {result.history.total_evaluations} trainings "
    f"(+{result.history.total_cache_hits} duplicates reused from cache)"
)

new_model = result.best_model  # a fitted sklearn MLPClassifier, ready to use
print(
    "predictions on 5 test rows:",
    new_model.predict(data.X_test[:5]),
    "truth:",
    data.y_test[:5],
)
