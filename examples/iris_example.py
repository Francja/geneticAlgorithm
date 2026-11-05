"""The thesis experiment (evolve an Iris classifier's hidden layers), redone
with dnnga.

Differences from the original notebook, all deliberate:

* three-way split: the GA scores candidates on a *validation* split; the *test*
  split is touched once, on the winner (the notebook selected on the test CSV
  itself);
* the winner is returned (architecture + trained model), not just printed as a
  number;
* depth can change (add/remove layer), the best individual is never lost
  (elitism), duplicate architectures are not re-trained (cache), and a random
  seed makes it reproducible.

Run: python examples/iris_example.py
"""

import logging
import tempfile
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dnnga import (
    EvolutionConfig,
    GeneticArchitectureSearch,
    TrainingData,
    configure_logging,
)
from dnnga.backends import SklearnMLPBackend

configure_logging(level=logging.INFO)

X, y = load_iris(return_X_y=True)
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=0
)
scaler = StandardScaler().fit(X_train)  # fit on TRAIN only: no leakage

data = TrainingData(
    scaler.transform(X_train),
    y_train,
    scaler.transform(X_val),
    y_val,
    X_test=scaler.transform(X_test),
    y_test=y_test,
)
print(f"split: train={len(y_train)} val={len(y_val)} test={len(y_test)}\n")

config = EvolutionConfig(
    min_layers=1,
    max_layers=4,              # was MIN_LAYER / MAX_LAYER
    min_neurons=2,
    max_neurons=24,            # was MAX_NEURONS
    population_size=10,
    n_generations=8,           # was POPULATION / GENERATIONS (any size, not "divisible by 4")
    epochs_per_candidate=30,   # cheap, noisy proxy budget used to RANK architectures
    final_training_epochs=600, # full budget, spent once, on the winner
    stagnation_generations=5,  # was NO_EVOLVE_GENS
    random_state=42,
)
backend = SklearnMLPBackend(random_state=0, learning_rate_init=0.01)
result = GeneticArchitectureSearch(backend, config).run(data, seed=[8])

print("\n" + result.history.summary_text())
print("\n" + repr(result))
print(f"winner : {result.best_architecture}")
print(f"val    : {result.best_fitness:.4f} <- optimistically biased (max over many candidates)")
print(f"test   : {result.test_score:.4f} <- the honest number (data the GA never selected on)")

out = Path(tempfile.gettempdir())
result.history.to_json(str(out / "dnnga_iris_history.json"))
result.history.to_csv(str(out / "dnnga_iris_history.csv"))
print(f"history written to {out}/dnnga_iris_history.(json|csv)")
