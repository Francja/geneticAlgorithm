# dnnga — evolve a network's layers and neurons with a genetic algorithm

[![CI](https://github.com/xWuWux/geneticAlgorithm/actions/workflows/ci.yml/badge.svg)](https://github.com/xWuWux/geneticAlgorithm/actions/workflows/ci.yml)
[![Mutation testing](https://github.com/xWuWux/geneticAlgorithm/actions/workflows/mutation.yml/badge.svg)](https://github.com/xWuWux/geneticAlgorithm/actions/workflows/mutation.yml)

Give it a DNN (or just a list of hidden-layer widths) and some data; it searches
for a better *structure* — more or fewer layers, wider or narrower layers — by
evolving a population of candidate architectures, training each one, and keeping
the fittest.

`dnnga` grew out of the 2021 engineering thesis *"Optymalizacja sieci
neuronowej za pomocą algorytmów genetycznych"* (Optimising a neural network with
genetic algorithms). It is a framework-agnostic, tested rewrite of the thesis
code (`geneticAlgorithm.py` and the Colab notebook) that fixes the bugs and
method problems found when reviewing it — see
[`docs/CODE_REVIEW.md`](docs/CODE_REVIEW.md).

> **Cost warning.** Every candidate is trained, so a run costs roughly
> `population_size × n_generations` trainings (less with the built-in cache and
> elitism). It is a search tool, not a free lunch.

## Contents

- [Background: the thesis](#background-the-thesis)
- [How it works](#how-it-works) (flowchart)
- [Install](#install) · [Quick start](#quick-start--a-function-that-takes-a-dnn-and-returns-a-new-dnn) · [Command line](#command-line)
- [Software variables](#software-variables)
- [The original thesis experiment](#the-original-thesis-experiment) — algorithm, parameters, results
- [What the code review found](#what-the-code-review-found)
- [Error handling and logging](#error-handling-and-logging) · [Writing a backend](#writing-a-backend) · [Using it honestly](#using-it-honestly) · [Limitations](#limitations-not-implemented)
- [Repository layout](#repository-layout) · [Development](#development) · [Project status and roadmap](#project-status-and-roadmap)
- [Authors, citation and license](#authors-citation-and-license)

## Background: the thesis

| | |
|---|---|
| **Title** | *Optymalizacja sieci neuronowej za pomocą algorytmów genetycznych* (Optimising a neural network with genetic algorithms) |
| **Authors** | Adam Bielewicz, Wojciech Woźniak |
| **Type** | Engineering thesis (*praca inżynierska*), Computer Science |
| **Institution** | WSB University in Gdańsk (Wyższa Szkoła Bankowa w Gdańsku), Faculty of Economics and Management, Gdynia |
| **Supervisor** | dr Ireneusz Miciuła |
| **Year** | 2021 |
| **Full text** | [PDF in `docs/`](docs/A.%20BIelewicz%2C%20W.Wo%C5%BAniak%20%20-%20Optymalizacja%20sieci%20neuronowej%20za%20pomoc%C4%85%20algorytm%C3%B3w%20genetycznych.pdf) (Polish, 58 pages) |

**Goal (from the introduction).** Show that a genetic algorithm can find a good
*topology* for an artificial neural network — how many hidden layers and how many
neurons in each — instead of choosing it by hand, and compare GA-driven training
with gradient-based optimisation.

**Contents.** Chapters 1–4 cover the theory: evolutionary and genetic algorithms
(terminology, the classic GA with roulette selection, single-point crossover and
bit-flip mutation, illustrated on 6-bit chromosomes maximising *f(x) = x^(3/2)*),
the biological and McCulloch–Pitts neuron, the perceptron, activation functions
(sigmoid, tanh, softmax, softsign, ReLU, ELU), network types (RBF, Kohonen,
recurrent), and training methods (gradient descent, backpropagation). Chapter 5
describes the GA written for the thesis and its results on the Iris dataset —
summarised in [The original thesis experiment](#the-original-thesis-experiment).

**Conclusion (from the thesis).** Under the conditions studied (one network type,
one GA), a GA can reach higher classification accuracy, and applying it to
existing networks may be worthwhile; the cost in run time and compute, and how
much such accuracy matters commercially, are left as open questions.

## How it works

`dnnga` treats a network's **structure** as a genome — the tuple of hidden-layer
widths, e.g. `(64, 32, 16)` — and evolves a population of such genomes. Each
candidate is built, trained briefly and scored on the **validation** split; the
best ones are kept and recombined until a stopping criterion fires. The winner
is optionally retrained with a full budget and scored once on the **test**
split, which is never used for selection.

```mermaid
flowchart TD
    IN(["Input: seed model or layer list, TrainingData, backend"]) --> CFG["Validate EvolutionConfig<br/>(layers_delta / neurons_delta → min/max bounds)"]
    CFG --> INIT["Generation 0:<br/>seed + random architectures within bounds"]
    INIT --> EVAL

    subgraph EVAL ["Evaluate population"]
        direction TB
        K{"Fitness already known?<br/>(elite or seed_fitness)"} -- no --> C{"Architecture<br/>in cache?"}
        C -- no --> T["build → train for epochs_per_candidate<br/>→ evaluate on validation split"]
        T -- "error / NaN" --> ERR["on_evaluation_error:<br/>penalize · skip · raise"]
        C -- yes --> R["Reuse cached score"]
        K -- yes --> KEEP["Keep score"]
    end

    EVAL --> SORT["Sort by fitness, update best-so-far,<br/>record generation in RunHistory"]
    SORT --> STOP{"Stop?<br/>target_fitness reached ·<br/>stagnation_generations ·<br/>n_generations"}
    STOP -- no --> ELITE["Copy top elitism_count unchanged"]
    ELITE --> SEL["Select parents<br/>(tournament / roulette)"]
    SEL --> X["Crossover with p_crossover<br/>(single_point / uniform / none)"]
    X --> MUT["Mutate with mutation_rate — one of:<br/>tweak_neurons · add_layer · remove_layer"]
    MUT --> EVAL
    STOP -- yes --> FIN["Retrain winner for final_training_epochs<br/>(if set)"]
    FIN --> TEST["Score winner once on test split<br/>(if given)"]
    TEST --> OUT(["EvolutionResult:<br/>best_model, best_architecture,<br/>best_fitness, test_score, history"])
```

* **Genotype** — `Architecture((64, 32, 16))`: hidden-layer widths only
  (input/output sizes come from the data). Immutable.
* **Mutation** (with probability `mutation_rate`, exactly one of):
  `tweak_neurons` (width ± up to `neuron_mutation_step`, clamped), `add_layer`
  (random width, random position), `remove_layer`. Operators that are
  infeasible at the current depth are skipped.
* **Fitness** — whatever your backend's `evaluate` returns on the validation
  split; set `direction="minimize"` for losses.
* **Efficiency** — elites are never re-trained; `cache_evaluations` reuses the
  score of an architecture seen before (duplicates become common as the
  population converges); `epochs_per_candidate` is a cheap proxy,
  `final_training_epochs` spends the real budget once.

## Install

```bash
pip install -e ".[sklearn]"         # scikit-learn backend (light, CPU)
pip install -e ".[sklearn,keras]"   # + Keras/TensorFlow backend
pip install -e ".[dev]"             # pytest
```

Needs Python ≥ 3.9 (developed and tested on 3.12 with NumPy 2.5,
scikit-learn 1.9, TensorFlow 2.21 / Keras 3.15). `import dnnga` itself needs
only NumPy; each backend imports its framework lazily.

## Quick start — "a function that takes a DNN and returns a new DNN"

```python
from dnnga import TrainingData, evolve_architecture
from dnnga.backends import KerasMLPBackend   # or SklearnMLPBackend

data = TrainingData(
    X_train, y_train, X_val, y_val,      # candidates are scored + selected on val
    X_test=X_test, y_test=y_test,        # optional: scored once, on the winner only
)
backend = KerasMLPBackend(learning_rate=0.01, random_state=0)

result = evolve_architecture(
    my_model, data, backend,        # a model object, or a plain list such as [64, 32]
    layers_delta=(-1, 2),           # may remove 1 / add 2 layers (relative to my_model)
    neurons_delta=(-8, 16),         # widths from (smallest-8) to (largest+16), floored at 1
    epochs_per_candidate=20,        # how long each candidate trains
    final_training_epochs=100,      # winner is retrained once with a full budget
    population_size=12,
    n_generations=15,
    target_fitness=0.97,            # "train until this accuracy" - stops as soon as it is reached
    return_result=True,             # False (default) returns just the best model
)

result.best_model               # trained model (same framework as the backend)
result.best_architecture        # e.g. Architecture([16, 3])
result.best_fitness, result.test_score   # validation score (optimistic) / held-out score (honest)
print(result.history.summary_text())     # per-generation log; .to_json(path) / .to_csv(path) also exist
```

* **Two ways to bound the search.** Relative to your network (`layers_delta`,
  `neurons_delta`) or absolute (`min_layers=1, max_layers=5, min_neurons=2,
  max_neurons=64`). Combining a delta with the matching absolute bound raises
  `ConfigError`. `neurons_delta` produces one *global* width range for all
  layers; if you meant a per-layer range, use absolute bounds.
  `layers_delta=(0, 0)` fixes the depth and automatically disables the
  add/remove-layer mutations.
* **"Train until a desired accuracy"** is the same function with
  `target_fitness=...`. If you already know your network's score pass
  `seed_fitness=` and it is not re-trained ("I input the current accuracy");
  otherwise it is scored like any candidate.
* **Class API** (more control):
  `GeneticArchitectureSearch(backend, EvolutionConfig(...)).run(data, seed=[16, 8])`.
* **Only the seed's structure is used.** Its weights are not reused: every
  candidate, the seed included, is trained from scratch (unless you supply
  `seed_fitness`, in which case your model object is kept as that individual's
  model).

### Measured examples

From `examples/` — seeds fixed, so numbers may differ slightly across library
versions.

`keras_example.py` — two concentric circles, where a 1-neuron network cannot
succeed; the baseline gets the *same* 80-epoch budget:

```text
your model [1]: val 0.656 | test 0.637 (80 epochs)
evolved [16, 3] (max_generations): val 0.981 | test 0.988 [16 trainings, 20 cache hits]
```

`target_accuracy_example.py` — start from `[2]` on the wine data, stop at 97 %:

```text
starting network [2]: validation accuracy 0.667
stopped because: target_fitness_reached
new architecture: [16] (validation 0.972, held-out test 0.972)
cost: 9 trainings
```

`iris_example.py` — the thesis experiment. Iris is *too easy* to show anything:
generation 0 already contains a candidate at 1.000 validation accuracy. It does
show the selection bias, though: the winner scored 1.000 during the search, and
the same architecture trained properly scores 0.967 on validation and on test.

## Command line

Installing the package also installs a `dnnga` console script, which runs one
search without writing Python:

```bash
dnnga --mode sklearn --population 12 --generations 8 --target-score 0.97
dnnga --mode synthetic                    # toy fitness: no framework, no training
dnnga --mode keras --seed-layers 16,8 --seed 0 --log-file run.log
python -m dnnga --mode synthetic          # the same entry point
```

* `--mode synthetic` (default) scores architectures with the original script's
  toy formula, so it needs only NumPy and trains nothing — a smoke test and a
  way to watch the GA loop itself.
* `--mode sklearn` / `--mode keras` use the Iris dataset with a
  train/validation/test split: candidates are scored on **validation**, and the
  winner is scored once on **test**, so the printed held-out score was not used
  for selection.
* `--population` / `--generations` accept any value (the original required a
  population divisible by 4). `--seed-layers 16,8` starts from a known
  structure; `--seed` makes the GA, the data split and weight initialisation
  reproducible. Other options map to `EvolutionConfig` fields
  (`--min-layers`, `--max-neurons`, `--epochs-per-candidate`,
  `--final-training-epochs`, `--elitism-count`, `--direction`, …); run
  `dnnga --help` for the full list.
* Exit code `2` means the input was rejected before any compute was spent; the
  reason is printed on stderr.

## Software variables

### `evolve_architecture(seed, data, backend, config=None, **kwargs)`

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `seed` | model, `Architecture` or `list[int]` | — | starting network; only its structure is used |
| `data` | `TrainingData` | — | train / validation (+ optional test) arrays |
| `backend` | `ModelBackend` | — | `SklearnMLPBackend`, `KerasMLPBackend` or your own |
| `config` | `EvolutionConfig` | `EvolutionConfig()` | base settings; keyword arguments override it |
| `seed_fitness` | `float` | `None` | known score of the seed; skips retraining it |
| `layers_delta` | `(int, int)` | `None` | depth range relative to the seed, e.g. `(-1, 2)` |
| `neurons_delta` | `(int, int)` | `None` | width range relative to the seed's smallest / largest layer |
| `return_result` | `bool` | `False` | return the full `EvolutionResult` instead of the model |
| `**config_overrides` | — | — | any `EvolutionConfig` field below |

### `TrainingData`

| Field | Meaning |
|---|---|
| `X_train`, `y_train` | used to train every candidate |
| `X_val`, `y_val` | used to score and **select** candidates |
| `task` | `"classification"` (default) or `"regression"` |
| `X_test`, `y_test` (optional) | scored once, on the winner only → `result.test_score` |

### `EvolutionConfig`

| Field | Type | Default | Original name | Meaning |
|---|---|---|---|---|
| `min_layers` / `max_layers` | `int` | 1 / 6 | `MIN_LAYER` / `MAX_LAYER` | depth range |
| `min_neurons` / `max_neurons` | `int` | 1 / 64 | (fixed 1) / `MAX_NEURONS` | width range, every layer |
| `population_size` | `int` | 20 | `POPULATION` | any integer ≥ 2 |
| `n_generations` | `int` | 20 | `GENERATIONS` | hard cap on generations |
| `epochs_per_candidate` | `int` | 20 | `STEPS`* | training epochs per candidate |
| `final_training_epochs` | `int \| None` | `None` | — | retrain the winner from scratch with this budget |
| `target_fitness` | `float \| None` | `None` | `TARGET_SCORE` | stop when reached (metric's own units) |
| `stagnation_generations` | `int \| None` | 10 | `NO_EVOLVE_GENS` | stop after this many generations without improvement |
| `stagnation_min_delta` | `float` | 1e-4 | — | smallest change that counts as improvement |
| `direction` | `"maximize" \| "minimize"` | `"maximize"` | — | `"minimize"` for loss-like metrics |
| `selection_strategy` | `"tournament" \| "roulette"` | `"tournament"` | (roulette) | parent selection; roulette is scale-sensitive |
| `tournament_size` | `int` | 3 | — | contestants per tournament |
| `elitism_count` | `int` | 1 | — | top-k carried over unchanged, never re-trained |
| `crossover_strategy` | `"single_point" \| "uniform" \| "none"` | `"single_point"` | — | how two parents are combined |
| `p_crossover` | `float` | 0.8 | — | probability a parent pair is crossed over |
| `mutation_rate` | `float` | 0.2 | `MUTATION_CHANCE` | per-individual chance of one mutation |
| `neuron_mutation_step` | `int` | 4 | (always 1) | max width change per tweak |
| `mutation_boundary_mode` | `"clamp" \| "wrap"` | `"clamp"` | (wrap) | `"wrap"` reproduces the original's wrap-around |
| `mutation_operator_weights` | `dict` | tweak .7 / add .15 / remove .15 | — | relative operator frequencies |
| `cache_evaluations` | `bool` | `True` | — | reuse scores of repeated architectures (noisy backends keep one draw) |
| `on_evaluation_error` | `"penalize" \| "raise" \| "skip"` | `"penalize"` | — | what to do when a candidate fails — see below |
| `random_state` | `int \| None` | `None` | — | seeds the GA; seed your backend too for full reproducibility |

\* `STEPS` were estimator gradient *steps*, not epochs — don't copy the number.

### `EvolutionResult` (with `return_result=True`)

| Attribute | Meaning |
|---|---|
| `best_model` | trained winning model (same framework as the backend) |
| `best_architecture` | winning `Architecture`, e.g. `Architecture([16, 3])` |
| `best_fitness` | winner's **validation** score — optimistically biased |
| `test_score` | winner's held-out test score, if a test split was given — report this one |
| `history` | `RunHistory`: per-generation best / mean / worst / best-so-far / trainings / cache hits / failures, `stop_reason`; `.to_json()`, `.to_csv()`, `.summary_text()` |
| `final_population` | last generation's individuals |

## The original thesis experiment

This section records what the 2021 code did and what the thesis reports, so the
rewrite can be compared against it. Sources: thesis chapter 5,
`geneticAlgorithm.py`, `Genetic_algorithm_for_DNN.ipynb`, and the review in
[`docs/CODE_REVIEW.md`](docs/CODE_REVIEW.md).

### Original algorithm (thesis §5.1)

1. **Genome** — a flat list of layer widths, e.g. `[3, 10, 1, 9, 4]`; once scored
   it is stored as `[[3, 10, 1, 9, 4], 0.552]` (architecture, fitness).
2. **Initial population** — random depth in `[MIN_LAYER, MAX_LAYER]`, random
   width in `[1, MAX_NEURONS]` per layer.
3. **Fitness** — test accuracy of a `tf.estimator.DNNClassifier` with those
   hidden units, trained on Iris for `STEPS` steps (5000 in the notebook,
   batch size 256). A
   toy arithmetic formula was used to test the GA without training.
4. **Selection** — roulette wheel: each individual's fitness divided by the sum
   of all fitnesses gives its probability of entering the parent pool.
5. **Crossover** — parents are paired via a shuffled index; a cut point is drawn
   up to *(length of the shorter parent − 1)*, so parents of different depth can
   mate: `[1, 2 | 3]` × `[6, 7 | 8, 9, 10, 11]` → `[1, 2, 8, 9, 10, 11]` and
   `[6, 7, 3]`.
6. **Mutation** — with `MUTATION_CHANCE`, a random layer's width changes by
   ±1 (equal odds); a mutated parent's stored fitness is discarded.
7. **Stopping** — `TARGET_SCORE` reached, `GENERATIONS` exhausted, or no change
   for `NO_EVOLVE_GENS` generations.

### Parameters used

| Variable | Thesis Iris run (Fig. 26, Table 3) | Thesis run 2 (Fig. 27) | Thesis run 3 (text) | `geneticAlgorithm.py` defaults (toy cost) | Notebook defaults |
|---|---|---|---|---|---|
| `MIN_LAYER` / `MAX_LAYER` | 4 / 6 | 2 / 4 | 4 / 8 | 2 / 12 | 2 / 3 |
| `MAX_NEURONS` | 8 | 8 | 20 | 128 | 5 |
| `POPULATION` | 80 | 32 | 120 | 76 | 8 |
| `GENERATIONS` | 30 | 30 | 30 | 10 | 3 |
| `TARGET_SCORE` | 0.995 | 0.995 | 0.995 | 500000 | 0.5 |
| `MUTATION_CHANCE` | 0.05 | 0.05 | 0.05 | 0.05 | 0.01 |
| `NO_EVOLVE_GENS` | 6 | 6 | 6 | 25 | 3 |
| `STEPS` (training) | not stated¹ | not stated¹ | not stated¹ | — | 5000 |

Runs 2 and 3 are described as changes to the first run ("layers changed to
(2, 4) and population reduced to 32"; "layers (4, 8), max neurons 20,
population 120"); their other values are assumed unchanged. `POPULATION` had to
be divisible by 4 (see the review below). ¹ The thesis names the `STEPS`
variable but not its value; the notebook hard-codes 5000 training steps.

**Data.** Iris: 150 samples (50 each of *Iris setosa*, *versicolor*,
*virginica*), 4 features (sepal and petal length and width), 3 classes. The
notebook uses TensorFlow's Iris CSVs; every reported accuracy is a multiple of
1/30, consistent with the 30-row `iris_test.csv` (120 rows for training).

### Results reported in the thesis (§5.3)

**GA on toy cost functions** (no training; Figs. 23–25). Several independent
runs per function all climbed, some quickly, some slowly; plateaus lasting until
a lucky mutation were common.

| Cost function *c(i)* of the layer list *i* | Layers | Max neurons |
|---|---|---|
| `i[0]·i[1]·i[2]·i[3] / (i[4]·i[5]·100)` | 6 | 10 |
| `i[1]·i[2]·i[3] / 1000` | 3 | 10 |
| `(i[0]·i[2]·i[4]·i[6] − i[1]·i[3]·i[5]) / (i[7]·i[3]·i[0])` | 8 | 50 |

**GA on Iris, run 1** (Fig. 26). Best accuracy stopped improving at
generation 16 at **96.7 %**; the stagnation rule ended the run six generations
later. The five best architectures in the last generation (Table 3, as printed —
the first two rows are identical in the original):

| # | Architecture | Best accuracy | Copies in population (of 80) | Mean accuracy |
|---|---|---|---|---|
| 1 | `[6, 6, 5, 6]` | 0.9667 | 8 | 0.600 |
| 2 | `[6, 6, 5, 6]` | 0.9667 | 8 | 0.600 |
| 2 | `[6, 6, 6, 6]` | 0.9333 | 8 | 0.587 |
| 3 | `[6, 4, 8, 6, 5]` | 0.9000 | 3 | 0.767 |
| 4 | `[6, 3, 5, 6]` | 0.9000 | 1 | 0.900 |

The same architecture `[6, 6, 5, 6]` scored 96.7 % at best, **60 % on average
and 23 % at worst** — the thesis notes this weakens crossover and mutation. The
two top architectures made up 20 % of the population.

**Run 2** (Fig. 27, population 32, 2–4 layers). Accuracy reached 97 % at
generation 6, then dropped to 73 % (the original had no elitism), and the target
was reached only at generation 19.

**Run 3** (population 120, 4–8 layers, ≤ 20 neurons). The target (100 %) was
reached in generation 3, with more than half of the networks above 95 %. The
thesis attributes this to Iris being easy: larger networks almost always do
well.

### What these results can and cannot show

From [`docs/CODE_REVIEW.md` §3](docs/CODE_REVIEW.md#3-method-what-the-iris-experiment-can-and-cannot-show):

* **Selection on the test set.** Fitness was test accuracy, so the reported
  best is the maximum over many candidates on the rows it is reported on —
  optimistically biased. `dnnga` selects on a validation split and scores the
  test split once, on the winner only.
* **Resolution.** One test row = 3.3 percentage points; `TARGET_SCORE = 0.995`
  is reachable only by 30/30.
* **Noise ≥ signal.** Run-to-run variance of one architecture (23–97 %) is far
  larger than the differences between architectures.
* **No baseline.** No comparison with a hand-picked network or with random
  search on the same budget (in `dnnga`: `population_size=N, n_generations=1`).
* **Claims not measured by the experiment:** speed-up from parallel computation
  (evaluation was sequential and untimed, and the GA wraps backpropagation
  rather than competing with it); accuracy on "images" (Iris is four numeric
  features); the influence of crossover (no experiment varies it).

## What the code review found

[`docs/CODE_REVIEW.md`](docs/CODE_REVIEW.md) lists every problem found in the
original code, with reproductions. Summary, and where `dnnga` handles each:

| # | Finding in the original | Kind | In `dnnga` |
|---|---|---|---|
| 1.1 | Roulette appends the *same list object* several times; in-place mutation then corrupts other individuals' genomes while they keep their old fitness (~10 of 76 slots aliased per generation) | Bug (measured) | Immutable `Architecture`; operators return new objects |
| 1.2 | The notebook's last line is a syntax error, so it cannot run as shipped | Bug | Library + runnable `examples/` |
| 1.3 | Importing the module prints "Please correct POPULATION or MIN_LAYER" | Minor | No import-time side effects |
| 1.4 | Population must be divisible by 4, and `POPULATION = 4` still crashes (the summary reads 5 individuals) | Bug | Any `population_size ≥ 2` |
| 1.5 | Roulette edge cases: all-zero fitness divides by zero, negative fitness gives invalid probabilities | Minor | `random.choices` with shifting; clear error if all failed |
| 1.6 | Mutation wraps around at the edges, so a ±1 nudge jumps across the whole range (128 → 1) | Design | `clamp` by default, `wrap` optional |
| 1.7 | Dead code (unused `sort()`, noise arrays never applied, an unrunnable TF1 TensorBoard cell) | Minor | Not carried over |
| 2.1 | Depth can never change: no operator creates a new depth (0 new depths in 2,000 generations) | Design (measured) | `add_layer` / `remove_layer` mutations |
| 2.2 | No elitism, so the best can be lost (Fig. 26 drops 0.933 → 0.733, Fig. 27 drops 0.967 → 0.733), and the best network is never returned | Design | `elitism_count`, `EvolutionResult` |
| 2.3 | Stagnation compares floats for exact equality: it fired in the thesis run only because Iris accuracy is a multiple of 1/30, and cannot fire for finer metrics or credit many small gains | Design | Improvement over the last credited best, with `stagnation_min_delta` |
| 2.4 | `tf.estimator.DNNClassifier` is deprecated | Design | Keras 3 and scikit-learn backends |
| 2.5 | Hard-wired and wasteful (re-trains identical networks) | Design | Cache, elites not retrained, final retrain, model cleanup |
| 3.1 | Selection on the test set | Method | Validation for selection, test scored once |

To check that the tests are not vacuous, 15 realistic bugs were injected one at
a time into a copy of the library; all 15 were caught. Mutation testing now
measures this continuously (see [Development](#development)).

## Error handling and logging

* `EvolutionConfig` validates itself and **reports every problem at once**
  (`ConfigError`); it is re-validated at `run()`.
* Seeds outside the search bounds, bad data shapes and unknown keyword arguments
  produce clear errors before any compute is spent.
* A candidate that raises (or returns NaN/inf) is handled per
  `on_evaluation_error`: `"penalize"` logs a warning and gives it the worst
  fitness so the run continues; `"raise"` stops with `EvaluationError`;
  `"skip"` drops it. If *every* candidate fails, the run raises instead of
  returning garbage. `KeyboardInterrupt` is never swallowed. A `train()` that
  forgets to return the model is diagnosed by name.
* Logging uses the standard `logging` module under the name `"dnnga"`; the
  library is silent until you call
  `configure_logging(level=logging.INFO, filepath="run.log")` (`DEBUG` adds
  per-candidate lines).
* `result.history` is structured data (per-generation best / mean / worst /
  best-so-far / trainings / cache hits / failures) with `.to_json()`,
  `.to_csv()` and `.summary_text()`.

## Writing a backend

Subclass `dnnga.backends.ModelBackend` and implement `build`, `train`
(**return the trained model** — Keras' `fit` returns a `History`) and
`evaluate` (finite float on `data.X_val`). `extract_architecture` (to accept
model objects as seeds) and `cleanup` are optional. See `sklearn_backend.py` /
`keras_backend.py`.

An illustrative **PyTorch sketch — not shipped and not tested here**:

```python
class TorchMLPBackend(ModelBackend):
    def __init__(self, lr=1e-2, batch_size=32):
        self.lr, self.bs = lr, batch_size

    def build(self, arch, data):
        layers, prev = [], data.n_features
        for w in arch.layer_sizes:
            layers += [nn.Linear(prev, w), nn.ReLU()]
            prev = w
        return nn.Sequential(*layers, nn.Linear(prev, data.n_outputs))

    def train(self, model, data, epochs):
        X = torch.as_tensor(data.X_train, dtype=torch.float32)
        y = torch.as_tensor(np.searchsorted(np.unique(data.y_train), data.y_train))
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss = nn.CrossEntropyLoss()
        for _ in range(epochs):
            for idx in torch.randperm(len(X)).split(self.bs):
                opt.zero_grad()
                loss(model(X[idx]), y[idx]).backward()
                opt.step()
        return model

    def evaluate(self, model, data):
        with torch.no_grad():
            out = model(torch.as_tensor(data.X_val, dtype=torch.float32))
        return float((np.unique(data.y_train)[out.argmax(1).numpy()] == data.y_val).mean())
```

## Using it honestly

* **Report `result.test_score`, not `best_fitness`.** The latter is the maximum
  of many validation scores and is biased upward. Keep validation sets big
  enough that one row is not several percentage points.
* **Noise is real.** The same architecture can score very differently run to
  run (thesis Table 3: 23 %–97 %). The cache keeps one draw per architecture;
  for decisions that matter, repeat the search with several `random_state`s.
* **Compare against random search with the same budget:**
  `EvolutionConfig(population_size=N, n_generations=1)` evaluates exactly `N`
  random architectures. If the GA cannot beat that, the structure search is not
  buying you anything on your task.
* Small `epochs_per_candidate` ranks architectures by *how fast they learn*,
  which favours small/shallow ones; use `final_training_epochs` and sanity-check
  the winner with a longer run.

## Limitations (not implemented)

* **Dense stacks only.** The genotype is hidden-layer widths. No
  activation/optimizer/learning-rate/dropout genes, no conv/recurrent/attention
  blocks; `KerasMLPBackend` refuses models containing such layers rather than
  flattening them, and warns that Dropout/BatchNorm are not carried over.
* **No weight inheritance** — every candidate trains from scratch
  (function-preserving "net2net" growth/shrinking would cut cost a lot; a
  bigger project).
* **Sequential evaluation.** No parallelism. (The thesis's "parallel
  computation" claim is not something this code does either. On GPUs, naive
  multiprocessing is usually the wrong tool anyway.)
* **Single objective** (no accuracy-vs-size/latency trade-off), and **no
  PyTorch backend** (sketch above).

## Repository layout

```text
.
├── src/dnnga/                  the library
│   ├── architecture.py         genotype (immutable tuple of layer widths)
│   ├── config.py               EvolutionConfig + validation, relative bounds
│   ├── operators.py            selection, crossover, mutation
│   ├── engine.py               the GA loop, evolve_architecture, EvolutionResult
│   ├── history.py              RunHistory (per-generation records, JSON/CSV)
│   ├── data.py                 TrainingData (train / validation / test)
│   ├── individual.py           Individual (architecture, fitness, model)
│   ├── logging_utils.py        configure_logging
│   ├── exceptions.py           DnnGaError hierarchy
│   ├── cli.py, __main__.py     `dnnga` console script / python -m dnnga
│   └── backends/               base contract, scikit-learn MLP, Keras MLP
├── tests/                      pytest suite (FakeBackend for engine logic; real training for backends)
├── examples/                   iris_example.py, target_accuracy_example.py, keras_example.py
├── docs/
│   ├── CODE_REVIEW.md          review of the original code and experiment
│   └── *.pdf                   the thesis (Polish)
├── reference/original_fork/    unmodified original files, kept for provenance
├── geneticAlgorithm.py         original 2021 GA script (toy cost function)
├── Genetic_algorithm_for_DNN.ipynb   original 2021 Colab notebook (Iris, TensorFlow)
└── .github/workflows/          ci.yml (tests, Keras, wheel), mutation.yml (weekly mutmut)
```

**Provenance.** `reference/original_fork/` holds unmodified copies of the
original code: `geneticAlgorithm_fork.py`, the fork's notebook
(`Genetic_algorithm_for_DNN_fork_original.ipynb`, whose last line is a syntax
error) and a separately patched copy (`Genetic_algorithm_for_DNN_wojtek_copy.ipynb`:
syntax fix, file logging, a `STEPS` variable). Nothing in `src/` or `tests/`
imports them; they exist so every finding in `docs/CODE_REVIEW.md` can be traced
to the exact code it was found in.

## Development

```bash
pip install -e ".[sklearn,dev]"
pytest                                  # Keras tests auto-skip without TensorFlow
pytest --cov=dnnga --cov-report=term-missing
python examples/iris_example.py         # also: target_accuracy_example.py, keras_example.py
```

**CI** (`.github/workflows/ci.yml`) runs on every pull request and on pushes to
`master`:

| Job | What it checks |
|---|---|
| tests (Python 3.9–3.13) | full suite with scikit-learn; examples on 3.12 |
| tests (Python 3.12, Keras/TensorFlow) | full suite with TensorFlow, coverage gate 95 %, Keras example |
| build and test wheel | sdist + wheel build, `twine check`, suite against the installed wheel |

**Mutation testing** (`.github/workflows/mutation.yml`, weekly and on demand)
runs [mutmut](https://github.com/boxed/mutmut) over `src/dnnga` (Keras backend
excluded): it makes small changes to the code and checks that some test fails.
The job fails below a 75 % score; the baseline was 77.1 % (1269 of 1645 mutants
killed). Locally:

```bash
pip install "mutmut==3.8.1"
mutmut run && mutmut results            # list surviving mutants
mutmut show <mutant-name>               # see one change no test caught
```

## Project status and roadmap

`dnnga` 0.2.0 lives on the `dnnga-library` branch (PR #2) and has not been
released. Planned work is tracked in GitHub issues:

| Area | Issues |
|---|---|
| Release cycle (license, versioning, PyPI, signed assets) | #14 and sub-issues #16–#26 |
| Test quality | #3 (mutation-testing gaps), #9 (reproducibility) |
| Error handling | #15 |
| CI coverage | #4 (minimum dependencies), #5 (no extras), #6 (ruff/mypy), #7 (Python 3.14, Windows/macOS), #8 (README examples), #10 (security scans), #11 (Dependabot), #12 (upstream canary), #13 (timeouts) |

Issues and PRs carry labels on five axes:

| Prefix | Values | Meaning |
|---|---|---|
| `type:` | `bug`, `feature`, `test`, `ci`, `docs`, `release`, `security`, `chore` | what kind of work it is |
| `area:` | `engine`, `config`, `operators`, `backends`, `data`, `history`, `cli`, `packaging`, `ci`, `docs`, `tests` | which part of the repo it touches |
| `priority:` | `P0` (blocker) … `P3` (nice to have) | order of work |
| `status:` | `needs decision`, `blocked`, `ready`, `in progress` | whether it can be picked up |
| `effort:` | `S` (< 1 day), `M` (days), `L` (week+) | rough size |

`epic` marks an umbrella issue tracked through sub-issues (#14);
`good first issue` marks small, self-contained tasks.

## Authors, citation and license

**Thesis and original code:** Adam Bielewicz, Wojciech Woźniak (2021).

To cite the thesis:

```text
Bielewicz A., Woźniak W. (2021). Optymalizacja sieci neuronowej za pomocą
algorytmów genetycznych [Optimising a neural network with genetic algorithms].
Engineering thesis, WSB University in Gdańsk, Faculty of Economics and
Management in Gdynia. Supervisor: dr Ireneusz Miciuła.
```

**License: not chosen yet** (#16). The upstream repository
(`Francja/geneticAlgorithm`) has none; until a license is added, no rights to
reuse the code are granted. Check with the upstream author before publishing
derived work.
