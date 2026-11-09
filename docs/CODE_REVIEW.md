# Review of the original code and thesis experiment

**Reviewed:** `geneticAlgorithm.py` and `Genetic_algorithm_for_DNN.ipynb` from
`geneticAlgorithm-master.zip`; your `Kopia_notatnika_Genetic_algorithm_for_DNN.ipynb`
(a different version of the notebook, presumably the newer one — it has Drive
logging, a `STEPS` variable, and the `final_text` bug already fixed); and
chapter 5 of the thesis. Everything below was either read directly in the code,
measured by running the **original, unmodified** `geneticAlgorithm.py`, or
(where marked *inference*) deduced from the thesis's own numbers.

Severity: **Bug** = wrong behaviour · **Design** = works, but defeats the purpose
· **Method** = limits what the experiment can show · **Minor**.

---

## 1. Bugs

### 1.1 Aliasing corrupts fitness values — **Bug (measured)**

`roulette()` appends the *same object* each time an individual is picked
(`new_pop.append(individual)`, `.py` line 63). Picking an individual several
times is how roulette selection is supposed to work, so aliases are the norm
(measured: ~10 of 76 slots per generation share an object with another slot).
`mutate()` then edits that object **in place** (`ind[0][target_cell] = new_value`,
lines 114 and 121). The slot that was *not* chosen to mutate changes too, but
keeps its old cost.

Minimal reproduction (original code, `MUTATION_CHANCE=0.5`, seed 10):

```text
one individual [10, 20, 30] (cost 250.0) picked twice -> same object in 2 slots; one slot mutates:
[[9, 20, 30], 250.0] <- keeps the OLD cost
[9, 20, 30]          <- bare list, cost cleared, will be recomputed
cost([9, 20, 30]) is 249.0, not 250.0
```

Measured over 20 runs × 100 generations with the original defaults: **61 % of
generations contained at least one individual whose stored cost did not belong
to its architecture** (mean 1.48 per generation). This is a *lower bound*:
mismatches are only counted when the cost actually differs, and the toy cost
happens to ignore some genes. With real DNN fitness the bad pairs then feed the
next roulette round and the elite/"best" statistics.

*Fix in `dnnga`:* `Architecture` is a frozen dataclass over a tuple; every
operator returns a new object.

### 1.2 The notebook in the zip cannot run — **Bug**

The last statement of the code cell is `final_text(pop_bag, text_summary))`: an
unmatched `)` (SyntaxError) and the wrong first argument (`pop_bag` instead of
`stop_condition`; the `.py` file has it right). Your newer copy fixed this.

### 1.3 Importing the module prints an error — **Minor**

`else: print("Please correct POPULATION or MIN_LAYER")` (lines 201-202) is
attached to `if __name__ == "__main__" and MIN_LAYER >= 2 and POPULATION % 4 == 0`,
so *any import* prints it.

### 1.4 `POPULATION = 4` crashes; `POPULATION % 4` is a hidden rule — **Bug**

`gather_text` reads `pop[4-n]` for n = 0..4, i.e. needs ≥ 5 individuals; the
guard allows 4. (Your notebook comment "needs to be dividable by 4. (avoid 4)"
is the workaround.) `dnnga` accepts any `population_size ≥ 2`.

### 1.5 Roulette edge cases — **Minor / theoretical**

All costs 0 → `ZeroDivisionError` (`cost_sum == 0`); negative costs give invalid
probabilities with no check. Floating-point error can leave the last cumulative
probability just under 1.0, so a draw above it selects nobody and the generation
comes up one short — but the probability is ~1e-16 per draw, so **this one is
not observable in practice**.

### 1.6 Wrap-around mutation — **Design (probably unintended)**

`new_value = 1 if new_value > MAX_NEURONS` and `MAX_NEURONS if new_value == 0`
(lines 112-113): a "±1" nudge at a boundary becomes a jump across the entire
range (128 → 1). `dnnga` clamps by default; `mutation_boundary_mode="wrap"`
restores it.

### 1.7 Dead code — **Minor**

`sort()` is never called. In your notebook the `noise` arrays ("noise for
dataset blurring") are generated in three places and never applied, so no noise
experiment is actually happening. The notebook's **second cell** is an
unfinished TensorBoard snippet using TF1 APIs (`tf.Session`,
`tf.summary.FileWriter`) and undefined names (`loss1`, `result`,
`network_output`, `input`, `input_data`); it cannot run.

---

## 2. Design flaws

### 2.1 Depth can never grow — **Design (measured)**

The only mutation nudges one width by ±1; crossover gives child 1 the *length of
parent 2* and vice versa. So crossover redistributes depths that already exist,
and no operator creates a new one. Measured: in 2,000 generations a depth absent
from the initial random population appeared **0 times**. Structural diversity
can shrink, never grow — whereas "add/remove layers" is exactly what you now
want.

*Fix:* `add_layer`/`remove_layer` mutation operators.

### 2.2 No elitism; the winner is never returned — **Design**

A generation's best individual survives only if roulette re-picks it *and* it is
not chosen to mutate (mutation also clears its cost, forcing a noisy
re-training). The thesis's own plots show it: Fig. 26 drops 0.933 → 0.733,
Fig. 27 drops 0.967 → 0.733. And nothing returns the best architecture —
`max_score` is only a number; the per-generation top-5 is printed text.

*Fix:* `elitism_count`, best-ever tracking, `EvolutionResult.best_architecture /
best_model`.

### 2.3 Stagnation test uses exact float equality — **Design**

`score_table[-NO_EVOLVE_GENS:].count(score_table[-1]) >= NO_EVOLVE_GENS`
(line 179). It *can* fire when the metric is coarse — it did in your Fig. 26
run, where accuracy is a multiple of 1/30 — but not for any finer metric (larger
validation set, loss, R²), and it cannot credit many tiny improvements that add
up.

*Fix:* improvement relative to the last credited best, with
`stagnation_min_delta` (Keras `EarlyStopping` semantics).

### 2.4 `tf.estimator.DNNClassifier` is deprecated — **Design**

TensorFlow's API docs mark every Estimator class `(deprecated)`. `dnnga` has a
`KerasMLPBackend` instead.

### 2.5 Hard-wired and wasteful — **Design**

Iris, feature columns and CSV paths are globals; in your newer notebook `DNN()`
re-reads both CSVs for every candidate; each candidate is trained from scratch
and **identical architectures are re-trained** (Table 3: eight copies of
`[6, 6, 5, 6]`, each trained separately).

*Fix:* backend interface, `cache_evaluations`, `final_training_epochs` (cheap
search, one full training of the winner).

---

## 3. Method (what the Iris experiment can and cannot show)

1. **Selection on the test set.** Fitness is
   `classifier.evaluate(... test ...)` (notebook lines 249-250). The reported
   "best accuracy" is therefore the *maximum over many candidates on the very
   rows it is reported on* — optimistically biased, with no untouched data left
   for an honest number. *Fix:* separate validation (selection) and test
   (reported once).
2. **Resolution.** Every accuracy in Fig. 26/27 and Table 3 is a multiple of
   1/30 (0.9667 = 29/30, 0.9333 = 28/30, 0.8333 = 25/30). *Inference:* the test
   file has 30 rows (TensorFlow's `iris_test.csv` does). One row = 3.3 points,
   and `TARGET_SCORE = 0.995` is reachable only by 30/30.
3. **Noise ≥ signal.** Table 3: architecture `[6, 6, 5, 6]` scored best 96.7 %,
   mean 60 %, worst 23 %. Run-to-run variance of one architecture is far larger
   than the differences between architectures the GA tries to exploit.
4. **No baseline.** No comparison with a hand-picked architecture or with random
   search using the same number of trainings; p. 54 itself notes that larger
   networks solve Iris "almost always", so the GA's contribution is not
   isolated. (In `dnnga`, equal-budget random search is `population_size=N,
   n_generations=1`.)
5. **Claims the experiment does not measure** (worth softening if the thesis is
   reused): (a) the introduction says parallel computation speeds things up
   relative to plain backpropagation — evaluation is sequential, no timing is
   reported, and the GA *wraps* backpropagation (each candidate is still trained
   by gradient descent) rather than competing with it; (b) the conclusion says
   accuracy improved "classifying images" — Iris is four numeric features;
   (c) the introduction says the influence of crossover is studied — no
   experiment varies crossover.

---

## 4. How `dnnga` addresses each point (and which test guards it)

| Finding | In `dnnga` | Guarded by |
|---|---|---|
| 1.1 aliasing | immutable `Architecture`; operators return new objects | `TestImmutability`, `test_parents_unaffected_by_editing_children` |
| 1.5 roulette edge cases | `random.choices`; shift for negatives; `-inf` weight 0; clear error if all failed | `TestRoulette` (incl. `-inf`, never-short regression) |
| 1.6 wrap-around | `clamp` default, `wrap` optional | `test_clamp_stops_at_the_edge_while_wrap_jumps_across` |
| 1.4 population rule | any `population_size ≥ 2` | `test_any_population_size_works_not_just_multiples_of_four` |
| 2.1 depth frozen | `add_layer` / `remove_layer` | `TestDepthEvolves` (first test reproduces the original limitation) |
| 2.2 no elitism / no result | `elitism_count`; `EvolutionResult` | `TestElitismAndBestEver`, `test_result_contains_best_architecture_and_trained_model` |
| 2.3 stagnation | credited-improvement + `min_delta` | `test_stagnation_credits_many_tiny_steps_that_add_up` |
| 2.5 waste | cache, final retrain, model cleanup | `TestCachingCleanupRetrain` |
| 3.1 test-set selection | validation for selection, test scored once | `test_test_split_is_scored_once_at_the_end` |

To check that these tests are not vacuous, 15 realistic bugs (no elitism,
original-style stagnation, no clamping, test score computed on validation data,
…) were injected one at a time into a copy of the library: **all 15 were
caught**.

## 5. Not fixed by any code — your decisions

Add a baseline and repeated runs (≥ 5 seeds, report mean ± std) before claiming
the GA helps; use a task where capacity matters (Iris is solved by almost any
network — see `examples/keras_example.py`); decide what to claim about speed.
