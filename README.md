# Genetic algorithm for neural network architecture search

A genetic algorithm that searches for the hidden-layer layout of a neural network
(how many layers, how many neurons in each) instead of choosing it by hand.
Written in 2021 as part of my thesis: <!-- TODO: thesis title, university, degree -->.

## How it works

Each individual is a list of integers, one per hidden layer, for example `[4, 1, 3]` =
three hidden layers with 4, 1 and 3 neurons. Its fitness is the test accuracy of a
network built with that layout.

1. **Initial population:** random individuals with `MIN_LAYER`–`MAX_LAYER` layers and
   1–`MAX_NEURONS` neurons per layer.
2. **Selection:** roulette wheel (fitness-proportional); half of the population survives.
3. **Crossover:** random pairs of parents, single cut point; each pair gives two
   children, so the population is back to full size. Parents can have different
   lengths, so children can gain or lose layers.
4. **Mutation:** with probability `MUTATION_CHANCE` one layer of an individual
   gets +1 or −1 neuron (wrapping around at the limits).
5. **Evaluation:** only new and mutated individuals are scored again, because
   training a network is the expensive step.
6. **Stop:** target score reached, no improvement for `NO_EVOLVE_GENS` generations,
   or `GENERATIONS` used up. The five best layouts of each generation are printed.

## Files

| File | What it is |
|---|---|
| `geneticAlgorithm.py` | The algorithm on its own, with a cheap placeholder cost function: for testing the GA mechanics without training anything |
| `Genetic_algorithm_for_DNN.ipynb` | The same algorithm with a real fitness: a TensorFlow `DNNClassifier` trained on the Iris dataset, test accuracy as the score |

## Running

```bash
python geneticAlgorithm.py    # GA only, standard library, runs in seconds
```

The notebook was written for Google Colab in 2021 (TensorFlow 2.x with the Estimator
API). The Estimator API has since been removed from TensorFlow, so the notebook does not
run on a current Colab runtime as is; it would need a port to Keras
(`tf.keras.Sequential` with `Dense` layers built from the individual).

## Results and limitations

This is a proof of concept, not a tuned search. The saved notebook run uses a tiny
configuration (population 8, up to 3 layers of up to 5 neurons, 3 generations,
target accuracy 0.5) so that it finishes quickly. The best layout found was
`[4, 1, 3]` with 0.57 test accuracy; a well-sized network reaches far higher on Iris,
so the run demonstrates the search loop rather than a good architecture.

What I would change today:
- elitism (always keep the best individual), so a good layout cannot be lost by chance;
- train each network more than once (or with a fixed seed), since accuracy from a
  single short training run is noisy and the GA ends up optimising noise;
- a Keras port, a validation split for the fitness and the test set only for the final model;
- separate the GA from the fitness function so the two files share one implementation.

## Credits

<!-- TODO: co-author / contributors -->
