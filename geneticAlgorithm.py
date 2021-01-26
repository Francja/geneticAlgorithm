from random import random
from random import randint
from random import shuffle


# Length of table, how many layers would neural network have
MAX_LAYER = 8
# Max value of element in table, max neurons in each layer
MAX_NEURONS = 128
# Population of individuals. !!! Needs to be dividable by 4
POPULATION = 240
# How many iterations will the algorithm go through
GENERATIONS = 50
# Target score, after it's accomplished algorithm stops
TARGET_SCORE = 1000000
# Mutation chance - it's applied to each individual in each generation
MUTATION_CHANCE = 0.05
# Number of generations without change in score after the algorithm will stop
NO_EVOLVE_GENS = 15


def generate_individual():
    new_ind = []
    for n in range(0, MAX_LAYER):
        new_ind.append(randint(1, MAX_NEURONS))
    return new_ind


def initialize():
    pop = []
    for n in range(0, POPULATION):
        pop.append(generate_individual())
    return pop


def cost(ind):
    c = (ind[0]/ind[1]/ind[2]*ind[3]*ind[4]*ind[5]/ind[6]/ind[7])
    return c


def sort(pop_with_cost):
    pop_with_cost.sort(key=lambda x: x[1])
    return pop_with_cost


def roulette(pop_with_cost):
    cost_sum = sum(n for _, n in pop_with_cost)

    cost_table = []
    for ind in pop_with_cost:
        cost_table.append(ind[1])

    rel_cost = [cost_ind / cost_sum for cost_ind in cost_table]
    probability = [sum(rel_cost[:i + 1]) for i in range(len(rel_cost))]

    new_pop = []
    for n in range(POPULATION//2):
        r = random()
        for (i, individual) in enumerate(pop_with_cost):
            if r <= probability[i]:
                new_pop.append(individual)
                break
    return new_pop


def cross_parents(parent1, parent2, cross_point):
    child = []
    child.extend(parent1[:cross_point])
    child.extend(parent2[cross_point:])
    return child


def crossover(pop):
    random_index_table = []

    for x in range(0, len(pop)):
        random_index_table.append(x)
    shuffle(random_index_table)

    for x in range(0, len(pop), 2):
        p1 = pop[random_index_table[x]][0]
        p2 = pop[random_index_table[x+1]][0]
        cross_point = randint(1, MAX_LAYER)
        pop.append(cross_parents(p1, p2, cross_point))
        pop.append(cross_parents(p2, p1, cross_point))
    return pop


def calculate_missing_cost(pop):
    pop_with_scores = []
    for ind in pop:
        if len(ind) == 2 and len(ind[0]) == MAX_LAYER:
            pop_with_scores.append(ind)
        elif len(ind) == MAX_LAYER:
            pop_with_scores.append([ind, cost(ind)])
    return pop_with_scores


def mutate(pop):
    mutated_pop = []
    for ind in pop:
        rand = random()
        if rand <= MUTATION_CHANCE:
            mutated_pop.append(generate_individual())
        else:
            mutated_pop.append(ind)
    return mutated_pop


def final_text(pop, stop):
    if stop == "success":
        print(f"SUCCESS: Target of score of {TARGET_SCORE} was accomplished.")
    elif stop == "stagnation":
        print(f"FAILED: Population stopped evolving.")
    else:
        print("FAILED: Target was not accomplished.")
    pop.sort(key=lambda x: x[1], reverse=True)
    print("Top 5 scores: ")
    for n in range(0, 5):
        print(f"#{n + 1}: {pop[n][0]}; \tscore: {round(pop[n][1], 2)}")


if __name__ == "__main__":
    stop_condition = ""
    pop_bag = []

    # creates the initial randomized population bag:
    init_pop_bag = initialize()

    # initial pop go through neural network and setting the score:
    for init_ind in init_pop_bag:
        pop_bag.append([init_ind, cost(init_ind)])

    # creates places to gather scores
    best_of_gen = max(pop_bag, key=lambda x: x[1])[1]
    max_score = best_of_gen
    score_table = [best_of_gen]

    # loop that goes through rest of generations:
    for gen in range(0, GENERATIONS):

        # checks if generations maximum was better that top score
        if max_score <= best_of_gen:
            max_score = best_of_gen

        print(f"Generation {gen + 1} ~~ best score = {round(best_of_gen, 2)}")

        # stop condition check
        if max_score >= TARGET_SCORE:
            stop_condition = "success"
            break

        # checks if the population stopped evolving
        if score_table[-NO_EVOLVE_GENS:].count(score_table[-1]) >= NO_EVOLVE_GENS:
            stop_condition = "stagnation"
            break

        # choosing new population based on roulette
        pop_bag = roulette(pop_bag)

        # crossover of randomly selected parents
        pop_bag = crossover(pop_bag)

        # mutating
        pop_bag = mutate(pop_bag)

        # calculating score for new children and mutated ones
        pop_bag = calculate_missing_cost(pop_bag)

        # getting best score of generation
        best_of_gen = max(pop_bag, key=lambda x: x[1])[1]
        score_table.append(best_of_gen)

    final_text(pop_bag, stop_condition)


