import math
import random
import numpy as np
from Genome import Genome
import copy
from Fitness_function import Fitness_function_type


class NSGA_II:
    # A comprehensive NSGA-II paper: https://web.njit.edu/~horacio/Math451H/download/Seshadri_NSGA-II.pdf

    def __init__(self,
                 genome,
                 fitness_functions,
                 population_size=80,
                 offspring_size=20,
                 num_generations=10,
                 num_solutions_tournament=3,
                 recombination_probability=0.8,
                 mutation_probability_global=0.05,
                 data_train=None,
                 validate=False,
                 data_val=None,
                 on_generation_finish_callback=None
                 ):
        self.__genome = genome
        self.__fitness_functions = fitness_functions
        self.__population_size = population_size
        self.__offspring_size = offspring_size
        self.__num_generations = num_generations
        self.__num_solutions_tournament = num_solutions_tournament
        self.__recombination_probability = recombination_probability
        self.__mutation_probability_global = mutation_probability_global
        self.__data_train = data_train
        self.__validate = validate
        self.__data_val = data_val
        self.__on_generation_finish_callback = on_generation_finish_callback

        self.__genome.set_mutation_probabilities(self.__mutation_probability_global)

    def optimize(self):
        generation_number = 1
        population = self.__generate_random_population()

        non_dominated_sorted_population = self.__perform_non_dominated_sort(population)

        for i, _ in enumerate(non_dominated_sorted_population):
            self.__calculate_crowding_distance(non_dominated_sorted_population[i])

        while True:
            if generation_number > self.__num_generations:
                return non_dominated_sorted_population

            # Generate offspring
            offspring = self.__generate_offspring(population)
            population += offspring

            non_dominated_sorted_population = self.__perform_non_dominated_sort(population)

            for i, _ in enumerate(non_dominated_sorted_population):
                self.__calculate_crowding_distance(non_dominated_sorted_population[i])

            non_dominated_sorted_population = self.__choose_next_generation(non_dominated_sorted_population)
            self.__calculate_crowding_distance(non_dominated_sorted_population[-1])

            population = [solution for pareto_front in non_dominated_sorted_population for solution in pareto_front]

            if self.__on_generation_finish_callback is not None:
                self.__on_generation_finish_callback(generation_number, self.__num_generations, non_dominated_sorted_population)

            generation_number += 1

    def __evaluate_solution(self, solution, data):
        ff_values = []
        for fitness_function in self.__fitness_functions:
            ff_values.append(fitness_function.function(solution, data))
        return ff_values

    def __generate_random_population(self):
        population = []
        for _ in range(self.__population_size):
            solution = copy.deepcopy(self.__genome)
            solution.randomize()
            solution.fitness_values_train = self.__evaluate_solution(solution, self.__data_train)
            if self.__validate:
                solution.fitness_values_val = self.__evaluate_solution(solution, self.__data_val)
            population.append(solution)

        return population

    def __perform_non_dominated_sort(self, population):
        # `list_of_dominated_indices[n]` will store indices of solutions `population[n]` dominates over.
        list_of_dominated_indices = [[] for _ in population]

        # `domination_count[n]` will store how many solutions dominate over `population[n]`.
        domination_count = np.zeros(len(population))

        pareto_fronts = [[]]

        for i, _ in enumerate(population):
            for j, _ in enumerate(population):

                # We don't want to compare a solution with itself.
                if i == j:
                    continue

                # A positive number in `fitness_diff` indicates superiority of population[i], while a negative number
                # indicates superiority of population [j].
                fitness_diff = []

                for k, _ in enumerate(self.__fitness_functions):
                    if self.__fitness_functions[k].type == Fitness_function_type.MIN:
                        # We want to minimize this FF, therefore the subtraction should return a positive number when
                        # population[i] has a lower FF value.
                        fitness_diff.append(population[j].fitness_values_train[k] - population[i].fitness_values_train[k])
                    elif self.__fitness_functions[k].type == Fitness_function_type.MAX:
                        # We want to maximize this FF, therefore the subtraction should return a positive number when
                        # population[i] has a higher FF value.
                        fitness_diff.append(population[i].fitness_values_train[k] - population[j].fitness_values_train[k])

                # Check if one solutions dominates over the other, or if they are equal.
                difference = np.sign(fitness_diff)

                plus_present = False
                minus_present = False

                if 1 in difference:
                    plus_present = True
                if -1 in difference:
                    minus_present = True

                if plus_present and not minus_present:
                    # In this case, population[i] dominates over population[j].
                    list_of_dominated_indices[i].append(j)
                elif not plus_present and minus_present:
                    # In this case, population[j] dominates over population[i].
                    domination_count[i] += 1
                # else:
                #     # The only remaining case is that population[i] and population[j]
                #     # are equivalent, so we do nothing.

            if domination_count[i] == 0:
                # Solution population[i] is not dominated by any other solution,
                # therefore it belongs to the first (best) pareto front.
                population[i].rank = 0
                pareto_fronts[0].append(i)

        i = 0
        # Iterate until each solution is assigned to a pareto front.
        while len(pareto_fronts[i]) > 0:
            # A list where solutions that belong to the next pareto front
            # will be saved.
            next_pareto_front = []

            # Iterate over solutions on the last pareto front.
            for j in pareto_fronts[i]:
                for k in list_of_dominated_indices[j]:
                    # Reduce domination count for the solutions that are dominated
                    # by the individuals on the current pareto front.
                    domination_count[k] -= 1

                    # If the solution is no longer dominated, that is, all the
                    # solutions that dominated over the current solution were
                    # deployed to pareto fronts, add current solution to the
                    # next pareto front.
                    if domination_count[k] == 0:
                        population[k].rank = i + 1
                        next_pareto_front.append(k)

            # Jump to next pareto front.
            i += 1

            # Add current pareto front to the list of all pareto fronts.
            pareto_fronts.append(next_pareto_front)

        # Last pareto front is empty (check 'while' condition above), so
        # we remove it.
        del pareto_fronts[-1]

        # Turn pareto front indices into objects; Replace index with the
        # corresponding object in `population`.

        object_pareto_fronts = []

        for pareto_front in pareto_fronts:
            current_front = []
            for index in pareto_front:
                current_front.append(population[index])
            object_pareto_fronts.append(current_front)

        return object_pareto_fronts

    def __calculate_crowding_distance(self, pareto_front):
        # Sort solutions on the pareto front according to ff_path_length in
        # ascending order.

        for k, _ in enumerate(self.__fitness_functions):
            sorted_pareto_front = sorted(
                pareto_front,
                key=lambda solution: solution.fitness_values_train[k]
            )

            sorted_pareto_front[0].crowding_distance = math.inf
            sorted_pareto_front[-1].crowding_distance = math.inf

            ff_range = sorted_pareto_front[-1].fitness_values_train[k] - sorted_pareto_front[0].fitness_values_train[k]

            # Later, we divide by ff_range, so we want to make sure it's not 0.
            if ff_range <= 0:
                ff_range = 1

            # Iterate over solutions on the current pareto front and calculate
            # the contribution of each fitness function to the crowding distance.
            for i in range(1, len(sorted_pareto_front) - 1):
                # Contribution of ...

                sorted_pareto_front[i].crowding_distance += (
                    (sorted_pareto_front[i + 1].fitness_values_train[k] - sorted_pareto_front[i - 1].fitness_values_train[k]) / ff_range
                )

    def __generate_offspring(self, population):
        offspring = []

        # Generate a predefined number of individuals.
        for _ in range(self.__offspring_size):
            offspring.append(self.__generate_single_solution(population))

        return offspring

    def __generate_single_solution(self, population):
        # Pick two parents.
        parent_1 = self.__tournament_select_parent(population)
        child = parent_1

        if np.random.rand() < self.__recombination_probability:
            parent_2 = self.__tournament_select_parent(population)
            child = parent_1.recombination(parent_2)

        # Mutate a child, introduce slight variation.
        child.mutate()

        child.fitness_values_train = self.__evaluate_solution(child, self.__data_train)
        if self.__validate:
            child.fitness_values_val = self.__evaluate_solution(child, self.__data_val)

        return child

    def __tournament_select_parent(self, population):
        # Copy tournament size so we can decrement it later.
        num_battles = self.__num_solutions_tournament - 1

        random_parent_index = np.random.randint(0, len(population))

        while True:
            random_opponent_index = np.random.randint(0, len(population))

            # We don't want to compare a parent with itself.
            if random_parent_index == random_opponent_index:
                continue

            # Create objects from indices.
            random_parent = population[random_parent_index]
            random_opponent = population[random_opponent_index]

            # Pick a winner.
            if random_opponent.rank < random_parent.rank or \
                    (random_opponent.rank == random_parent.rank and random_opponent.crowding_distance > random_parent.crowding_distance):
                random_parent_index = random_opponent_index

            # One less battle remaining.
            num_battles -= 1

            # We have a winner, return the best parent.
            if num_battles <= 0:
                return population[random_parent_index]

    def __choose_next_generation(self, non_dominated_sorted_population):
        next_generation = []
        size = 0

        for pareto_front in non_dominated_sorted_population:
            if len(pareto_front) + size <= self.__population_size:
                # If the whole pareto front fits into next generation, add it.
                next_generation.append(pareto_front)
                size += len(pareto_front)
            else:
                # Otherwise, add the individuals with the highest crowding distance to preserve genetic diversity.
                if self.__population_size - size <= 0:
                    break

                pareto_front.sort(key=lambda solution: solution.crowding_distance)

                next_generation.append(pareto_front[-(self.__population_size - size):])
                break

        return next_generation

