#!/usr/bin/env python3


import math
import random
import numpy as np
from Genome import Genome


class NSGA_II:
    # A comprehensive NSGA-II paper: https://web.njit.edu/~horacio/Math451H/download/Seshadri_NSGA-II.pdf

    def __init__(self,
                 genome,
                 fitness_functions,
                 data_train,
                 data_val,
                 population_size=80,
                 offspring_size=20,
                 num_generations=10,
                 num_solutions_tournament=3,
                 recombination_probability=0.8,
                 mutation_probability=0.05,
                 on_generation_finish_callback=None
                 ):
        self.genome = genome
        self.fitness_functions = fitness_functions
        self.data_train = data_train
        self.data_val = data_val
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.num_generations = num_generations
        self.num_solutions_tournament = num_solutions_tournament
        self.recombination_probability = recombination_probability
        self.mutation_probability = mutation_probability
        self.on_generation_finish_callback = on_generation_finish_callback

    def calculate(self):
        generation_number = 1
        population = self.generate_random_population()

        non_dominated_sorted_population = self.perform_non_dominated_sort(population)

        for i, _ in enumerate(non_dominated_sorted_population):
            self.calculate_crowding_distance(non_dominated_sorted_population[i])

        while True:
            if generation_number > self.num_generations:
                break

            print('Generation: {}/{}'.format(generation_number, self.num_generations))

            # Generate offspring
            offspring = self.generate_offspring(population)
            population += offspring

            non_dominated_sorted_population = self.perform_non_dominated_sort(population)

            for i, _ in enumerate(non_dominated_sorted_population):
                self.calculate_crowding_distance(non_dominated_sorted_population[i])

            population = self.choose_next_generation(non_dominated_sorted_population)
            generation_number += 1

        pareto_fronts = self.perform_non_dominated_sort(population)
        return pareto_fronts

    def generate_random_population(self, cities_to_visit):
        """Generate N random individuals.

        Use self.population_size.

        Parameters
        ----------
        cities_to_visit : list
            List of strings. Cities that need to be visited.

        Returns
        -------
        List of Route objects.
        """

        population = []
        for _ in range(self.population_size):
            # IMPORTANT! Make a copy of the list. Otherwise, all the
            # solutions in the population would be the same.
            random_city_sequence = cities_to_visit[:]

            # This shuffles the list in-place.
            random.shuffle(random_city_sequence)

            solution = self.Route(random_city_sequence)

            solution.ff_path_length = self.ff_calc_path_length(solution.city_list)
            solution.ff_order = self.ff_calc_order(solution.city_list)

            population.append(solution)

        return population

    def perform_non_dominated_sort(self, population):
        """Divide the population into pareto fronts.

        Parameters
        ----------
        population : list
            List of Route objects.

        Returns
        -------
        List of lists of Route objects.
            E.g., [[Route#1, Route#2, ...], ...]
        """

        # list_of_dominated_indices[n] will store indices of solutions
        # population[n] dominates over.
        list_of_dominated_indices = [[] for _ in population]

        # domination_count[n] will store how many solutions dominate over
        # population[n].
        domination_count = np.zeros(len(population))

        pareto_fronts = [[]]

        for i, _ in enumerate(population):
            for j, _ in enumerate(population):

                if i == j:
                    continue

                # Check if one solutions dominates over the other, or they
                # are equal.
                difference = np.sign(
                                [
                                    population[j].ff_path_length - population[i].ff_path_length,
                                    population[j].ff_order - population[i].ff_order
                                ]
                            )

                plus_present = False
                minus_present = False

                for ff_diff in difference:
                    if ff_diff == 1:
                        plus_present = True
                    elif ff_diff == -1:
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
                #     pass

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
        # corresponding object in 'population'.

        object_pareto_fronts = []

        for pareto_front in pareto_fronts:
            current_front = []
            for index in pareto_front:
                current_front.append(population[index])
            object_pareto_fronts.append(current_front)

        return object_pareto_fronts

    def calculate_crowding_distance(self, pareto_front):
        """Calculate crowding distance for a single pareto front.

        Crowding distance is calculated for each pareto front separately.
        This function modifies object parameters directly and returns nothing.

        Parameters
        ----------
        pareto_front : list
            List of Route objects.
        """

        # Sort solutions on the pareto front according to ff_path_length in
        # ascending order.
        sorted_front_ff_path_length = sorted(
            pareto_front,
            key=lambda solution: solution.ff_path_length
        )

        # Sort solutions on the pareto front according to ff_order in
        # ascending order.
        sorted_front_ff_order = sorted(
            pareto_front,
            key=lambda solution: solution.ff_order
        )

        # First and last solution in the sorted arrays have infinite
        # crowding distance because they only have one neighbour.
        sorted_front_ff_path_length[0].distance = math.inf
        sorted_front_ff_path_length[-1].distance = math.inf

        sorted_front_ff_order[0].distance = math.inf
        sorted_front_ff_order[-1].distance = math.inf

        # Calculate maximum distance for each fitness function separately.
        max_ff_path_length = sorted_front_ff_path_length[-1].ff_path_length - sorted_front_ff_path_length[0].ff_path_length
        max_ff_order = sorted_front_ff_order[-1].ff_order - sorted_front_ff_order[0].ff_order

        # Later, we divide by max_ff values, so we want to make sure they
        # are not 0.
        if max_ff_path_length <= 0:
            max_ff_path_length = 1

        if max_ff_order <= 0:
            max_ff_order = 1

        # Iterate over solutions on the current pareto front and calculate
        # the contribution of each fitness function to the crowding distance.
        for i in range(1, len(pareto_front) - 1):
            # Contribution of ff_path_length
            sorted_front_ff_path_length[i].distance += \
            (sorted_front_ff_path_length[i+1].ff_path_length - sorted_front_ff_path_length[i-1].ff_path_length) / max_ff_path_length
            # Contribution of ff_order
            sorted_front_ff_order[i].distance += (sorted_front_ff_order[i+1].ff_order - sorted_front_ff_order[i-1].ff_order) / max_ff_order

    def generate_offspring(self, population):
        """Generate offspring.

        Use self.offspring_size.

        Parameters
        ----------
        population : list
            List of Route objects.

        Returns
        -------
        List of Route objects.
            E.g., [Route#1, Route#2, ...]
        """

        offspring = []

        # Generate a predefined number of individuals.
        for _ in range(self.offspring_size):
            offspring.append(self.generate_single_solution(population))

        return offspring

    def generate_single_solution(self, population):
        """Generate a single child.

        Parameters
        ----------
        population : list
            List of Route objects.

        Returns
        -------
        Route object.
        """
        # Pick two parents.
        first_parent = self.tournament_select_parent(population)
        second_parent = self.tournament_select_parent(population)

        # Select where to merge chromosomes.
        recombination_index = np.random.randint(0, len(first_parent.city_list))

        # Take cities from first_parent up until recombination_index.
        first_part = first_parent.city_list[:recombination_index]

        # Fill the child with remaining cities from the second_parent.
        second_part = []
        for i in range(len(second_parent.city_list)):
            if second_parent.city_list[i] not in first_part:
                second_part.append(second_parent.city_list[i])

        child_city_list = first_part + second_part
        child = self.Route(child_city_list)

        # Mutate a child, introduce slight variation.
        self.mutate(child)

        child.ff_path_length = self.ff_calc_path_length(child.city_list)
        child.ff_order = self.ff_calc_order(child.city_list)

        return child

    def tournament_select_parent(self, population):
        """Select one parent by tournament selection.

        Use self.num_solutions_tournament.

        Parameters
        ----------
        population : list
            List of Route objects.

        Returns
        -------
        Route object.
        """

        # Select a random parent.
        random_parent = population[np.random.randint(0, len(population))]

        for i in range(self.num_solutions_tournament-1):
            # Select random opponent.
            random_opponent = population[np.random.randint(0, len(population))]

            # Pick a winner.
            if random_opponent.rank < random_parent.rank or \
                (random_opponent.rank == random_parent.rank and random_opponent.distance > random_parent.distance):
                random_parent = random_opponent

        return random_parent

    def mutate(self, child):
        """Mutate a child.

        This function modifies child object in-place and returns nothing.

        Parameters
        ----------
        child : Route object.
        """

        # Select first city for swap.
        first_random_city = np.random.randint(0, len(child.city_list))

        while True:
            # Select second city for swap. The city must be different from
            # the first one.
            second_random_city = np.random.randint(0, len(child.city_list))

            if first_random_city != second_random_city:
                break

        # Swap the cities.
        temp = child.city_list[first_random_city]
        child.city_list[first_random_city] = child.city_list[second_random_city]
        child.city_list[second_random_city] = temp

    def selection(self, non_dominated_sorted_population):
        """Select individuals for the next generation.

        Use self.population_size.

        Parameters
        ----------
        non_dominated_sorted_population : List of lists of Route objects.
            E.g., [[Route#1, Route#2, ...], ...]

        Returns
        -------
        List of Route objects.
            E.g., [Route#1, Route#2, ...]
        """

        next_generation = []

        for pareto_front in non_dominated_sorted_population:
            if len(pareto_front) + len(next_generation) <= self.population_size:
                # If the whole pareto front fits into next generation, add it.
                next_generation.extend(pareto_front)
            else:
                # Otherwise, add the individuals with the highest crowding distance
                # to preserve genetic diversity.
                pareto_front.sort(key=lambda solution: solution.distance)
                next_generation.extend(
                    pareto_front[-(self.population_size-len(next_generation)):]
                )
                break

        return next_generation
