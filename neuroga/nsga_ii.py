import math
import numpy as np
import copy
import os
from multiprocessing import Process, Queue
from neuroga.fitness_function import FitnessFunctionType


class NSGAII:
    """
    K. Deb, A. Pratap, S. Agarwal and T. Meyarivan
    "A fast and elitist multiobjective genetic algorithm: NSGA-II,"
    in IEEE Transactions on Evolutionary Computation, vol. 6, no. 2, pp. 182-197, April 2002
    doi: 10.1109/4235.996017.
    """

    def __init__(self,
                 genome,
                 fitness_functions,
                 population_size=80,
                 offspring_size=20,
                 num_generations=10,
                 num_solutions_tournament=3,
                 recombination_probability=0.9,
                 mutation_probability_global=0.05,
                 data_train=None,
                 validate=False,
                 data_val=None,
                 on_generation_finish_callback=None,
                 num_processes=None
                 ):
        # A genome that defines an individual. This is used as a template for constructing individual solutions.
        self.__genome = genome

        self.__fitness_functions = fitness_functions
        self.__population_size = population_size
        self.__offspring_size = offspring_size
        self.__num_generations = num_generations
        self.__num_solutions_tournament = num_solutions_tournament
        self.__recombination_probability = recombination_probability

        # This mutation probability will be used for subgenomes that don't have a local mutation probability defined.
        self.__mutation_probability_global = mutation_probability_global

        # The data used in fitness functions to evaluate individuals and direct the algorithm. This is optional.
        self.__data_train = data_train

        # Whether validation is necessary. The algorithm is not aware of the results scored by individuals during
        # validation, therefore these results can help detect overfitting.
        self.__validate = validate

        # The data used in fitness functions for validation purposes. This is optional.
        self.__data_val = data_val

        # This function will be called at the end of each generation.
        self.__on_generation_finish_callback = on_generation_finish_callback

        # Controls the maximum number of processes to be used when creating children.
        self.__num_processes = num_processes

        self.__genome.set_mutation_probabilities(self.__mutation_probability_global)

    def optimize(self):
        """Performs the optimization and returns pareto fronts of the last generation.

        :return: List[List[Genome]]
        """
        generation_number = 1
        population = self.__generate_random_population()

        # If there is only a single evaluation function, a large speed-up can be made by
        # swapping non-dominated sort and crowding distance calculation with a simple 1D sort.
        if len(self.__fitness_functions) > 1:
            non_dominated_sorted_population = self.__perform_non_dominated_sort(population)
            for i, _ in enumerate(non_dominated_sorted_population):
                self.__calculate_crowding_distance(non_dominated_sorted_population[i])
        else:
            non_dominated_sorted_population = self.__perform_1d_sort(population)

        while True:
            if generation_number > self.__num_generations:
                return non_dominated_sorted_population

            # Generate offspring and add them to the population. This enforces elitism.
            offspring = self.__generate_offspring(population)
            population += offspring

            # If we have a single evaluation function, perform 1D sort.
            if len(self.__fitness_functions) > 1:
                non_dominated_sorted_population = self.__perform_non_dominated_sort(population)
                for i, _ in enumerate(non_dominated_sorted_population):
                    self.__calculate_crowding_distance(non_dominated_sorted_population[i])
            else:
                non_dominated_sorted_population = self.__perform_1d_sort(population)

            # Choose the best individuals for the next generation.
            non_dominated_sorted_population = self.__choose_next_generation(non_dominated_sorted_population)

            # If we have a single evaluation function, avoid crowding distance calculation.
            if len(self.__fitness_functions) > 1:
                # When choosing individuals for the next generation, there is a chance some solutions from the last pareto
                # front will be removed, which affects crowding distance metric. Therefore, we calculate the correct
                # crowding distances for the last pareto front.
                self.__calculate_crowding_distance(non_dominated_sorted_population[-1])

            # Put the individuals from all pareto fronts in the same list.
            population = [solution for pareto_front in non_dominated_sorted_population for solution in pareto_front]

            if self.__on_generation_finish_callback is not None:
                # The callback forwards current generation number, maximum number of generations and the current
                # population divided into pareto fronts.
                self.__on_generation_finish_callback(
                    generation_number,
                    self.__num_generations,
                    non_dominated_sorted_population
                )

            generation_number += 1

    def __evaluate_solution(self, solution, data):
        """Evaluate solutions using all fitness functions.

        :param genome solution: A solutions we want to evaluate.
        :param Object data: The data that can be used during evaluation (optional).
        :return: List[float]: Fitness scores for all fitness functions.
        """
        ff_values = []
        for fitness_function in self.__fitness_functions:
            ff_values.append(fitness_function.function(solution, data))
        return ff_values

    def __generate_random_population(self):
        """Generate `self.__population_size` random individuals.

        :param List[Genome] population
        :return: List[Genome]
        """

        population = []

        if self.__num_processes is not None:
            # Parallelize solution creation.

            population = self.__dispatch_mp(self.__generate_random_solution, self.__population_size)

        else:
            # Create solutions sequentially, without parallelization.

            for _ in range(self.__population_size):
                population.append(self.__generate_random_solution())

        return population

    def __generate_random_solution(self):
        """Generate one random solution.

        :return: Genome
        """

        # Create solution by copying template.
        solution = copy.deepcopy(self.__genome)

        # Randomize it.
        solution.randomize()

        # Evaluate it.
        solution.fitness_values_train = self.__evaluate_solution(solution, self.__data_train)
        if self.__validate:
            solution.fitness_values_val = self.__evaluate_solution(solution, self.__data_val)

        return solution

    def __perform_non_dominated_sort(self, population):
        """Performs non-dominated sorting of the population.

        Refer to the paper for more details.

        :param List[Genome] population: Individuals that need to be ranked and sorted.
        :return: List[List[Genome]]: Sorted population.
        """
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
                    if self.__fitness_functions[k].function_type == FitnessFunctionType.MIN:
                        # We want to minimize this FF, therefore the subtraction should return a positive number when
                        # population[i] has a lower FF value.
                        fitness_diff.append(population[j].fitness_values_train[k] - population[i].fitness_values_train[k])
                    elif self.__fitness_functions[k].function_type == FitnessFunctionType.MAX:
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
                # Solution population[i] is not dominated by any other solution, therefore it belongs to the first
                # (best) pareto front. Smaller rank is better.
                population[i].rank = 0
                pareto_fronts[0].append(i)

        i = 0
        # Iterate until each solution is assigned to a pareto front.
        while len(pareto_fronts[i]) > 0:
            # A list where solutions that belong to the next pareto front will be saved.
            next_pareto_front = []

            # Iterate over solutions on the last pareto front.
            for j in pareto_fronts[i]:
                for k in list_of_dominated_indices[j]:
                    # Reduce domination count for the solutions that are dominated by the individuals on the current
                    # pareto front.
                    domination_count[k] -= 1

                    # If the solution is no longer dominated, that is, all the solutions that dominated over the
                    # current solution were deployed to pareto fronts, add current solution to the next pareto front.
                    if domination_count[k] == 0:
                        population[k].rank = i + 1
                        next_pareto_front.append(k)

            # Jump to next pareto front.
            i += 1

            # Add current pareto front to the list of all pareto fronts.
            pareto_fronts.append(next_pareto_front)

        # Last pareto front is empty (check 'while' condition above), so we remove it.
        del pareto_fronts[-1]

        # Turn pareto front indices into objects; Replace index with the corresponding object in `population`.
        object_pareto_fronts = []

        for pareto_front in pareto_fronts:
            current_front = []
            for index in pareto_front:
                current_front.append(population[index])
            object_pareto_fronts.append(current_front)

        return object_pareto_fronts

    def __calculate_crowding_distance(self, pareto_front):
        """Calculate and assign crowding distance to each solution on the current pareto front.

        :param List[Genome] pareto_front
        """

        # Iterate over all fitness functions.
        for k, _ in enumerate(self.__fitness_functions):
            # Sort in ascending order according to current FF.
            sorted_pareto_front = sorted(
                pareto_front,
                key=lambda solution: solution.fitness_values_train[k]
            )

            # First and last solution have infinite crowding distance.
            sorted_pareto_front[0].crowding_distance = math.inf
            sorted_pareto_front[-1].crowding_distance = math.inf

            ff_range = sorted_pareto_front[-1].fitness_values_train[k] - sorted_pareto_front[0].fitness_values_train[k]

            # Later, we divide by ff_range, so we want to make sure it's not 0.
            if ff_range <= 0:
                ff_range = 1

            # Iterate over solutions on the current pareto front, excluding first and last one, and calculate the
            # contribution of each fitness function to the crowding distance.
            for i in range(1, len(sorted_pareto_front) - 1):
                # Contribution of kth fitness function to the ith solution.
                sorted_pareto_front[i].crowding_distance += (
                    (sorted_pareto_front[i + 1].fitness_values_train[k] - sorted_pareto_front[i - 1].fitness_values_train[k]) / ff_range
                )

    def __perform_1d_sort(self, population):
        """Performs a simple 1D sort.

        Sorts the `population` according to the one fitness function, taking into account the type of
        the problem (minimization or maximization). After that, each solution is given a rank and positioned
        on a separate pareto front. Crowding distance remains 0 for all solutions during the optimization process.

        :param List[Genome] population: Individuals that need to be ranked and sorted.
        :return: List[List[Genome]]: Sorted population.
        """

        # Determine sort type with regards to the problem.
        if self.__fitness_functions[0].function_type == FitnessFunctionType.MIN:
            reverse_sort = False
        elif self.__fitness_functions[0].function_type == FitnessFunctionType.MAX:
            reverse_sort = True

        # Sort the population.
        sorted_population = sorted(population, key=lambda solution: solution.fitness_values_train[0], reverse=reverse_sort)

        # Rank each solution and it on a separate pareto front.
        object_pareto_fronts = []

        for index, solution in enumerate(sorted_population):
            # Lower rank indicates higher quality.
            solution.rank = index
            object_pareto_fronts.append([solution])

        return object_pareto_fronts

    def __generate_offspring(self, population):
        """Generate `self.__offspring_size` number of individuals.

        :param List[Genome] population
        :return: List[Genome]
        """

        offspring = []

        if self.__num_processes is not None:
            # Parallelize children creation.

            offspring = self.__dispatch_mp(self.__generate_single_solution, self.__offspring_size, population)

        else:
            # Create children sequentially, without parallelization.

            for _ in range(self.__offspring_size):
                offspring.append(self.__generate_single_solution(population))

        return offspring

    def __generate_single_solution(self, population):
        """Create and return one child.

        :param List[Genome] population
        :return: Genome
        """
        # Pick the first parent.
        parent_1 = self.__tournament_select_parent(population)

        # Decide if we should just clone the first parent or perform recombination.
        if np.random.rand() <= self.__recombination_probability:
            # Pick the second parent and create a child.
            while True:
                parent_2 = self.__tournament_select_parent(population)
                if parent_1 is not parent_2:
                    break

            child = parent_1.recombination(parent_2)
        else:
            # Create a clone.
            child = copy.deepcopy(parent_1)

        # Mutate a child, introduce slight variation.
        child.mutate()

        # Evaluate the child.
        child.fitness_values_train = self.__evaluate_solution(child, self.__data_train)
        if self.__validate:
            child.fitness_values_val = self.__evaluate_solution(child, self.__data_val)

        return child

    def __tournament_select_parent(self, population):
        """Perform tournament selection between `self.__num_solutions_tournament` and pick the best one.

        Lower `rank` and higher `crowding distance` are desirable.

        :param List[Genome] population
        :return: Genome: The winner of the tournament.
        """
        # How many matches will be held.
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
                    (random_opponent.rank == random_parent.rank and
                     random_opponent.crowding_distance > random_parent.crowding_distance):
                random_parent_index = random_opponent_index

            # One less battle remaining.
            num_battles -= 1

            # We have a winner, return the best individual.
            if num_battles <= 0:
                return population[random_parent_index]

    def __choose_next_generation(self, non_dominated_sorted_population):
        """Pick individuals for the next generation.

        :param List[List[Genome]] non_dominated_sorted_population
        :return: List[List[Genome]]
        """
        next_generation = []
        size = 0

        # We start from the best pareto front and work our way towards the worst.
        for pareto_front in non_dominated_sorted_population:
            if len(pareto_front) + size <= self.__population_size:
                # If the whole pareto front fits into next generation, add it.
                next_generation.append(pareto_front)
                size += len(pareto_front)
            else:
                # The next generation is full, there are no more positions available.
                if self.__population_size - size <= 0:
                    break

                # If not the whole pareto front fits, add the individuals with the highest crowding distance to
                # preserve genetic diversity.
                pareto_front.sort(key=lambda solution: solution.crowding_distance)
                next_generation.append(pareto_front[-(self.__population_size - size):])
                break

        return next_generation


    # Multiprocessing

    def __dispatch_mp(self, creator_function, array_size, arguments_mp=None):
        """Generate `array_size` number of individuals using multiprocessing.

        :param Function creator_function: A function which will be used to create a single solution.
        :param int array_size: Number of solutions to create.
        :param Tuple arguments_mp: Arguments to be forwarded to the `creator_function`.
        :return: List[Genome]
        """

        array = []
        processes = []

        # Shared queue for messages.
        queue = Queue()

        # The maximum number of processes is dictated by `array_size`.
        num_proc = min(self.__num_processes, array_size)

        num_solutions = 0

        while True:
            processes.append(
                Process(
                    target=self.__generate_mp,
                    args=(queue, creator_function, arguments_mp)
                )
            )

            processes[-1].start()
            num_solutions += 1

            if len(processes) >= num_proc or num_solutions >= array_size:
                # A maximum number of processes is currently executing, wait for them to finish.

                for process in processes:
                    # Wait for the process to finish to successfully join it. If the process has not finished, it
                    # may be because the message queue is full. To avoid potential deadlocks between a child and
                    # a parent, empty the queue while waiting.
                    while process.is_alive():
                        while not queue.empty():
                            # Add children from the message queue to the `array` list.
                            array.append(queue.get())

                    process.join()

                processes.clear()

            # If `array_size` individuals were created, break the loop.
            if num_solutions >= array_size:
                break

        # Add the remaining solutions to the `array`.
        while not queue.empty():
            array.append(queue.get())

        return array

    def __generate_mp(self, queue, creator_function, arguments_mp):
        """A multiprocessing way to generate and return a solution.

        :param multiprocessing.Queue queue
        :param Function creator_function: A function which will be used to create a single solution.
        :param Tuple arguments_mp: Arguments to be forwarded to the `creator_function`.
        """

        # Each child process inherits an identical memory state, including random number generator. Therefore, to avoid
        # creating identical solutions by all the processes, setting a new random seed is necessary.
        np.random.seed(os.getpid() ** 2 % 10000)

        # Generate a solution and put it in a shared memory queue so it is available to the parent process.
        if arguments_mp is not None:
            queue.put(creator_function(arguments_mp))
        else:
            queue.put(creator_function())
