import unittest

from fitness_function import FitnessFunction, FitnessFunctionType
from genome import Genome
from nsga_ii import NSGAII


class MutationPropagationTest(unittest.TestCase):
    """A superclass for testing mutation propagation.

    This is used to test if mutation parameters propagate accurately in subgenomes.
    """

    def run_algo(self, subgenome, mutation_probability_global=None):
        """Run the algorithm birefly.

        :param Subgenome subgenome: A subgenome that is tested.
        :param float mutation_probability_global: Global mutation probability parameter forwarded to the algorithm.
        :return: List[List[Genome]]
        """

        # Construct the algorithm by specifying global mutation probability parameter.
        algo = NSGAII(
            genome=Genome(
                subgenomes={
                    'var': subgenome
                }
            ),
            fitness_functions=[
                FitnessFunction(function=lambda solution, data: 1, function_type=FitnessFunctionType.MIN)
            ],
            population_size=2,
            offspring_size=1,
            num_generations=2,
            num_solutions_tournament=2,
            mutation_probability_global=mutation_probability_global
        )

        # If `mutation_probability_global` is `None`, then construct the algorithm without global mutation probability
        # parameter. This leads to the default global mutation probability being used.
        if mutation_probability_global is None:
            algo = NSGAII(
                genome=Genome(
                    subgenomes={
                        'var': subgenome
                    }
                ),
                fitness_functions=[
                    FitnessFunction(function=lambda solution, data: 1, function_type=FitnessFunctionType.MIN)
                ],
                population_size=2,
                offspring_size=1,
                num_generations=2,
                num_solutions_tournament=2
            )

        # Return pareto fronts.
        return algo.optimize()

    def examine_simple(self, mutation_probability, mutation_probability_global, solution):
        """Perform a simple examination of mutation probability propagation.

        We simply check if mutation probability of a `subgenome` is correct. Local mutation probability overrides the
        global one.

        :param float mutation_probability: Local mutation probability for this subgenome.
        :param float mutation_probability_global: Global mutation probability forwarded to the algorithm.
        :param Subgenome solution
        """

        # Both local and global mutation probabilities are `None`.
        if mutation_probability is None and mutation_probability_global is None:
            # Mutation probability should use the default global value and should be a `float`.
            self.assertTrue(isinstance(solution.get_mutation_probability(), float))

        # Local mutation probability is `None`, but the global one is not.
        elif mutation_probability is None:
            # Mutation probability should be equal the global mutation probability.
            self.assertAlmostEqual(
                solution.get_mutation_probability(),
                mutation_probability_global
            )

        # In any other case, local mutation probability is not `None`, and the genome should have mutation probability
        # equal to the local one.
        else:
            self.assertAlmostEqual(
                solution.get_mutation_probability(),
                mutation_probability
            )
