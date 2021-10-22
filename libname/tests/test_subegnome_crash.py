import unittest
import numpy as np

from libname.fitness_function import FitnessFunction, FitnessFunctionType
from libname.genome import Genome
from libname.nsga_ii import NSGAII

from libname.subgenomes.real_number import RealNumber
from libname.subgenomes.real_number_sequence import RealNumberSequence
from libname.subgenomes.real_number_sequence_individual import RealNumberSequenceIndividual
from libname.subgenomes.ordered_nd_list import OrderedNDList
from libname.subgenomes.keras_nn import KerasNN


def run_algo_crash_test(subgenome):
    """Run the algorithm briefly, just to check for crash.

    :param Subgenome subgenome: A subgenome that is tested.
    """
    algo = NSGAII(
        genome=Genome(
            subgenomes={
                'var': subgenome
            }
        ),
        fitness_functions=[
            # A simple fitness function that always returns 1.
            FitnessFunction(function=lambda solution, data: 1, function_type=FitnessFunctionType.MIN)
        ],
        population_size=10,
        offspring_size=5,
        num_generations=10,
        num_solutions_tournament=3
    )

    # Run the algorithm.
    algo.optimize()


class TestSubgenomeCrash(unittest.TestCase):
    """Test if subgenomes crash."""

    def test_real_number(self):
        """Check if RealNumber subgenome crashes."""
        run_algo_crash_test(
            RealNumber()
        )
        self.assertTrue(True)

    def test_real_number_sequence(self):
        """Check if RealNumberSequence subgenome crashes."""
        run_algo_crash_test(
            RealNumberSequence(
                num_of_values=2
            )
        )
        self.assertTrue(True)

    def test_real_number_sequence_individual(self):
        """Check if RealNumberSequenceIndividual subgenome crashes."""
        run_algo_crash_test(
            RealNumberSequenceIndividual(
                num_of_values=2
            )
        )
        self.assertTrue(True)

    def test_ordered_nd_list(self):
        """Check if OrderedNDList subgenome crashes."""
        run_algo_crash_test(
            OrderedNDList(
                items=['A', 'B', 'C', 'D']
            )
        )
        self.assertTrue(True)

    def test_keras_nn(self):
        """Check if KerasNN subgenome crashes."""
        run_algo_crash_test(
            KerasNN(
                model_weights=[
                    np.random.random((3, 4, 5, 6)),
                    np.random.random((3, 4, 5, 6))
                ]
            )
        )
        self.assertTrue(True)
