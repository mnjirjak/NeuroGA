from libname.tests.test_mutation_propagation.mutation_propagation_test import MutationPropagationTest
from parameterized import parameterized

from libname.subgenomes.real_number_sequence_individual import RealNumberSequenceIndividual


class RealNumberSequenceIndividualMutationPropagationTest(MutationPropagationTest):
    """Test mutation propagation of `OrderedNDList` subgenomes."""

    def examine_real_number_sequence_individual(self, mutation_probability, mutation_probability_global):
        """Check if correct mutation probability is propagated.

        Since this is a complex subgenome, i.e. it consists of many `RealNumber` subgenomes, we need to check
        if correct mutation probability is set in subgenomes objects as well as in the subgenome itself.

        :param float mutation_probability: Local mutation probability for this subgenome.
        :param float mutation_probability_global: Global mutation probability forwarded to the algorithm.
        """

        # Both probabilities are `None`, so use default parameter values.
        if mutation_probability is None and mutation_probability_global is None:
            pareto_fronts = self.run_algo(
                subgenome=RealNumberSequenceIndividual(num_of_values=2)
            )
        # Global mutation probability is not `None`, so forward it to the algorithm.
        elif mutation_probability is None and mutation_probability_global is not None:
            pareto_fronts = self.run_algo(
                subgenome=RealNumberSequenceIndividual(num_of_values=2),
                mutation_probability_global=mutation_probability_global
            )
        # Local mutation probability is not `None`, so forward it to the subgenome.
        elif mutation_probability is not None and mutation_probability_global is None:
            pareto_fronts = self.run_algo(
                subgenome=RealNumberSequenceIndividual(
                    num_of_values=2,
                    mutation_probability=mutation_probability
                )
            )
        # Both probabilities are not `None`, so forward them to the algorithm.
        else:
            pareto_fronts = self.run_algo(
                subgenome=RealNumberSequenceIndividual(
                    num_of_values=2,
                    mutation_probability=mutation_probability
                ),
                mutation_probability_global=mutation_probability_global
            )

        # Check if correct values are propagated in this subgenome.
        self.examine_simple(
            mutation_probability,
            mutation_probability_global,
            pareto_fronts[0][0].get_subgenomes()['var']
        )

        # Check if correct values are propagated in the subgenomes of this subgenome.
        for real_number_object in pareto_fronts[0][0].get_subgenomes()['var'].real_numbers:
            self.examine_simple(
                mutation_probability,
                mutation_probability_global,
                real_number_object
            )

    @parameterized.expand([
        [None, None],
        [0.6, None],
        [None, 0.3],
        [0.6, 0.3]
    ])
    def test_real_number_sequence_individual(self, mutation_probability, mutation_probability_global):
        """Test for correct results with multiple values using parametrized tests.

        :param float mutation_probability: Local mutation probability for this subgenome.
        :param float mutation_probability_global: Global mutation probability forwarded to the algorithm.
        """
        self.examine_real_number_sequence_individual(mutation_probability, mutation_probability_global)
