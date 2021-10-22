from tests.test_mutation_propagation.mutation_propagation_test import MutationPropagationTest
from parameterized import parameterized

from subgenomes.ordered_nd_list import OrderedNDList


class OrderedNDListMutationPropagationTest(MutationPropagationTest):
    """Test mutation propagation of `OrderedNDList` subgenomes."""

    def examine_ordered_nd_list(self, mutation_probability, mutation_probability_global):
        """Check if correct mutation probability is propagated.

        :param float mutation_probability: Local mutation probability for this subgenome.
        :param float mutation_probability_global: Global mutation probability forwarded to the algorithm.
        """

        # Both probabilities are `None`, so use default parameter values.
        if mutation_probability is None and mutation_probability_global is None:
            pareto_fronts = self.run_algo(
                subgenome=OrderedNDList(items=['A', 'B'])
            )
        # Global mutation probability is not `None`, so forward it to the algorithm.
        elif mutation_probability is None and mutation_probability_global is not None:
            pareto_fronts = self.run_algo(
                subgenome=OrderedNDList(items=['A', 'B']),
                mutation_probability_global=mutation_probability_global
            )
        # Local mutation probability is not `None`, so forward it to the subgenome.
        elif mutation_probability is not None and mutation_probability_global is None:
            pareto_fronts = self.run_algo(
                subgenome=OrderedNDList(
                    items=['A', 'B'],
                    mutation_probability=mutation_probability
                )
            )
        # Both probabilities are not `None`, so forward them to the algorithm.
        else:
            pareto_fronts = self.run_algo(
                subgenome=OrderedNDList(
                    items=['A', 'B'],
                    mutation_probability=mutation_probability
                ),
                mutation_probability_global=mutation_probability_global
            )

        # Check if correct values are propagated.
        self.examine_simple(
            mutation_probability,
            mutation_probability_global,
            pareto_fronts[0][0].get_subgenomes()['var']
        )

    @parameterized.expand([
        [None, None],
        [0.6, None],
        [None, 0.3],
        [0.6, 0.3]
    ])
    def test_ordered_nd_list(self, mutation_probability, mutation_probability_global):
        """Test for correct results with multiple values using parametrized tests.

        :param float mutation_probability: Local mutation probability for this subgenome.
        :param float mutation_probability_global: Global mutation probability forwarded to the algorithm.
        """
        self.examine_ordered_nd_list(mutation_probability, mutation_probability_global)
