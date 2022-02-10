from NeuroGA.neuroga.subgenomes.real_number_sequence import RealNumberSequence
import numpy as np


class NormalizedRealNumberSequence(RealNumberSequence):
    """A normalized sequence of real numbers bound by max value. Min values is always 0."""

    def __init__(self,
                 num_of_values,
                 max_value=1.0,
                 min_mutation_value=-0.05,
                 max_mutation_value=0.05,
                 mutation_probability=None):
        """
        :param int num_of_values: The number of real values in the sequence.
        :param float max_value: Maximum value of the number.
        :param float min_mutation_value: Minimum value when choosing mutation.
        :param float max_mutation_value: Maximum value when choosing mutation.
        :param float mutation_probability: Local mutation probability. If not specified, global mutation probability
                                           will be used.
        """

        super().__init__(num_of_values=num_of_values,
                         min_value=0.0,
                         max_value=max_value,
                         min_mutation_value=min_mutation_value,
                         max_mutation_value=max_mutation_value,
                         mutation_probability=mutation_probability)

    def randomize(self):
        """Randomly assign values in [self.__min_value, self.__max_value) range and normalize them."""

        super().randomize()
        self.values = self.__normalize_values(self.values)

    def recombination(self, partner):
        """Combine this individual with `partner`.

        Recombination details are left to the superclass, RealNumberSequence.

        :param NormalizedRealNumberSequence partner
        :return: NormalizedRealNumberSequence
        """

        # Create a new child object.
        child = super().recombination(partner)
        child.values = self.__normalize_values(child.values)

        return child

    def mutate(self):
        """Perform mutation, introduce a slight variation.

        Mutation details are left to the superclass, RealNumberSequence.
        """

        child = super().mutate()
        child.values = self.__normalize_values(child.values)

    def __normalize_values(self, values):
        """Normalizes `values` to [0, 1] range.

        :param np.array[np.float] values: A Numpy array of values to be normalized.
        """

        return values/np.sum(values)
