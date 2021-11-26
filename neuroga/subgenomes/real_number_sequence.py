from neuroga.subgenome import Subgenome
import numpy as np


class RealNumberSequence(Subgenome):
    """A sequence of real numbers bound by min and max values."""

    def __init__(self,
                 num_of_values,
                 min_value=-1.0,
                 max_value=1.0,
                 min_mutation_value=-0.05,
                 max_mutation_value=0.05,
                 mutation_probability=None):
        """
        :param int num_of_values: The number of real values in the sequence.
        :param float min_value: Minimum value of the number.
        :param float max_value: Maximum value of the number.
        :param float min_mutation_value: Minimum value when choosing mutation.
        :param float max_mutation_value: Maximum value when choosing mutation.
        :param float mutation_probability: Local mutation probability. If not specified, global mutation probability
                                           will be used.
        """
        super().__init__(mutation_probability)

        self.__num_of_values = num_of_values

        self.__min_value = min_value
        self.__max_value = max_value

        self.__min_mutation_value = min_mutation_value
        self.__max_mutation_value = max_mutation_value

        self.values = np.zeros(self.__num_of_values)

    def randomize(self):
        """Randomly assign values in [self.__min_value, self.__max_value) range."""
        self.values = np.random.random(self.__num_of_values) * \
                           (self.__max_value - self.__min_value) + self.__min_value

    def recombination(self, partner):
        """Combine this individual with `partner` using arithmetic mean.

        An arithmetic mean of two values which are in [self.__min_value, self.__max_value) range is also in the same
        range. Therefore, we don't need to check min and max boundaries after recombination.

        :param RealNumberSequence partner
        :return: RealNumberSequence
        """

        # Create a new child object.
        child = RealNumberSequence(
            num_of_values=self.__num_of_values,
            min_value=self.__min_value,
            max_value=self.__max_value,
            min_mutation_value=self.__min_mutation_value,
            max_mutation_value=self.__max_mutation_value,
            mutation_probability=self._mutation_probability
        )

        # Set child values.
        child.values = (self.values + partner.values) / 2

        return child

    def mutate(self):
        """Perform mutation, introduce a slight variation.

        Value is mutated by adding random numbers in [self.__min_mutation_value, self.__max_mutation_value)
        range to real number values. After the mutation, we must check min and max boundaries.
        """

        mutation_values = np.random.rand(self.__num_of_values) * \
            (self.__max_mutation_value - self.__min_mutation_value) - self.__min_mutation_value

        self.values = min(max(self.values + mutation_values, self.__min_value), self.__max_value)
