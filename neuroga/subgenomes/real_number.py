from neuroga.subgenome import Subgenome
import numpy as np


class RealNumber(Subgenome):
    """A real number bound by min and max values."""

    def __init__(self,
                 min_value=-1.0,
                 max_value=1.0,
                 min_mutation_value=-0.05,
                 max_mutation_value=0.05,
                 mutation_probability=None):
        """
        :param float min_value: Minimum value of the number.
        :param float max_value: Maximum value of the number.
        :param float min_mutation_value: Minimum value when choosing mutation.
        :param float max_mutation_value: Maximum value when choosing mutation.
        :param float mutation_probability: Local mutation probability. If not specified, global mutation probability
                                           will be used.
        """
        super().__init__(mutation_probability)

        self.__min_value = min_value
        self.__max_value = max_value

        self.__min_mutation_value = min_mutation_value
        self.__max_mutation_value = max_mutation_value

        self.real_number = 0.0

    def randomize(self):
        """Randomly assign value in [self.__min_value, self.__max_value) range."""
        self.real_number = np.random.rand() * (self.__max_value - self.__min_value) + self.__min_value

    def recombination(self, partner):
        """Combine this individual with `partner` using arithmetic mean.

        An arithmetic mean of two values which are in [self.__min_value, self.__max_value) range is also in the same
        range. Therefore, we don't need to check min and max boundaries after recombination.

        :param RealNumber partner
        :return: RealNumber
        """

        # Create a new child object.
        child = RealNumber(
            min_value=self.__min_value,
            max_value=self.__max_value,
            min_mutation_value=self.__min_mutation_value,
            max_mutation_value=self.__max_mutation_value,
            mutation_probability=self._mutation_probability
        )

        # Set child value.
        child.real_number = (self.real_number + partner.real_number) / 2

        return child

    def mutate(self):
        """Perform mutation, introduce a slight variation.

        Value is mutated by adding a random number in [self.__min_mutation_value, self.__max_mutation_value)
        range. After the mutation, we must check min and max boundaries.
        """
        mutation_value = np.random.rand() * \
            (self.__max_mutation_value - self.__min_mutation_value) - self.__min_mutation_value

        self.real_number = min(max(self.real_number + mutation_value, self.__min_value), self.__max_value)
