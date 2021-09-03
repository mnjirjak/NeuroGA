from subgenome import Subgenome
import numpy as np


class RealNumber(Subgenome):
    """A real number bound by min and max values."""
    def __init__(self, min_value=0.0, max_value=1.0, value=0.0, mutation_probability=None):
        """
        :param float min_value: Minimum value of the number.
        :param float max_value: Maximum value of the number.
        :param float value: Current value of the number.
        :param float mutation_probability: Local mutation probability. If not specified, global mutation probability
                                           will be used.
        """
        super().__init__(mutation_probability)

        self.__min_value = min_value
        self.__max_value = max_value

        self.real_number = value

    def randomize(self):
        """Randomly assign value in [self.min_value, self.max_value) range."""
        self.real_number = np.random.rand() * (self.__max_value - self.__min_value) + self.__min_value

    def recombination(self, partner):
        """Combine this individual with `partner` using arithmetic mean.

        While combining them, we must respect min and max boundaries.

        :param RealNumber partner
        :return: RealNumber
        """
        return RealNumber(
            min_value=self.__min_value,
            max_value=self.__max_value,
            value=min(max((self.real_number + partner.real_number) / 2, self.__min_value), self.__max_value),
            mutation_probability=self.mutation_probability
        )

    def mutate(self):
        """Perform mutation, introduce a slight variation.

        Value is mutated by adding or subtracting a small number.

        Two key steps:
        1. Select sign (+ or -).
        2. Select the degree of change with respect to `self.mutation_probability` and `self.real_number`.
        """
        self.real_number += \
            (1.0 if np.random.rand() <= 0.5 else -1.0) * (np.random.rand() * self.mutation_probability) * self.real_number
        self.real_number = min(max(self.real_number, self.__min_value), self.__max_value)
