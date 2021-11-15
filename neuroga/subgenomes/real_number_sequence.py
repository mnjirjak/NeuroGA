from neuroga.subgenome import Subgenome
import numpy as np


class RealNumberSequence(Subgenome):
    """A sequence of real numbers bound by min and max values."""

    def __init__(self, num_of_values, min_value=0.0, max_value=1.0, mutation_probability=None):
        """
        :param int num_of_values: The number of real values in the sequence.
        :param float min_value: Minimum value of the number.
        :param float max_value: Maximum value of the number.
        :param float mutation_probability: Local mutation probability. If not specified, global mutation probability
                                           will be used.
        """
        super().__init__(mutation_probability)

        self.num_of_values = num_of_values

        self.__min_value = min_value
        self.__max_value = max_value

        self.values = np.zeros(self.num_of_values)

    def randomize(self):
        """Randomly assign values in [self.__min_value, self.__max_value) range."""
        self.values = np.random.random(self.num_of_values) * \
                           (self.__max_value - self.__min_value) + self.__min_value

    def recombination(self, partner):
        """Combine this individual with `partner` using multi point crossover.

        :param RealNumberSequence partner
        :return: RealNumberSequence
        """
        # Create a mask of the same shape as `self.values`.
        mask = np.random.random(self.num_of_values)

        # Make approximately 50% of the `mask` values 1.0, and 50% 0.0.
        mask[mask >= 0.5] = 1.0
        mask[mask != 1.0] = 0.0

        # Invert the `mask`. `mask_inv` will have 1.0 where `mask` has 0.0 and vice versa.
        mask_inv = np.abs(mask - 1.0)

        # Create a new child object.
        child = RealNumberSequence(
            num_of_values=self.num_of_values,
            min_value=self.__min_value,
            max_value=self.__max_value,
            mutation_probability=self._mutation_probability
        )

        # Take a portion of the values from `self.values` and the rest from `partner.values`.
        child.values = (self.values * mask + partner.values * mask_inv)

        return child

    def mutate(self):
        """Perform mutation, introduce a slight variation.

        Values are mutated by randomly replacing `self._mutation_probability` sequence members with random values
        in [self.__min_value, self.__max_value) range.
        """
        # Generate an array of random numbers.
        random_array = np.random.random(self.num_of_values) * \
                       (self.__max_value - self.__min_value) + self.__min_value

        # Create a mask of the same shape as `self.values`.
        mask = np.random.random(self.num_of_values)

        # Make approximately `self._mutation_probability` values in `mask` equal to 0.0, and the rest equal to 1.0.
        mask[mask > self._mutation_probability] = 1.0
        mask[mask != 1.0] = 0.0

        # Invert the `mask`. `mask_inv` will have 1.0 where `mask` has 0.0 and vice versa.
        mask_inv = np.abs(mask - 1.0)

        # Preserve the values denoted by 1.0s in `mask`, and replace the rest with values from `random_array`.
        # This assigns random values to approximately `self._mutation_probability` percent of the values in
        # `self.values`.
        self.values = self.values * mask + random_array * mask_inv
