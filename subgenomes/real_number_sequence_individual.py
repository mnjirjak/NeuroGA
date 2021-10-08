from subgenome import Subgenome
from subgenomes.real_number import RealNumber


class RealNumberSequenceIndividual(Subgenome):
    """A sequence of RealNumber objects bound by min and max values."""
    def __init__(self, num_of_values=2, min_value=0.0, max_value=1.0, mutation_probability=None):
        """
        :param int num_of_values: The number of RealNumber objects in the sequence.
        :param float min_value: Minimum value of the number.
        :param float max_value: Maximum value of the number.
        :param float mutation_probability: Local mutation probability. If not specified, global mutation probability
                                           will be used.
        """
        super().__init__(mutation_probability)

        self.num_of_values = num_of_values

        self.__min_value = min_value
        self.__max_value = max_value

        self.real_numbers = [
            RealNumber(
                min_value=self.__min_value,
                max_value=self.__max_value,
                mutation_probability=self.__mutation_probability
            ) for _ in range(self.num_of_values)
        ]

    def randomize(self):
        """Call `randomize()` for each RealNumber object in `self.real_numbers`."""
        for real_number in self.real_numbers:
            real_number.randomize()

    def recombination(self, partner):
        """Perform recombination between self and partner.

        Call `recombination(partner)` for each RealNumber object in `self.real_numbers`.
        """
        child_real_numbers = []

        # Fill `child_real_numbers` with new RealNumber objects obtained by recombination.
        for i in range(len(self.real_numbers)):
            child_real_numbers.append(self.real_numbers[i].recombination(partner.real_numbers[i]))

        # Create a new child object.
        child = RealNumberSequenceIndividual(
            num_of_values=self.num_of_values,
            min_value=self.__min_value,
            max_value=self.__max_value,
            mutation_probability=self.__mutation_probability
        )

        # Set child value.
        child.real_numbers = child_real_numbers

        return child

    def mutate(self):
        """Call `mutate()` for each RealNumber object in `self.real_numbers`."""
        for real_number in self.real_numbers:
            real_number.mutate()

    def set_mutation_probability(self, mutation_probability):
        """Set `self.__mutation_probability` to desired value and propagate it to subgenomes in `self.real_numbers`."""
        super().set_mutation_probability(mutation_probability)

        for real_number in self.real_numbers:
            real_number.set_mutation_probability(mutation_probability)
