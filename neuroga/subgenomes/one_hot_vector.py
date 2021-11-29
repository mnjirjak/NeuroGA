from neuroga.subgenome import Subgenome
import numpy as np


class OneHotVector(Subgenome):
    """One-hot vector subgenome.

    A list of zeros with only a single member equal to 1.
    """

    def __init__(self, num_of_items, mutation_probability=None):
        """
        :param int num_of_items: Specifies vector length, i.e. the number of members.
        :param List[np.uint8] items: One-hot vector. Primarily used when creating a child during recombination.
        :param float mutation_probability: Local mutation probability. If not specified, global mutation probability
                                           will be used.
        """

        super().__init__(mutation_probability)

        self.__num_of_items = num_of_items
        self.items = np.zeros(self.__num_of_items, dtype=np.uint8)

    def randomize(self):
        """Create a random one-hot vector."""

        self.items[np.random.randint(0, self.__num_of_items)] = 1

    def recombination(self, partner):
        """Combine this individual with `partner`.

        The individuals are combined by taking an arithmetic mean of the indices that mark the positions of 1s in the
        vectors, using integer division. A child will have a 1 placed at the the position marked by the resulting index.

        :param OneHotVector partner
        :return: OneHotVector
        """

        first_index = np.where(self.items == 1)[0][0]
        second_index = np.where(partner.items == 1)[0][0]

        recombination_index = (first_index + second_index) // 2

        child_items = np.zeros(self.__num_of_items, dtype=np.uint8)
        child_items[recombination_index] = 1

        # Create a new child object.
        child = OneHotVector(
            num_of_items=self.__num_of_items,
            mutation_probability=self._mutation_probability
        )

        # Forward the vector to the child.
        child.items = child_items

        return child

    def mutate(self):
        """Perform mutation, introduce a slight variation.

        The vector is mutated in only 5% of the cases. Mutation occurs by randomly moving the 1 one place to the left,
        or one place to the right. If the start or the end of the vector is reached, mutation is not introduced.
        """

        # Get the position of a 1.
        index = np.where(self.items == 1)[0][0]

        # Save the position for swapping.
        old_index = index

        # Randomly move to the right or to the left.
        if np.random.rand() >= 0.5:
            index += 1
        else:
            index -= 1

        # If `index` is in vector position range, perform the mutation.
        if 0 <= index < self.__num_of_items:
            self.items[old_index] = 0
            self.items[index] = 1
