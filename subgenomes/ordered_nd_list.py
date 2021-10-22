import copy
from subgenome import Subgenome
import numpy as np
import random


class OrderedNDList(Subgenome):
    """Ordered, non-duplicate list.

    List is without duplicates and the order of the items matters.
    """

    def __init__(self, items, mutation_probability=None):
        """
        :param List[Object] items: A list of items whose optimal order needs to be found.
        :param float mutation_probability: Local mutation probability. If not specified, global mutation probability
                                           will be used.
        """
        super().__init__(mutation_probability)

        self.items = items

    def randomize(self):
        """Randomly shuffle the list."""
        random.shuffle(self.items)

    def recombination(self, partner):
        """Combine this individual with `partner`.

        The individuals are combined by taking a part of the `self.items`, defined by `random_index`, and filling the
        missing items from `partner.items` in the respective order.

        :param OrderedNDList partner
        :return: OrderedNDList
        """

        random_index = np.random.randint(low=0, high=len(self.items))
        child = self.items[:random_index]

        for i in range(len(partner.items)):
            # Append the items that are not yet in child.
            if partner.items[i] not in child:
                child.append(partner.items[i])

        return OrderedNDList(
            # Create a copy of items. This avoids the problem with references.
            items=copy.deepcopy(child),
            mutation_probability=self._mutation_probability
        )

    def mutate(self):
        """Perform mutation, introduce a slight variation.

        The list is mutated by conducting `num_swaps` swaps. This ensures variation is introduced and, at the same time,
        no duplicates are inserted. The `num_swaps` is calculated with respect to `self._mutation_probability` and is
        1 or higher.
        """
        num_swaps = max(1, int(round(self._mutation_probability / 2 * len(self.items))))

        for i in range(num_swaps):
            random_index_1 = np.random.randint(low=0, high=len(self.items))

            # Loop until the random indices are not equal.
            while True:
                random_index_2 = np.random.randint(low=0, high=len(self.items))
                if random_index_1 != random_index_2:
                    break

            # Swap elements.
            temp = self.items[random_index_1]
            self.items[random_index_1] = self.items[random_index_2]
            self.items[random_index_2] = temp
