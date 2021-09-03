from abc import ABC, abstractmethod


class Subgenome(ABC):
    """Abstract class used to define a Subgenome and its methods.

    Each subgenome must extend it.
    """
    def __init__(self, mutation_probability=None):
        self.mutation_probability = mutation_probability

    @abstractmethod
    def randomize(self):
        """Randomly initialize values in subgenome."""
        pass

    @abstractmethod
    def recombination(self, partner):
        """Perform recombination between current subgenome and `partner`.

        Be careful to return a new object and not just a reference to an existing one.

        :param Subgenome partner: A subgenome we want to combine with.
        :return Subgenome
        """
        pass

    @abstractmethod
    def mutate(self):
        """Mutate current subgenome."""
        pass
