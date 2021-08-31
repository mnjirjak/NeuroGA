from abc import ABC, abstractmethod


class Subgenome(ABC):
    """An interface used to define a Subgenome and its methods.

    Each subgenome must implement it.
    """
    @abstractmethod
    def randomize(self):
        """Randomly initialize values in subgenome."""
        pass

    @abstractmethod
    def recombination(self, partner):
        """Perform recombination between current subgenome and `partner`.

        :param Subgenome partner: A subgenome we want to combine with.
        :return SubGenome
        """
        pass

    @abstractmethod
    def mutate(self):
        """Mutate current subgenome."""
        pass
