import numpy as np


class Genome:
    """A class that merges multiple subgenomes."""
    def __init__(self, subgenomes):
        """Construct a genome composed of subgenomes.

        :param Dict[Subgenome] subgenomes: A dictionary of subgenomes that define an individual.
        """
        self.__subgenomes = subgenomes

        # Individual's fitness values for train and validation.
        self.fitness_values_train = None
        self.fitness_values_val = None

        # NSGA-II rank and crowding distance parameters.
        self.rank = -1
        self.crowding_distance = 0

    def randomize(self):
        """Randomize subgenomes."""
        for _, subgenome in self.__subgenomes.items():
            subgenome.randomize()

    def recombination(self, genome):
        """Perform recombination and return new genome.

        :param Genome genome: An individual we want to recombine with.
        :return: Genome
        """
        subgenome_dict = {}
        for key, subgenome in genome.get_subgenomes().items():
            subgenome_dict[key] = self.__subgenomes[key].recombination(subgenome)

        return Genome(subgenome_dict)

    def mutate(self):
        """Mutate all the subgenomes."""
        for _, subgenome in self.__subgenomes.items():
            # Choose whether to mutate the subgenome with respect to its mutation probability.
            if np.random.rand() <= subgenome.get_mutation_probability():
                subgenome.mutate()

    def set_mutation_probabilities(self, mutation_probability_global):
        """Set mutation probabilities for subgenomes.

        For each subgenome that doesn't have a mutation probability defined, use global mutation probability.

        :param float mutation_probability_global: A real number indicating how likely it is a mutation will happen.
        """
        for _, subgenome in self.__subgenomes.items():
            if subgenome.get_mutation_probability() is None:
                subgenome.set_mutation_probability(mutation_probability_global)

    def get_subgenomes(self):
        """
        :return: Dict[Subgenomes]
        """
        return self.__subgenomes
