class Genome:
    """A class that merges multiple subgenomes."""
    def __init__(self, subgenomes):
        """Construct a genome composed of subgenomes.

        :param List[Subgenome] subgenomes: A list of subgenomes that define an individual.
        """
        self.__subgenomes = subgenomes
        self.fitness_values_train = None
        self.fitness_values_val = None
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
            subgenome.mutate()

    def set_mutation_probabilities(self, mutation_probability_global):
        for _, subgenome in self.__subgenomes.items():
            if subgenome.mutation_probability == None:
                subgenome.mutation_probability = mutation_probability_global

    def get_subgenomes(self):
        """
        :return: List[Subgenomes]
        """
        return self.__subgenomes
