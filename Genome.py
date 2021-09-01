class Genome:
    """A class that merges multiple subgenomes."""
    def __init__(self, subgenomes):
        """Construct a genome composed of subgenomes.

        :param List[Subgenome] subgenomes: A list of subgenomes that define an individual.
        """
        self.__subgenomes = subgenomes
        self.fitness_values_train = []
        self.fitness_values_val = []
        self.rank = -1
        self.crowding_distance = 0

    def randomize(self):
        """Randomize subgenomes."""
        for subgenome in self.__subgenomes:
            subgenome.randomize()

    def recombination(self, genome):
        """Perform recombination and return new genome.

        :param Genome genome: An individual we want to recombine with.
        :return: Genome
        """
        subgenome_list = []
        for i, _ in enumerate(genome.get_subgenomes()):
            subgenome_list.append(self.__subgenomes[i].recombine(genome.get_subgenomes()[i]))

        return Genome(subgenome_list)

    def mutate(self):
        """Mutate all the subgenomes."""
        for subgenome in self.__subgenomes:
            subgenome.mutate()

    def get_subgenomes(self):
        """
        :return: List[Subgenomes]
        """
        return self.__subgenomes
