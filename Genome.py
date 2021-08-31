class Genome:
    """A class that merges multiple subgenomes."""
    def __init__(self, subgenomes):
        """Construct a genome composed of subgenomes.

        :param List[Subgenome] subgenomes: A list of subgenomes that define an individual.
        """
        self.__subgenomes = subgenomes
        self.__fitness_values = []

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
        for subgenome_index, _ in enumerate(genome.get_subgenomes()):
            subgenome_list.append(
                self.__subgenomes[subgenome_index].recombine(
                    genome.get_subgenomes()[subgenome_index]
                )
            )

        return Genome(subgenome_list)

    def mutate(self):
        """Mutate all the subgenomes."""
        for subgenome in self.__subgenomes:
            subgenome.mutate()

    def get_subgenomes(self):
        return self.__subgenomes
