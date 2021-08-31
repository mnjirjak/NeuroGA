class Genome:
    """A class that merges multiple subgenomes."""
    def __init__(self, subgenomes):
        """Construct a genome composed of subgenomes.

        :param List[Subgenome] subgenomes: A list of subgenomes that define an individual.
        """
        self.__subgenomes = subgenomes
        self.__fitness_values = []
        self.__rank = -1
        self.__crowding_distance = 0

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

    def set_rank(self, rank):
        """
        :param int rank: Individual's NSGA-II rank
        """
        self.__rank = rank

    def set_crowding_distance(self, crowding_distance):
        """
        :param float crowding_distance: Individual's NSGA-II crowding_distance
        """
        self.__crowding_distance = crowding_distance

    def get_rank(self, rank):
        """
        :param int rank: Individual's NSGA-II rank
        """
        return self.__rank

    def get_crowding_distance(self, crowding_distance):
        """
        :param float crowding_distance: Individual's NSGA-II crowding_distance
        """
        return self.__crowding_distance

    def get_subgenomes(self):
        """
        :return: List[Subgenomes]
        """
        return self.__subgenomes
