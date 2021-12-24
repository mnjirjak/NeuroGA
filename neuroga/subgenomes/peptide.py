from neuroga.subgenome import Subgenome
import numpy as np


class ModifiedPeptide(Subgenome):
    """A real number bound by min and max values."""

    def __init__(self,
                 initial_peptide="",
                 control_mask=None,
                 min_length=5,
                 max_length=50,
                 amino_acids=None,
                 mutation_probability=None):
        """
        :param float min_value: Minimum value of the number.
        :param float max_value: Maximum value of the number.
        :param float min_mutation_value: Minimum value when choosing mutation.
        :param float max_mutation_value: Maximum value when choosing mutation.
        :param float mutation_probability: Local mutation probability. If not specified, global mutation probability
                                           will be used.
        """
        super().__init__(mutation_probability)

        # Dosta ovih membera može biti private __.

        self.__initial_peptide = initial_peptide
        self.__control_mask = control_mask

        self.__min_length = min_length
        self.__max_length = max_length

        self.__amino_acids = amino_acids

        self.__important_groups = []

        if control_mask is not None:
            self.__important_groups, self.mandatory_length = self.extract_important_groups(initial_peptide, control_mask)

        # Min i max length su zapravo ovisni o ovoj mandatory_length




        self.__peptide_amino_acids = []
        self.__peptide_important_groups = {}

    def randomize(self):
        """Ovdje možemo imati dva pristupa:
        1. Na random inicijaliziramo peptide koji sadrže bite skupine.
        2. Mrvicu modificiramo početni peptid, pa populacija bude u okolini početnog peptida. (za ovo se može mutacija pozavti nekoliko puta)
        """

        num_amino_acids = np.random.randint(
            low=self.__min_length - self.mandatory_length,
            high=self.__max_length - self.mandatory_length + 1
        )

        amino_acid_indices = np.random.randint(
            low=0,
            high=len(self.__amino_acids),
            size=num_amino_acids
        )

        self.__peptide_amino_acids = [self.__amino_acids[amino_acid_index] for amino_acid_index in amino_acid_indices]

        for group in self.__important_groups:
            self.__peptide_important_groups[group] = []

            for _ in range(self.__important_groups[group]):
                insert_index = np.random.randint(
                    low=0,
                    high=(len(self.__peptide_amino_acids) + 1)
                )

                self.__peptide_important_groups[group].append(insert_index)


    def recombination(self, partner):
        """Combine this individual with `partner` using arithmetic mean.

        An arithmetic mean of two values which are in [self.__min_value, self.__max_value) range is also in the same
        range. Therefore, we don't need to check min and max boundaries after recombination.

        :param RealNumber partner
        :return: RealNumber
        """

        child_amino_acids_length = len(self.__peptide_amino_acids) + len(partner.get_peptide_amino_acids())

        crossover_index_parent_1 = np.random.randint(low=0, high=len(self.__peptide_amino_acids))
        crossover_index_parent_2 = np.random.randint(low=0, high=len(partner.get_peptide_amino_acids()))

        child_amino_acids = self.__peptide_amino_acids[:crossover_index_parent_1] + \
                            partner.get_peptide_amino_acids()[crossover_index_parent_2:]


# Dulčjina peptida i gdje je imp. group
        child_important_groups = {}

        for group in self.__important_groups:
            child_important_groups[group] = []

            for i in range(self.__important_groups[group]):
                if np.random.rand() < 0.5:
                    insert_index = self.__peptide_important_groups[group][i]
                else:
                    insert_index = partner.get_peptide_important_groups()[group][i]

                child_important_groups[group].append(insert_index)







        first_parent = ''.join(self.peptide)
        second_parent = ''.join(partner.peptide)

        child_length = (len(first_parent) + len(second_parent)) // 2 # Ovdje možda ne aritmetička sredina

        random_length_first_parent = np.random.randint(low=0, high=child_length + 1)
        second_parent_length = child_length - random_length_first_parent

        # Mislim da ovo može i bez petlje
        while True:
            if random_length_first_parent > len(first_parent):
                random_length_first_parent -= 1
                second_parent_length += 1
            elif second_parent_length > len(second_parent):
                random_length_first_parent += 1
                second_parent_length -= 1
            else:
                break

        random_start_index_first_parent = np.random.randint(low=0, high=len(first_parent) - random_length_first_parent + 1)
        random_start_index_second_parent = np.random.randint(low=0, high=len(second_parent) - second_parent_length + 1)

        child_sequence = first_parent[random_start_index_first_parent:random_length_first_parent] + \
                         second_parent[random_start_index_second_parent:second_parent_length]

        while True:
            space = 0

            for group in self.important_groups:
                if group not in child_sequence:
                    space += len(group)

            if space == 0:
                break

            # Hmm, ovdje krešemo samo s početka
            for _ in range(space):
                r = np.random.rand()

                if r <= 0.5:
                    child_sequence = child_sequence[1:]
                else:
                    child_sequence = child_sequence[:-1]

        for group in self.important_groups:
            if group not in child_sequence:
                r = np.random.rand()

                if r <= 0.5:
                    first_parent, second_parent = second_parent, first_parent

                i = first_parent.find(group)
                if i == -1:
                    i = second_parent.find(group)
                if i == -1:
                    i = np.random.randint(low=0, high=len(child_sequence))

                child_sequence = child_sequence[0:i] + group + child_sequence[i:]

        indices = []
        lens = []
        for group in self.important_groups:
            indices.append(child_sequence.find(group))
            lens.append(len(group))

        p = []

        for i in range(len(child_sequence)):
            if i in indices:
                p.append(child_sequence[i:lens[i]]) index of lens aaaaaaa
                i += lens[i]
            else:
                p.append(child_sequence[i])






        # Create a new child object.
        child = RealNumber(
            min_value=self.__min_value,
            max_value=self.__max_value,
            min_mutation_value=self.__min_mutation_value,
            max_mutation_value=self.__max_mutation_value,
            mutation_probability=self._mutation_probability
        )

        # Set child value.
        child.real_number = (self.real_number + partner.real_number) / 2

        return child

    def mutate(self):
        """Perform mutation, introduce a slight variation.

        Value is mutated by adding a random number in [self.__min_mutation_value, self.__max_mutation_value)
        range. After the mutation, we must check min and max boundaries.
        """
        mutation_value = np.random.rand() * \
            (self.__max_mutation_value - self.__min_mutation_value) + self.__min_mutation_value

        self.real_number = min(max(self.real_number + mutation_value, self.__min_value), self.__max_value)

    def extract_important_groups(self, initial_peptide, control_mask):
        important_groups = []
        index = 0
        length = 0

        for marker in control_mask:

            if isinstance(marker, list):
                group = []

                for _ in marker:
                    group.append(initial_peptide[index])
                    index += 1

                index -= 1

                important_groups.append(''.join(group))
                length += len(group)

            elif marker == 1:
                important_groups.append(initial_peptide[index])
                length += 1

            index += 1

        return important_groups, length


