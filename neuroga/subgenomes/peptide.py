from neuroga.subgenome import Subgenome
import numpy as np


class ModifiedPeptide(Subgenome):
    """A real number bound by min and max values."""

    def __init__(self,
                 initial_peptide="",
                 control_mask=None,
                 initial_min_length=5,
                 initial_max_length=50,
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

        self.__initial_min_length = initial_min_length
        self.__initial_max_length = initial_max_length

        self.__amino_acids = amino_acids

        self.__important_groups = []

        if control_mask is not None:
            self.__important_groups, self.mandatory_length = \
                self.extract_important_groups(initial_peptide, control_mask)

        # Min i max length su zapravo ovisni o ovoj mandatory_length

        self.__peptide = []

    def randomize(self):
        """Ovdje možemo imati dva pristupa:
        1. Na random inicijaliziramo peptide koji sadrže bite skupine.
        2. Mrvicu modificiramo početni peptid, pa populacija bude u okolini početnog peptida. (za ovo se može mutacija pozavti nekoliko puta)
        """

        num_amino_acids = np.random.randint(
            low=self.__initial_min_length - self.mandatory_length,
            high=self.__initial_max_length - self.mandatory_length + 1
        )

        #Ovdje možeš maknuti low i ostaviti samo high i bez navođenja
        amino_acid_indices = np.random.randint(
            low=0,
            high=len(self.__amino_acids),
            size=num_amino_acids
        )

        self.__peptide = [self.__amino_acids[amino_acid_index] for amino_acid_index in amino_acid_indices]

        # Ovdje će biti dict za important groups

        self.__peptide += \
            [group for group in self.__important_groups for _ in range(self.__important_groups[group])]

        self.__peptide = list(np.random.permutation(self.__peptide))

    def recombination(self, partner):
        """Combine this individual with `partner` using arithmetic mean.

        An arithmetic mean of two values which are in [self.__min_value, self.__max_value) range is also in the same
        range. Therefore, we don't need to check min and max boundaries after recombination.

        :param RealNumber partner
        :return: RealNumber
        """
        crossover_index_parent_1 = len(self.__peptide) // 2
        crossover_index_parent_2 = len(partner.get_peptide()) // 2

        child_peptide = self.__peptide[:crossover_index_parent_1] + partner.get_peptide()[crossover_index_parent_2:]

        # Što ako su important grupe jedna aminokiselina, onda imamo problem

        # Dodaj grupe koje fale
        # možda na random grupe staviti
        for group in self.__important_groups:
            count = child_peptide.count(group)

            for i in range(self.__important_groups[group] - count):
                if np.random.rand() < 0.5:
                    r = self.__peptide
                else:
                    r = partner.get_peptide()

                indices = [i for i in range(len(r)) if r[i] == group]

                random_index = indices[np.random.randint(low=0, high=len(indices))]
                random_index_modified = int(((random_index + 1) / len(r)) * len(child_peptide))

                # Napraviti da se može i na kraj insertati
                child_peptide.insert(random_index_modified, group)

        # Makni grupe koje su viška
        # možda na random grupe staviti
        for group in self.__important_groups:
            count = child_peptide.count(group)

            # Ovdje ig_dict[group] nes mije biti veći od counta, čini mi se.
            for i in range(count - self.__important_groups[group]):
                indices = [i for i in range(len(child_peptide)) if child_peptide[i] == group]

                random_index = indices[np.random.randint(low=0, high=len(indices))]

                del child_peptide[random_index]




        c = ModifiedPeptide(
            initial_peptide=None,
            control_mask=None,
            initial_min_length=0,
            initial_max_length=0,
            amino_acids=None,
            mutation_probability=0.0
        )

        # Set child value.
        c.set_peptide(child_peptide)

        return c

    def mutate(self):
        """Perform mutation, introduce a slight variation.

        Value is mutated by adding a random number in [self.__min_mutation_value, self.__max_mutation_value)
        range. After the mutation, we must check min and max boundaries.
        """


        # 1. mutacija, dodaj random amino negdje

        # Napraviti da se može i na kraj insertati
        self.__peptide.insert(
            np.random.randint(
                low=0,
                high=len(self.__peptide)
            ),
            self.__amino_acids[np.random.randint(
                low=0,
                high=len(self.__amino_acids)
            )]
        )

        # 2. mutacija, makni amino random od nekud
        indices = [i for i in range(len(self.__peptide)) if len(self.__peptide[i]) == 1]

        random_index = indices[np.random.randint(
            low=0,
            high=len(indices)
        )]

        del self.__peptide[random_index]

        # 3. mutacija, zamijeni poredak 2 člana
        i1 = np.random.randint(
            low=0,
            high=len(self.__peptide)
        )

        i2 = np.random.randint(
            low=0,
            high=len(self.__peptide)
        )

        self.__peptide[i1], self.__peptide[i2] = self.__peptide[i2], self.__peptide[i1]

        # 4. mutacija, promijeni aminokis na nekom mjestu
        indices = [i for i in range(len(self.__peptide)) if len(self.__peptide[i]) == 1]

        random_index = indices[np.random.randint(
            low=0,
            high=len(indices)
        )]

        random_amino_acid = self.__amino_acids[
            np.random.randint(
                low=0,
                high=len(self.__amino_acids)
            )]

        self.__peptide[random_index] = random_amino_acid

    def extract_important_groups(self, initial_peptide, control_mask):
        """Extract which important amino groups, and how many of them, should be present in a peptide.

        :param str initial_peptide: Initial amino acid sequence that should be modified.
        :param List[int, List[int]] control_mask: A mask indicating important amino groups in a peptide. Individual
        amino acids are indicated by a 1, while groups of amino acids are indicated by a `List`.

        :return:
            - dict important_groups: A dictionary that contains the important groups and their count.
            - int length: Sums the length of all the important groups.
        """
        important_groups = {}
        index = 0
        length = 0

        # Loop though the members of the `control_mask`.
        for marker in control_mask:

            # If a member is and instance of a `List`, all the member of that list constitute a single important group,
            # regardless of the values of the members.
            if isinstance(marker, list):
                group = []

                # Add amino acids to an array.
                for _ in marker:
                    group.append(initial_peptide[index])
                    index += 1

                # We need to decrement `index` here to ensure `index += 1` at the end of the iteration sets `index` to
                # the member after the current group.
                index -= 1

                # Create a single string from `group` array.
                group_string = ''.join(group)

                # Add the group to the `important_groups` dictionary.
                if group_string in important_groups:
                    important_groups[group_string] += 1
                else:
                    important_groups[group_string] = 1

                # Increase the length of the important groups.
                length += len(group)

            # If a member is not a `List`, and is equal to 1, add it to important groups.
            elif marker == 1:
                # A '*' is added to easily differentiate individual amino acids from important groups containing a
                # single amino acid. This is beneficial during later stages, such as recombination and mutation, while
                # ensuring the correct amount of important amino groups is present in a peptide.
                single_amino_group = initial_peptide[index] + "*"

                # Add the group to the `important_groups` dictionary.
                if single_amino_group in important_groups:
                    important_groups[single_amino_group] += 1
                else:
                    important_groups[single_amino_group] = 1

                # Increase the length of the important groups.
                length += 1

            index += 1

        return important_groups, length
