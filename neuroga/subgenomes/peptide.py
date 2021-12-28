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
        2. Mrvicu modificiramo početni peptid, pa populacija bude u okolini početnog peptida. (za ovo se
        može mutacija pozavti nekoliko puta)
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
        """Combine this individual with `partner` using single point crossover.

        While combining the two parents, we must ensure a `child` contains all the important amino groups contained in
        `self.__important_groups` in the right amount.

        :param ModifiedPeptide partner
        :return: ModifiedPeptide
        """

        # Crossover points for each parent are approximately in the middle of the peptide sequences.
        crossover_index_parent_1 = len(self.__peptide) // 2
        crossover_index_parent_2 = len(partner.get_peptide()) // 2

        # Merge first part of the first parent, and second part of the second parent.
        child_peptide = self.__peptide[:crossover_index_parent_1] + partner.get_peptide()[crossover_index_parent_2:]

        # When slicing parents and creating a child sequence, there is a high chance of a child sequence having too much
        # or too little instances of specific (or all) important amino groups.

        # Iterate over all important groups in a random order.
        for group in np.random.permutation(list(self.__important_groups.keys())):

            # Check how many instances of the `group` there are in a child peptide.
            count = child_peptide.count(group)

            if self.__important_groups[group] > count:
                # There are `self.__important_groups[group] - count` instances missing. Add them.

                for _ in range(self.__important_groups[group] - count):
                    # Randomly choose a parent.
                    if np.random.rand() < 0.5:
                        parent = self.__peptide
                    else:
                        parent = partner.get_peptide()

                    # Get the positions (indices) of `group` in `parent`.
                    indices = [index for index in range(len(parent)) if parent[index] == group]

                    # Choose a random index from the available ones. This effectively picks one of the groups of type
                    # `group` from `parent`.
                    random_index = indices[np.random.randint(len(indices))]

                    # Since `parent` and `child_peptide` can have different lengths, we want to use relative positioning
                    # to calculate the appropriate insertion index in `child_peptide`.
                    # E. g., if `parent` is of length 10, and the chosen group is situated at index 5, then the same
                    # group should be inserted somewhere around index 50 if `child_peptide` is of length 100.
                    random_index_corrected = int((random_index / (len(parent) - 1)) * len(child_peptide))

                    # Insert `group` at the corrected index.
                    child_peptide.insert(random_index_corrected, group)

            elif count > self.__important_groups[group]:
                # There are `count - self.__important_groups[group]` instances extra. Remove them.

                for _ in range(count - self.__important_groups[group]):
                    # Get the positions (indices) of `group` in `child_peptide`.
                    indices = [index for index in range(len(child_peptide)) if child_peptide[index] == group]

                    # Choose a random index from the available ones. This effectively picks one of the groups of type
                    # `group` from `child_peptide`.
                    random_index = indices[np.random.randint(len(indices))]

                    # Remove the chosen group from `child_peptide`.
                    del child_peptide[random_index]

            # Otherwise, `count` is equal to `self.__important_groups[group]` and no modifications of `child_peptide`
            # are required.

############################################################
        child = ModifiedPeptide(
            initial_peptide=None,
            control_mask=None,
            initial_min_length=0,
            initial_max_length=0,
            amino_acids=None,
            mutation_probability=0.0
        )

        # Set child value.
        child.set_peptide(child_peptide)

        return child

    def mutate(self):
        """Perform mutation, introduce a slight variation.

        There are 4 possible ways to perform a mutation:
        1. Add a random amino acid somewhere in a peptide.
        2. Remove a random amino acid from peptide.
        3. Swap two peptide constituents.
        4. Change amino acid at a random place in a peptide.

        Each time, only a single type of mutation is performed. Each type of mutation has the same probability of
        being chosen.
        """

        random_mutation_choice = np.random.rand()

        if 0 <= random_mutation_choice < 0.25:
            # Add a random amino acid somewhere in a peptide.

            self.__peptide.insert(
                # The range for a random index is `[0, len(self.__peptide) + 1>` because we want the possibility to add
                # amino acids to the end of the sequence, as well as to the start.
                np.random.randint(len(self.__peptide) + 1),
                self.__amino_acids[np.random.randint(len(self.__amino_acids))]
            )

        elif 0.25 <= random_mutation_choice < 0.5:
            # Remove a random amino acid from peptide.

            # First, list indices of peptide constituents which are not important groups (because we mustn't remove any
            # of the important groups).
            indices = [index for index in range(len(self.__peptide)) if len(self.__peptide[index]) == 1]
            random_index = indices[np.random.randint(len(indices))]

            del self.__peptide[random_index]

        elif 0.5 <= random_mutation_choice < 0.75:
            # Swap two peptide constituents.

            index_1 = np.random.randint(len(self.__peptide))
            index_2 = np.random.randint(len(self.__peptide))

            self.__peptide[index_1], self.__peptide[index_2] = self.__peptide[index_2], self.__peptide[index_1]

        else:
            # 0.75 <= random_mutation_choice <= 1:

            # Change amino acid at a random place in a peptide.

            # First, list indices of peptide constituents which are not important groups (because we mustn't change any
            # of the important groups).
            indices = [index for index in range(len(self.__peptide)) if len(self.__peptide[index]) == 1]
            random_index = indices[np.random.randint(len(indices))]

            random_amino_acid = self.__amino_acids[
                np.random.randint(len(self.__amino_acids))
            ]

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
