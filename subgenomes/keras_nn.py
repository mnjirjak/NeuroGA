import copy
from subgenome import Subgenome
import numpy as np


class KerasNN(Subgenome):
    """A Keras neural network whose weights are in [min_weight_value, max_weight_value] range."""
    def __init__(self, model_weights=None, min_weight_value=-1.0, max_weight_value=1.0, mutation_probability=None):
        """
        :param List[numpy.ndarray] model_weights: A list of layer weights.
        :param float min_weight_value: Minimum weight value.
        :param float max_weight_value: Maximum weight value.
        :param float mutation_probability: Local mutation probability. If not specified, global mutation probability
                                     will be used.
        """
        super().__init__(mutation_probability)

        self.__min_weight_value = min_weight_value
        self.__max_weight_value = max_weight_value

        # `self.model_weights` is a list of numpy.ndarrays
        self.model_weights = model_weights

    def randomize(self):
        """Randomly assign weight values in [self.__min_weight_value, self.__max_weight_value) range."""
        for i in range(len(self.model_weights)):
            self.model_weights[i] = np.random.random(self.model_weights[i].shape) * \
                                    (self.__max_weight_value - self.__min_weight_value) + self.__min_weight_value

    def recombination(self, partner):
        """Combine this individual with `partner` using arithmetic mean.

        An arithmetic mean of two values which are in [self.__min_weight_value, self.__max_weight_value) range is also
        in the same range. Therefore, we don't need to check min and max boundaries after recombination.

        :param KerasNN partner
        :return: KerasNN
        """
        model_weights_copy = copy.deepcopy(self.model_weights)

        for i in range(len(model_weights_copy)):
            model_weights_copy[i] = (model_weights_copy[i] + partner.model_weights[i]) / 2

        return KerasNN(
            model_weights=model_weights_copy,
            min_weight_value=self.__min_weight_value,
            max_weight_value=self.__max_weight_value,
            mutation_probability=self.mutation_probability
        )

    def mutate(self):
        """Perform mutation, introduce a slight variation.

        Each weight is mutated by adding or subtracting a small number. This is done using numpy matrix operations since
        it is waaaaay faster. After the mutation, we must check min and max boundaries.
        """
        for i in range(len(self.model_weights)):
            # Generate a tensor of random numbers.
            random_matrix = (np.random.random(self.model_weights[i].shape) - 0.5) * \
                            (self.__max_weight_value - self.__min_weight_value) * self.mutation_probability

            # Generate tensors containing minimum and maximum values.
            minima = np.ones(self.model_weights[i].shape) * self.__min_weight_value
            maxima = np.ones(self.model_weights[i].shape) * self.__max_weight_value

            # Alter the weights and check min and max boundaries.
            self.model_weights[i] = np.minimum(np.maximum(self.model_weights[i] + random_matrix, minima), maxima)
