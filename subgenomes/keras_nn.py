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

    def model_weights_as_vector(self, layers):
        weights_vector = []

        for layer in layers:
            vector = np.reshape(layer, newshape=layer.size)
            weights_vector.extend(vector)

        return np.array(weights_vector)

    def model_weights_as_matrix(self, weights_vector, original_matrix):
        weights_matrix = []

        start = 0
        for _, layer in enumerate(original_matrix):
            layer_weights_shape = layer.shape
            layer_weights_size = layer.size

            layer_weights_vector = weights_vector[start:start + layer_weights_size]
            layer_weights_matrix = np.reshape(layer_weights_vector, newshape=layer_weights_shape)
            weights_matrix.append(layer_weights_matrix)

            start += layer_weights_size

        return weights_matrix

    def recombination(self, partner):
        """Combine this individual with `partner` using arithmetic mean.

        An arithmetic mean of two values which are in [self.__min_weight_value, self.__max_weight_value) range is also
        in the same range. Therefore, we don't need to check min and max boundaries after recombination.

        :param KerasNN partner
        :return: KerasNN
        """
        model_weights_copy = copy.deepcopy(self.model_weights)
        #
        # v1 = self.model_weights_as_vector(model_weights_copy)
        # v2 = self.model_weights_as_vector(partner.model_weights)
        #
        # index = np.random.randint(len(v1))
        #
        # v3 = []
        #
        # v3[:index] = v1[:index]
        # v3[index:] = v2[index:]
        #
        # v3 = np.array(v3)

        for i in range(len(model_weights_copy)):
            mask = np.random.random(model_weights_copy[i].size)
            mask[mask>=0.5]=1.0
            mask[mask!=1.0]=0.0
            mask_inv = np.abs(mask-1.0)

            model_weights_copy[i] = mask*model_weights_copy[i] + mask_inv*partner.model_weights[i]

        return KerasNN(
            model_weights=self.model_weights_as_matrix(v3, model_weights_copy),
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
            random_matrix = np.random.random(self.model_weights[i].shape) * (self.__max_weight_value - self.__min_weight_value) + self.__min_weight_value

            random_matrix_marker = np.random.random(self.model_weights[i].shape)
            random_matrix_marker[random_matrix_marker > self.mutation_probability] = 0.0
            random_matrix_marker[random_matrix_marker != 0.0] = 1.0

            marker_inv = np.abs(random_matrix_marker - 1)

            # Alter the weights and check min and max boundaries.
            self.model_weights[i] = self.model_weights[i] * marker_inv + random_matrix * random_matrix_marker
