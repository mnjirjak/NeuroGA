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
        """Combine this individual with `partner`.

        The weights of two neural network are combined by randomly taking a portion of the weights from the first
        network and then the remaining weights from the second network.

        :param KerasNN partner
        :return: KerasNN
        """
        model_weights_copy = copy.deepcopy(self.model_weights)

        for i in range(len(model_weights_copy)):
            # Create a mask of the same shape as `model_weights_copy[i]`.
            mask = np.random.random(model_weights_copy[i].size)

            # Make approximately 50% of the `mask` values 1.0, and 50% 0.0.
            mask[mask >= 0.5] = 1.0
            mask[mask != 1.0] = 0.0

            # Invert the `mask`. `mask_inv` will have 1.0 where `mask` has 0.0 and vice versa.
            mask_inv = np.abs(mask - 1.0)

            # Take a portion of the weights from `model_weights_copy[i]` and the rest from `partner.model_weights[i]`.
            model_weights_copy[i] = model_weights_copy[i] * mask + partner.model_weights[i] * mask_inv

        return KerasNN(
            model_weights=model_weights_copy,
            min_weight_value=self.__min_weight_value,
            max_weight_value=self.__max_weight_value,
            mutation_probability=self.mutation_probability
        )

    def mutate(self):
        """Perform mutation, introduce a slight variation.

        Neural network weights are mutated by replacing `self.mutation_probability` random weights with random values
        in [self.__min_weight_value, self.__max_weight_value) range.
        """
        for i in range(len(self.model_weights)):
            # Generate a tensor of random numbers.
            random_weights = np.random.random(self.model_weights[i].shape) * \
                            (self.__max_weight_value - self.__min_weight_value) + self.__min_weight_value

            # Create a mask of the same shape as `model_weights_copy[i]`.
            mask = np.random.random(self.model_weights[i].shape)

            # Make approximately `self.mutation_probability` values in `mask` equal to 1.0, and the rest equal to 0.0.
            mask[mask > self.mutation_probability] = 0.0
            mask[mask != 0.0] = 1.0

            # Invert the `mask`. `mask_inv` will have 1.0 where `mask` has 0.0 and vice versa.
            mask_inv = np.abs(mask - 1.0)

            # Preserve the weights denoted by 1.0s in `mask_inv`, and replace the rest with values from
            # `random_weights`. This assigns random values to approximately `self.mutation_probability` percent of the
            # values in `self.model_weights[i]`.

            self.model_weights[i] = self.model_weights[i] * mask_inv + random_weights * mask_inv
