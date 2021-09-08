import copy
from subgenome import Subgenome
import numpy as np
import keras


class KerasNN(Subgenome):
    """A real number bound by min and max values."""
    def __init__(self, model_weights=None, min_weight_value=-1.0, max_weight_value=1.0, mutation_probability=None):
        """
        :param float min_value: Minimum value of the number.
        :param float max_value: Maximum value of the number.
        :param float value: Current value of the number.
        :param float mutation_probability: Local mutation probability. If not specified, global mutation probability
                                           will be used.
        """
        super().__init__(mutation_probability)

        self.__min_weight_value = min_weight_value
        self.__max_weight_value = max_weight_value

        self.model_weights = model_weights

    def randomize(self):
        """Randomly assign value in [self.min_value, self.max_value) range."""
        for i in range(len(self.model_weights)):
            self.model_weights[i] = np.random.random(self.model_weights[i].shape) * (self.__max_weight_value - self.__min_weight_value) + self.__min_weight_value

    def recombination(self, partner):
        """Combine this individual with `partner` using arithmetic mean.

        While combining them, we must respect min and max boundaries.

        :param RealNumber partner
        :return: RealNumber
        """
        model_weights_copy = copy.deepcopy(self.model_weights)

        for i in range(len(model_weights_copy)):
            # Oba će biti u rasponu sigurno, staviti komentar
            model_weights_copy[i] = (model_weights_copy[i] + partner.model_weights[i]) / 2

        return KerasNN(
            min_weight_value=self.__min_weight_value,
            max_weight_value=self.__max_weight_value,
            model_weights=model_weights_copy,
            mutation_probability=self.mutation_probability
        )

    def mutate(self):
        """Perform mutation, introduce a slight variation.

        Value is mutated by adding or subtracting a small number.

        Two key steps:
        1. Select sign (+ or -).
        2. Select the degree of change with respect to `self.mutation_probability` and `self.real_number`.
        """
        for i in range(len(self.model_weights)):
            random_matrix = (np.random.random(self.model_weights[i].shape) - 0.5) * (self.__max_weight_value - self.__min_weight_value) * self.mutation_probability
            minima = np.ones(self.model_weights[i].shape) * self.__min_weight_value
            maxima = np.ones(self.model_weights[i].shape) * self.__max_weight_value
            self.model_weights[i] = np.minimum(np.maximum(self.model_weights[i] + random_matrix, minima), maxima)

    # def __deepcopy__(self, memodict=None):
    #     model_copy = keras.models.clone_model(self.model)
    #     model_copy.set_weights(self.model.get_weights())
    #
    #     deep_copy = KerasNN(
    #         model=model_copy,
    #         min_weight_value=self.__min_weight_value,
    #         max_weight_value=self.__max_weight_value,
    #         mutation_probability=self.mutation_probability
    #     )
    #
    #     return deep_copy