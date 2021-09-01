from Subgenome import Subgenome
import numpy as np


class RealNumber(Subgenome):
    def __init__(self, min_value=0.0, max_value=1.0):#, recomb_p=0.8, mut_p = 0.05):
        self.min_value = min_value
        self.max_value = max_value

        self.real_number = 0.0

    def randomize(self):
        self.real_number = np.random.rand() * (self.max_value - self.min_value) + self.min_value

    def recombination(self, partner):
        return min(max((self.real_number + partner.real_number) / 2, self.min_value), self.max_value)

    def mutate(self):
        self.real_number += min(max((1.0 if np.random.rand() <= 0.5 else -1.0) * (np.random.rand() / 10.0) * self.real_number, self.min_value), self.max_value)
