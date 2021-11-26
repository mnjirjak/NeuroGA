from enum import Enum


class FitnessFunctionType(Enum):
    """Enumerates two fitness function types.

    MIN - FF we want to minimize.
    MAX - FF we want to maximize.
    """
    MIN = 3
    MAX = 2


class FitnessFunction:
    """A wrapper around a fitness function. Used to define function type."""
    def __init__(self, function, function_type):
        self.function = function
        self.function_type = function_type
