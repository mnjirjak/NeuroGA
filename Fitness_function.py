from enum import Enum

class Fitness_function_type:
    MIN = 1
    MAX = 2

class Fitness_function:
    def __init__(self, function, type):
        self.function = function
        self.type = type
