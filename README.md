
# NeuroGA

NeuroGA is a Python library for optimization. It uses NSGA-II [[1]](#1) genetic algorithm and supports complex solutions, that is, a solution's genome can contain variables of different types simultaneously, such as decimal numbers and categorical options. So far, 5 subgenomes are available:
- KerasNN &#8594; Evolves the weights of an arbitrary Keras [[2]](#2) model.
- OrderedNDList &#8594; Searches for the best order of categorical variables in a list without repetition.
- RealNumber &#8594; Defines a decimal number, and the minimum and maximum value that it can take.
- RealNumberSequence &#8594; Defines a sequence of decimal numbers. The advantage of the subgenome over `RealNumber` is the easier definition of multiple variables and the use of NumPy [[3]](#3) matrix operations to accelerate recombination and mutation.
- RealNumberSequenceIndividual &#8594; Complex subgenome, consists of multiple `RealNumber` subgenomes. Ultimately, it works identically to `RealNumberSequence`, however, the goal is to demonstrate how a new subgenome can be created using other subgenomes. 

## Example usage
The optimization process strives to find the global optimums of two fitness function, `evaluate1` and `evaluate2`. The genome consists of 2 decimal numbers, `x` and `y`. The solutions are returned in the form of pareto fronts, where the first pareto front contains the utmost solutions.
```Python
def evaluate1(solution, data):
    # The function reaches its global minimum of 0 for x=10.
    return abs(solution.get_subgenomes()['x'].real_number) - 10

def evaluate2(solution, data):
    # The function reaches its global maximum of 20 for y=0.
    return -abs(solution.get_subgenomes()['y'].real_number) + 20

def on_generation_finish(generation_number, num_generations, pareto_fronts):
    # Print progress.
    print(generation_number,'/',num_generations)


# Algorithm initialization.
algo = NSGAII(
    genome=Genome(
        subgenomes={
            'x': RealNumber(min_value=-500, max_value=500, mutation_probability=0.1),
            'y': RealNumber(min_value=-500, max_value=500)
        }
    ),
    fitness_functions=[
        FitnessFunction(function=evaluate1, function_type=FitnessFunctionType.MIN),
        FitnessFunction(function=evaluate2, function_type=FitnessFunctionType.MAX)
    ],
    population_size=80,
    offspring_size=20,
    num_generations=500,
    num_solutions_tournament=3,
    recombination_probability=1.0,
    mutation_probability_global=0.01,
    data_train=None,
    validate=False,
    data_val=None,
    on_generation_finish_callback=on_generation_finish
)

# Start the optimization process and fetch the results.
pareto_fronts = algo.optimize()
```

## References
<a id="1">[1]</a> Deb, K. et al.,  "A fast and elitist multiobjective genetic algorithm: NSGA-II"
Journal: IEEE Transactions on Evolutionary Computation, vol. 6, num. 2, pages 182-197

<a id="2">[2]</a> Chollet, F. et al.,  "Keras"
URL: https://keras.io

<a id="3">[3]</a> Harris, C. R. et al.,  "Array programming with NumPy"
Journal: Nature, vol. 585, num. 7825, pages 357-362
