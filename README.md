# NeuroGA
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

# Start the optimization process and fetch the result.
pareto_fronts = algo.optimize()
```
