import random
import pandas as pd
from itertools import product

data_path = "/home/rouf-linux/evo-algo/real_data/old_PC.csv"

# GA Parameters
population_size = 10                # number of initial data point
number_of_generation = 10           # number of iteration
mutation_rate = 10/100              # percentage of change after crossover (random)
optimal_parent_count = 2            # must be less than population size (at least 1)
optimization_weights = [
    [100, 0, 0],
    [0, 100, 0],
    [0, 0, 100], 
    [10, 90, 0],
    [20, 80, 0],
    [30, 70, 0],
    [40, 60, 0],
    [50, 50, 0],
    [60, 40, 0],
    [70, 30, 0],
    [80, 20, 0],
    [90, 10, 0],
    [10, 0, 90],
    [20, 0, 80],
    [30, 0, 70],
    [40, 0, 60],
    [50, 0, 50],
    [60, 0, 40],
    [70, 0, 30],
    [80, 0, 20],
    [90, 0, 10],
    [0, 10, 90],
    [0, 20, 80],
    [0, 30, 70],
    [0, 40, 60],
    [0, 50, 50],
    [0, 60, 40],
    [0, 70, 30],
    [0, 80, 20],
    [0, 90, 10],
    [34, 33, 33],
    [10, 45, 45],
    [50, 25, 25],
    [45, 10, 45],
    [25, 50, 25],
    [45, 45, 10],
    [25, 25, 50]
]
opjective_functions = ["max-normalized", "range-normalized", "non-normalized"]

param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (150,), (50, 50), (50, 100), (50, 150), (100, 50), (100, 100), (100, 150), (150, 50), (150, 100), (150, 150), (50, 50, 50), (50, 50, 100), (50, 50, 150), (50, 100, 50), (50, 100, 100), (50, 100, 150), (50, 150, 50), (50, 150, 100), (50, 150, 150), (100, 50, 50), (100, 50, 100), (100, 50, 150), (100, 100, 50), (100, 100, 100), (100, 100, 150), (100, 150, 50), (100, 150, 100), (100, 150, 150), (150, 50, 50), (150, 50, 100), (150, 50, 150), (150, 100, 50), (150, 100, 100), (150, 100, 150), (150, 150, 50), (150, 150, 100), (150, 150, 150)], 
    'activation': ['identity','relu', 'tanh'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': [0.01, 0.001, 0.0001],
    'learning_rate': ['constant', 'adaptive', 'invscaling'],
    'warm_start': [True, False]
}

def search_space_init():
    search_dict = {}
    for hidden_layer_sizes, activation, solver, alpha, learning_rate, warm_start in product(*param_grid.values()): # search space initialization
        param_tuple = (hidden_layer_sizes, activation, solver, alpha, learning_rate, warm_start)
        search_dict[param_tuple] = {}
    return search_dict

def load_data():
    search_dict = search_space_init()
    csv_data = pd.read_csv(data_path)
    total_data = len(csv_data)
    max_f1 = -1
    max_memory = -1
    max_inference_time = -1
    min_f1 = float('inf')
    min_memory = float('inf')
    min_inference_time = float('inf')
    for i in range(total_data):
        row = csv_data.iloc[i]
        l1 = row['l1']
        l2 = row['l2']
        l3 = row['l3']
        layers = [l1, l2, l3]
        activation = row['activation']
        solver = row['solver']
        learning_rate = row['learning_rate']
        alpha = row['alpha']
        warm_start = row['warm_start']
        new_layers = []
        for layer in layers:
            if layer > 0:
                new_layers.append(layer)
        
        new_layers = tuple(new_layers)
        key = (new_layers, activation, solver, alpha, learning_rate, warm_start)
        search_dict[key] = {
            'f1': row['f1'],
            'inference_time': row['inference time (ms)'],
            'memory': row['total_addr_space(mem)(MiB)'],
            'idx': row['idx']
        }
        max_f1 = max_f1 if max_f1 > row['f1'] else row['f1']
        max_inference_time = max_inference_time if max_inference_time > row['inference time (ms)'] else row['inference time (ms)']
        max_memory = max_memory if max_memory > row['total_addr_space(mem)(MiB)'] else row['total_addr_space(mem)(MiB)']

        min_f1 = min_f1 if min_f1 < row['f1'] else row['f1']
        min_inference_time = min_inference_time if min_inference_time < row['inference time (ms)'] else row['inference time (ms)']
        min_memory = min_memory if min_memory < row['total_addr_space(mem)(MiB)'] else row['total_addr_space(mem)(MiB)']
    
    data = {
        'search_space': search_dict,
        'max': {
            'f1': max_f1,
            'memory': max_memory,
            'inference_time': max_inference_time
        },
        'min': {
            'f1': min_f1,
            'memory': min_memory,
            'inference_time': min_inference_time
        }
    }
    return data

data = load_data()

def generate_random_position():
    random_param_set = {}
    for key in param_grid:
        random_param_set[key] = random.randint(0, len(param_grid[key]) - 1)
    return random_param_set

def calculate_utility_from_position(position):
    key_tuple = []
    for key in param_grid:
        key_tuple.append(param_grid[key][position[key]])
    key_tuple = tuple(key_tuple)
    return data['search_space'][key_tuple]

def get_paramset_from_position(position):
    paramset = {}
    for key in param_grid:
        paramset[key] = param_grid[key][position[key]]
    return paramset

def fitness_function(utility, objective_function, weights):
    f1 = utility['f1']
    memory = utility['memory']
    inference_time = utility['inference_time']
    f1_weight = weights['f1']
    inference_time_weight = weights['inference_time']
    memory_weight = weights['memory']
    x = 0
    f1_component = 0
    inference_time_component = 0
    memory_component = 0
    if(objective_function == "max-normalized"):
        f1_component = ((f1)/(data['max']['f1'])) * f1_weight
        inference_time_component = ((inference_time)/(data['max']['inference_time'])) * inference_time_weight
        memory_component = ((memory)/(data['max']['memory'])) * memory_weight
    elif(objective_function == "range-normalized"):
        f1_component = ((f1 - data['min']['f1'])/(data['max']['f1'] - data['min']['f1'])) * f1_weight
        inference_time_component = ((inference_time - data['min']['inference_time'])/(data['max']['inference_time'] - data['min']['inference_time'])) * inference_time_weight
        memory_component = ((memory - data['min']['memory'])/(data['max']['memory'] - data['min']['memory'])) * memory_weight
    elif(objective_function == "non-normalized"):
        f1_component = (f1) * f1_weight
        inference_time_component = (inference_time/50) * inference_time_weight
        memory_component = (memory/200) * memory_weight
    x = f1_component - inference_time_component - memory_component
    return (1/(1 + x)) if x > 0 else (1 + abs(x))


def crossover(parents):
    child = {}
    for key in param_grid:
        parent_idx = random.randint(0, len(parents) - 1)
        child[key] = parents[parent_idx][key]
    return child

def mutate(child):
    mutated_child = child.copy()
    for key in param_grid:
        if random.random() < mutation_rate:  # mutation rate
            mutated_child[key] = random.randint(0, len(param_grid[key]) - 1)
    return mutated_child

def genetic_algorithm():
    population = [generate_random_position() for _ in range(population_size)]
    for _ in range(number_of_generation):
        population_fitness = [fitness_function(calculate_utility_from_position(individual)) for individual in population]
        sorted_population = [x for _, x in sorted(zip(population_fitness, population), key=lambda pair: pair[0])]
        parents = sorted_population[:optimal_parent_count]
        children = [crossover(parents) for _ in range(population_size - optimal_parent_count)]
        mutated_children = [mutate(child) for child in children]
        population = parents + mutated_children

    best_individual = min(population, key=lambda x: fitness_function(calculate_utility_from_position(x)))
    best_utility = calculate_utility_from_position(best_individual)

    return best_individual, best_utility
    
    

if __name__ == "__main__":
    best_individual, best_utility = genetic_algorithm()
    print("Optimal Accuracy: ", round(best_utility['f1'] * 100, 3), "%")
    print("Optimal inference time: ", round(best_utility['inference_time'], 3), "s")
    print("Optimal required memory: ", round(best_utility['memory'], 3), "MiB")
    print("Optimal idx: ", round(best_utility['idx'], 3), "th")
    print("Optimal param: ", get_paramset_from_position(best_individual))