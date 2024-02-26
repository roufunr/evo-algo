import random
import pandas as pd
from itertools import product

data_path = "/home/rouf/Documents/code/evo-algo/real_data/old_PC.csv"

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
    
    data = {
        'search_space': search_dict,
        'max': {
            'f1': max_f1,
            'memory': max_memory,
            'inference_time': max_inference_time
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
    best_param = {}
    for key in param_grid:
        best_param[key] = param_grid[key][position[key]]
    return best_param

def fitness_function(utility):
    f1_weight = 30/100
    memory_weight = (-1) * (50/100)
    inference_time_weight = (-1) * (20/100)
    
    f1 = utility['f1']
    memory = utility['memory']
    inference_time = utility['inference_time']
    
    x = f1_weight * (f1/data['max']['f1']) + inference_time_weight * (inference_time/data['max']['inference_time']) + memory_weight * (memory/data['max']['memory'])
    return (1/(1 + x)) if x > 0 else (1 + abs(x))

def crossover(parent1, parent2):
    child = {}
    for key in param_grid:
        if random.random() < 0.5:
            child[key] = parent1[key]
        else:
            child[key] = parent2[key]
    return child

def mutate(child):
    mutated_child = child.copy()
    for key in param_grid:
        if random.random() < 0.1:  # mutation rate
            mutated_child[key] = random.randint(0, len(param_grid[key]) - 1)
    return mutated_child

def genetic_algorithm():
    population_size = 6
    num_generations = 10
    population = [generate_random_position() for _ in range(population_size)]

    for _ in range(num_generations):
        population_fitness = [fitness_function(calculate_utility_from_position(individual)) for individual in population]
        sorted_population = [x for _, x in sorted(zip(population_fitness, population), key=lambda pair: pair[0])]

        parents = sorted_population[:2]
        children = [crossover(parents[0], parents[1]) for _ in range(population_size - 2)]
        mutated_children = [mutate(child) for child in children]

        population = parents + mutated_children

    best_individual = min(population, key=lambda x: fitness_function(calculate_utility_from_position(x)))
    best_utility = calculate_utility_from_position(best_individual)
    
    print("Optimal Accuracy: ", round(best_utility['f1'] * 100, 3), "%")
    print("Optimal inference time: ", round(best_utility['inference_time'], 3), "s")
    print("Optimal required memory: ", round(best_utility['memory'], 3), "MiB")
    print("Optimal idx: ", round(best_utility['idx'], 3), "th")
    print("Optimal param: ", get_paramset_from_position(best_individual))

if __name__ == "__main__":
    genetic_algorithm()
