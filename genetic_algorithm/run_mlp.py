import random
import pandas as pd
import itertools
import time
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('/home/rouf/logger/gen_mlp.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

data_path = "/home/rouf/evo-algo/real_data/ORIN.csv"

# GA Parameters
population_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]                              # number of initial data point
number_of_generation = [5, 10, 15, 20, 35, 30]                                          # number of iteration
mutation_rate = [5, 10, 15, 20]                                                         # percentage of change after crossover (random)
optimal_parent_percentage = [20, 30, 40, 50]                                            # must be less than population size (at least 1)
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
] # f1, inference_time, memory
objective_functions = ["max-normalized", "range-normalized", "non-normalized"]

# population_sizes = [5, 10, 15]                     # number of initial data point
# number_of_generation = [20]                         # number of iteration
# mutation_rate = [15]                # percentage of change after crossover (random)
# optimal_parent_percentage = [20]   # must be less than population size (at least 1)
# optimization_weights = [
#     # [100, 0, 0],
#     # [0, 100, 0],
#     # [0, 0, 100], 
#     # [10, 90, 0],
#     # [20, 80, 0],
#     # [30, 70, 0],
#     # [40, 60, 0],
#     # [50, 50, 0],
#     # [60, 40, 0],
#     # [70, 30, 0],
#     # [80, 20, 0],
#     # [90, 10, 0],
#     # [10, 0, 90],
#     # [20, 0, 80],
#     # [30, 0, 70],
#     # [40, 0, 60],
#     # [50, 0, 50],
#     # [60, 0, 40],
#     # [70, 0, 30],
#     # [80, 0, 20],
#     # [90, 0, 10],
#     # [0, 10, 90],
#     # [0, 20, 80],
#     # [0, 30, 70],
#     # [0, 40, 60],
#     # [0, 50, 50],
#     # [0, 60, 40],
#     # [0, 70, 30],
#     # [0, 80, 20],
#     # [0, 90, 10],
#     # [34, 33, 33],
#     # [10, 45, 45],
#     # [50, 25, 25],
#     # [45, 10, 45],
#     # [25, 50, 25],
#     # [45, 45, 10],
#     [25, 25, 50]
# ] # f1, inference_time, memory
# objective_functions = ["max-normalized", "range-normalized", "non-normalized"]


ga_param_combinations = list(itertools.product(population_sizes, number_of_generation, mutation_rate, optimal_parent_percentage))
print("Total GA Param combinations ", len(ga_param_combinations))

param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (150,), (50, 50), (50, 100), (50, 150), (100, 50), (100, 100), (100, 150), (150, 50), (150, 100), (150, 150), (50, 50, 50), (50, 50, 100), (50, 50, 150), (50, 100, 50), (50, 100, 100), (50, 100, 150), (50, 150, 50), (50, 150, 100), (50, 150, 150), (100, 50, 50), (100, 50, 100), (100, 50, 150), (100, 100, 50), (100, 100, 100), (100, 100, 150), (100, 150, 50), (100, 150, 100), (100, 150, 150), (150, 50, 50), (150, 50, 100), (150, 50, 150), (150, 100, 50), (150, 100, 100), (150, 100, 150), (150, 150, 50), (150, 150, 100), (150, 150, 150)], 
    'activation': ['identity','relu', 'tanh'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': [0.01, 0.001, 0.0001],
    'learning_rate': ['constant', 'adaptive', 'invscaling'],
    'warm_start': [True, False]
}

param_combinations = list(itertools.product(*param_grid.values()))

def search_space_init():
    search_dict = {}
    for hidden_layer_sizes, activation, solver, alpha, learning_rate, warm_start in itertools.product(*param_grid.values()): # search space initialization
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

def fitness_function(utility, weights, objective_function):
    f1 = utility['f1']
    memory = utility['memory']
    inference_time = utility['inference_time']
    f1_weight = weights[0]/100
    inference_time_weight = weights[1]/100
    memory_weight = weights[2]/100
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

def mutate(child, mutation_rate):
    mutated_child = child.copy()
    for key in param_grid:
        if random.random() < mutation_rate:  # mutation rate
            mutated_child[key] = random.randint(0, len(param_grid[key]) - 1)
    return mutated_child

def genetic_algorithm(population_size, number_of_generation, mutation_rate, optimal_parent_count, optimization_weights, objective_function):
    population = [generate_random_position() for _ in range(population_size)]
    iter_count = 0
    for _ in range(number_of_generation):
        population_fitness = [fitness_function(calculate_utility_from_position(individual), optimization_weights, objective_function) for individual in population]
        iter_count += population_size
        sorted_population = [x for _, x in sorted(zip(population_fitness, population), key=lambda pair: pair[0])]
        parents = sorted_population[:optimal_parent_count]
        children = [crossover(parents) for _ in range(population_size - optimal_parent_count)]
        mutated_children = [mutate(child, mutation_rate) for child in children]
        population = parents + mutated_children

    best_individual = min(population, key=lambda x: fitness_function(calculate_utility_from_position(x), optimization_weights, objective_function))
    iter_count += population_size
    best_utility = calculate_utility_from_position(best_individual)

    return best_individual, best_utility, iter_count
    
def optimal_algorithm(optimization_weights, objective_function):
    opt_fitness = float("inf")
    opt_params = []
    opt_utility = {}
    for params in param_combinations:
        key = []
        for p in params:
            key.append(p)
        key_tuple = tuple(key)
        utility = data['search_space'][key_tuple]
        fitness = fitness_function(utility, optimization_weights, objective_function)
        if fitness < opt_fitness:
            opt_fitness = fitness
            opt_params = params
            opt_utility = utility
    return opt_fitness, opt_params, opt_utility
    

if __name__ == "__main__":
    # best_individual, best_utility, iter_count = genetic_algorithm()
    results = []
    i = 1
    for weights in optimization_weights:
        for objective_fun in objective_functions:
            opt_fitness, opt_params, opt_utility = optimal_algorithm(weights, objective_fun)
            opt_params = {
                'hidden_layer_sizes': opt_params[0], 
                'activation': opt_params[1],
                'solver': opt_params[2],
                'alpha': opt_params[3],
                'learning_rate': opt_params[4],
                'warm_start': opt_params[5]
            }
            for ga_params in ga_param_combinations:
                s_time = time.time() * 1000
                population_size = ga_params[0]
                number_of_generation = ga_params[1]
                mutation_rate = ga_params[2]/100
                optimal_parent_count = round(population_size * (ga_params[3]/100))
                ga_best_individual, ga_best_utility, ga_iter_count = genetic_algorithm(population_size, number_of_generation, mutation_rate, optimal_parent_count, weights, objective_fun)
                ga_best_fitness = fitness_function(ga_best_utility, weights, objective_fun)
                ga_best_params = get_paramset_from_position(ga_best_individual)
                ga_best_param_idx = ga_best_utility['idx']
                result = [
                    i,
                    weights[0],
                    weights[1],
                    weights[2],
                    objective_fun,
                    population_size, 
                    number_of_generation,
                    mutation_rate,
                    optimal_parent_count,
                    ga_best_params, 
                    ga_best_param_idx,
                    ga_best_fitness,
                    ga_best_utility['f1'],
                    ga_best_utility['inference_time'],
                    ga_best_utility['memory'],
                    ga_iter_count,
                    opt_params,
                    opt_utility['idx'],
                    opt_utility['f1'],
                    opt_utility['inference_time'],
                    opt_utility['memory'],
                    abs(opt_utility['f1'] - ga_best_utility['f1']),
                    abs(opt_utility['inference_time'] - ga_best_utility['inference_time']),
                    abs(opt_utility['memory'] - ga_best_utility['memory'])
                ]
                results.append(result)
                i+=1
                e_time = time.time() * 1000
                logger.info(f"{i-1}-> {weights} {objective_fun} {ga_params} DONE & it takes {e_time - s_time} ms")
        
    columns = [
        'idx',
        'f1-weight',
        'inference-time-weight',
        'memory-weight',
        'objective-function',
        'population_size', 
        'number_of_generation',
        'mutation_rate',
        'optimal_parent_count',
        'ga_best_params', 
        'ga_best_param_idx',
        'ga_best_fitness',
        'ga_best_f1',
        'ga_best_inference_time',
        'ga_best_memory',
        'ga_iteration_count',
        'opt_params',
        'opt_param_idx',
        'opt_f1',
        'opt_inference_time',
        'opt_memory',
        'f1_diff',
        'f1_diff',
        'memory_diff'
    ]
    df = pd.DataFrame(results, columns=columns)
    df.to_csv('gen_ORIN.csv', index=False)
    
    
    
    
    # print("Optimal Accuracy: ", round(best_utility['f1'] * 100, 3), "%")
    # print("Optimal inference time: ", round(best_utility['inference_time'], 3), "s")
    # print("Optimal required memory: ", round(best_utility['memory'], 3), "MiB")
    # print("Optimal idx: ", round(best_utility['idx'], 3), "th")
    # print("Optimal param: ", get_paramset_from_position(best_individual))