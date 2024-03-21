import random
import pandas as pd
import itertools
import time
import logging
import copy
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('/home/rouf/logger/abc_mlp.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

data_path = "/home/rouf/evo-algo/real_data/ORIN.csv"
optimization_weights = [
    [34, 33, 33],
] # f1, inference_time, memory
objective_functions = ["max-normalized", "range-normalized", "non-normalized"]
counter = 0


# excluding 150
# param_grid = {
#     'hidden_layer_sizes': [(50,), (100,), (50, 50), (50, 100), (100, 50), (100, 100), (50, 50, 50), (50, 50, 100), (50, 100, 50), (50, 100, 100), (100, 50, 50), (100, 50, 100), (100, 100, 50), (100, 100, 100)], 
#     'activation': ['identity', 'relu', 'tanh'],
#     'solver': ['sgd', 'adam', 'lbfgs'],
#     'alpha': [0.01, 0.001, 0.0001],
#     'learning_rate': ['constant', 'adaptive', 'invscaling'],
#     'warm_start': [True, False]
# }


# including 150
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


def calculate_utility_from_position(position):
    key_tuple = []
    for key in param_grid:
        # print(position[key])
        key_tuple.append(param_grid[key][position[key]])
    key_tuple = tuple(key_tuple)
    return data['search_space'][key_tuple]

def get_paramset_from_position(position):
    best_param = {}
    for key in param_grid:
        best_param[key] = param_grid[key][position[key]]
    return best_param



iteration_nums = [10, 20, 30, 40, 50]                                         # you can set any number                                           
food_sources_nums = [5, 10, 15, 20, 25, 30]
dimensions = len(param_grid)                                # number of parameters

abandoned_solution_limit = 1
counter = 0

def generate_random_food_source():
    random_param_set = {}
    for key in param_grid:
        random_param_set[key] = random.randint(0, len(param_grid[key]) - 1)
    return random_param_set

def objective(utility, weights, objective_function):
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
    return x

def fitness(x):
    return (1/(1 + x)) if x > 0 else (1 + abs(x))

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
        f = fitness(objective(utility, optimization_weights, objective_function))
        if f < opt_fitness:
            opt_fitness = f
            opt_params = params
            opt_utility = utility
    return opt_fitness, opt_params, opt_utility


def perform_employed_phase(fs_nums, food_sources_data, f_x, fitnesses, trials, weights,obj_func):
    global counter
    for i in range(fs_nums):
        X = food_sources_data[i]
        choices_idx = [j for j in range(fs_nums)]
        choices_idx.remove(i)
        Xp = food_sources_data[random.choice(choices_idx)]
        dim_key_choice = random.choice(list(param_grid.keys()))
        phi = random.uniform(1/len(param_grid[dim_key_choice]), 1)
        x_new = X[dim_key_choice] + phi * (X[dim_key_choice] - Xp[dim_key_choice])
        x_new = round(x_new)
        if x_new < 0: 
            x_new = 0
        if x_new > (len(param_grid[dim_key_choice]) - 1): 
            x_new = (len(param_grid[dim_key_choice]) - 1)
        X_new = copy.deepcopy(X)
        X_new[dim_key_choice] = x_new
        new_output = objective(calculate_utility_from_position(X_new), weights, obj_func)
        counter += 1
        new_fitness = fitness(new_output)
        if fitnesses[i] > new_fitness:
            fitnesses[i] = new_fitness
            trials[i] = 0
            f_x[i] = new_output
            food_sources_data[i] = X_new
        else: 
            trials[i] += trials[i]

def perform_onlooker_phase(fs_nums, food_sources_data, f_x, fitnesses, trials, weights, obj_func):
    global counter
    i = 0                                   # index of current food source
    k = 0                                   # number of iteration
    l = 0
    sum_of_fitnessess = sum(fitnesses)
    probabilities = [fitness/sum_of_fitnessess for fitness in fitnesses]
    while i < fs_nums:
        r = random.random()
        if r < probabilities[i]: 
            X = food_sources_data[i]
            choices_idx = [j for j in range(fs_nums)]
            choices_idx.remove(i)
            Xp = food_sources_data[random.choice(choices_idx)]
            dim_key_choice = random.choice(list(param_grid.keys()))
            phi = random.uniform(1/len(param_grid[dim_key_choice]), 1)
            x_new = X[dim_key_choice] + phi * (X[dim_key_choice] - Xp[dim_key_choice])
            x_new = round(x_new)
            if x_new < 0: 
                x_new = 0
            if x_new > (len(param_grid[dim_key_choice]) - 1): 
                x_new = (len(param_grid[dim_key_choice]) - 1)
            X_new = copy.deepcopy(X)
            X_new[dim_key_choice] = x_new
            new_output = objective(calculate_utility_from_position(X_new), weights, obj_func)
            counter += 1
            new_fitness = fitness(new_output)
            if fitnesses[i] > new_fitness:
                fitnesses[i] = new_fitness
                trials[i] = 0
                f_x[i] = new_output
                food_sources_data[i] = X_new
            else: 
                trials[i] += trials[i]
            probabilities = [fitness/sum_of_fitnessess for fitness in fitnesses]
            i += 1
        k += 1
        k %= fs_nums
        l += 1
        particle_nums = dimensions * fs_nums              
        if l >= particle_nums:
            break

        
def perform_scout_phase(fs_nums, food_sources_data, f_x, fitnesses, trials, weights, obj_func):
    global counter
    for i in range(fs_nums):
        if trials[i] > abandoned_solution_limit:
            food_sources_data[i] = generate_random_food_source()
            f_x[i] = objective(calculate_utility_from_position(food_sources_data[i]), weights, obj_func)
            counter += 1
            fitnesses[i] = fitness(f_x[i])
            trials[i] = 0


def run_abc(fsources_nums,iter_nums, weights, obj_func): 
    global counter
    counter = 0

    food_sources_data = [generate_random_food_source() for _ in range(fsources_nums)]
    f_x = [objective(calculate_utility_from_position(x), weights, obj_func) for x in food_sources_data]            # maximize
    counter += fsources_nums
    fitnesses = [fitness(x) for x in f_x]                     # minimize
    trials = [0 for _ in range(fsources_nums)]
    for _ in range(iter_nums):
        perform_employed_phase(fsources_nums, food_sources_data, f_x, fitnesses, trials, weights, obj_func)
        perform_onlooker_phase(fsources_nums, food_sources_data, f_x, fitnesses, trials, weights, obj_func)
        perform_scout_phase(fsources_nums, food_sources_data, f_x, fitnesses, trials, weights, obj_func)
    
    max_idx = 0
    max_fx = f_x[0]
    for i in range(fsources_nums):
        if max_fx < f_x[i]:
            max_fx = f_x[i]
            max_idx = i
    
    return food_sources_data[max_idx] 

if __name__ == "__main__":
    # best_position = run_abc()
    # best_utility = calculate_utility_from_position(best_position)
    # print("Optimal Accuracy: ", round(best_utility['f1'] * 100, 3), "%")
    # print("Optimal inference time: ", round(best_utility['inference_time'], 3), "s")
    # print("Optimal required memory: ", round(best_utility['memory'], 3), "MiB")
    # print("Optimal idx: ", round(best_utility['idx'], 3), "th")
    # print("Optimal param: ", get_paramset_from_position(best_position))
    # print("Total inference:", counter)

    results = []
    i = 1
    for objective_fun in objective_functions:
        weights = optimization_weights[0]
        opt_fitness, opt_params, opt_utility = optimal_algorithm(weights, objective_fun)
        opt_params = {
            'hidden_layer_sizes': opt_params[0], 
            'activation': opt_params[1],
            'solver': opt_params[2],
            'alpha': opt_params[3],
            'learning_rate': opt_params[4],
            'warm_start': opt_params[5]
        }
        for fs_nums in food_sources_nums:
            for iter_num in iteration_nums:
                s_time = time.time() * 1000
                abc_best_individual = run_abc(fs_nums, iter_num, weights, objective_fun)
                abc_best_fitness = fitness(objective(calculate_utility_from_position(abc_best_individual), weights, objective_fun))
                abc_best_params = get_paramset_from_position(abc_best_individual)
                abc_best_utility = calculate_utility_from_position(abc_best_individual)
                abc_count = counter
                result = [
                    i,
                    weights[0],
                    weights[1],
                    weights[2],
                    objective_fun,
                    fs_nums, 
                    iter_num,
                    abc_best_params, 
                    abc_best_utility['idx'],
                    abc_best_fitness,
                    abc_best_utility['f1'],
                    abc_best_utility['inference_time'],
                    abc_best_utility['memory'],
                    abc_count,
                    opt_params,
                    opt_utility['idx'],
                    opt_utility['f1'],
                    opt_utility['inference_time'],
                    opt_utility['memory'],
                    abs(opt_utility['f1'] - abc_best_utility['f1']),
                    abs(opt_utility['inference_time'] - abc_best_utility['inference_time']),
                    abs(opt_utility['memory'] - abc_best_utility['memory'])
                ]
                results.append(result)
                i+=1
                e_time = time.time() * 1000
                print(f"{i-1}-> {optimization_weights[0]} {objective_fun} DONE & it takes {e_time - s_time} ms")
        
    columns = [
        'idx',
        'f1-weight',
        'inference-time-weight',
        'memory-weight',
        'objective-function',
        'number_of_food_sources', 
        'number_of_iteration',
        'abc_best_params', 
        'abc_best_param_idx',
        'abc_best_fitness',
        'abc_best_f1',
        'abc_best_inference_time',
        'abc_best_memory',
        'abc_iteration_count',
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
    df.to_csv('abc_ORIN.csv', index=False)
    
