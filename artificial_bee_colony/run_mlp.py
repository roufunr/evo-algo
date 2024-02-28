import random
import pandas as pd
from itertools import product
import copy

data_path = "/home/rouf/Documents/code/evo-algo/real_data/old_PC.csv"
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



iteration_nums = 10                                         # you can set any number                                           
food_sources_nums = 10
f1_weight = 30/100
memory_weight = (-1) * (50/100)
inference_time_weight = (-1) * (20/100)
dimensions = len(param_grid)                                # number of parameters
particle_nums = dimensions * food_sources_nums              # multiple of 3 because we have three parameters to be tuned (N)
abandoned_solution_limit = 1
counter = 0

def generate_random_food_source():
    random_param_set = {}
    for key in param_grid:
        random_param_set[key] = random.randint(0, len(param_grid[key]) - 1)
    return random_param_set

def calculate_fitness(x):
    return ((1/(1 + x)) if x > 0 else (1 + abs(x))) 

def objective_function(x):
    global counter
    counter += 1
    utilities = calculate_utility_from_position(x)
    f1 = utilities['f1']
    memory = utilities['memory']
    inference_time = utilities['inference_time']
    x = f1_weight * (f1/data['max']['f1']) + inference_time_weight * (inference_time/data['max']['inference_time']) + memory_weight * (memory/data['max']['memory'])
    return x

def perform_employed_phase(food_sources_data, f_x, fitnesses, trials):
    for i in range(food_sources_nums):
        X = food_sources_data[i]
        choices_idx = [j for j in range(food_sources_nums)]
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
        new_output = objective_function(X_new)
        new_fitness = calculate_fitness(new_output)
        if fitnesses[i] > new_fitness:
            fitnesses[i] = new_fitness
            trials[i] = 0
            f_x[i] = new_output
            food_sources_data[i] = X_new
        else: 
            trials[i] += trials[i]

def perform_onlooker_phase(food_sources_data, f_x, fitnesses, trials):
    i = 0                                   # index of current food source
    k = 0                                   # number of iteration
    l = 0
    sum_of_fitnessess = sum(fitnesses)
    probabilities = [fitness/sum_of_fitnessess for fitness in fitnesses]
    while i < food_sources_nums:
        r = random.random()
        if r < probabilities[i]: 
            X = food_sources_data[i]
            choices_idx = [j for j in range(food_sources_nums)]
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
            new_output = objective_function(X_new)
            new_fitness = calculate_fitness(new_output)
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
        k %= food_sources_nums
        l += 1
        if l >= particle_nums:
            break

        
def perform_scout_phase(food_sources_data, f_x, fitnesses, trials):
    for i in range(food_sources_nums):
        if trials[i] > abandoned_solution_limit:
            food_sources_data[i] = generate_random_food_source()
            f_x[i] = objective_function(food_sources_data[i])
            fitnesses[i] = calculate_fitness(f_x[i])
            trials[i] = 0


def run_abc(): 
    food_sources_data = [generate_random_food_source() for _ in range(food_sources_nums)]
    f_x = [objective_function(x) for x in food_sources_data]            # maximize
    fitnesses = [calculate_fitness(x) for x in f_x]                     # minimize
    trials = [0 for _ in range(food_sources_nums)]
    for _ in range(iteration_nums):
        perform_employed_phase(food_sources_data, f_x, fitnesses, trials)
        perform_onlooker_phase(food_sources_data, f_x, fitnesses, trials)
        perform_scout_phase(food_sources_data, f_x, fitnesses, trials)
    
    max_idx = 0
    max_fx = f_x[0]
    for i in range(food_sources_nums):
        if max_fx < f_x[i]:
            max_fx = f_x[i]
            max_idx = i
    
    return food_sources_data[max_idx] 

if __name__ == "__main__":
    best_position = run_abc()
    best_utility = calculate_utility_from_position(best_position)
    print("Optimal Accuracy: ", round(best_utility['f1'] * 100, 3), "%")
    print("Optimal inference time: ", round(best_utility['inference_time'], 3), "s")
    print("Optimal required memory: ", round(best_utility['memory'], 3), "MiB")
    print("Optimal idx: ", round(best_utility['idx'], 3), "th")
    print("Optimal param: ", get_paramset_from_position(best_position))
    print("Total inference:", counter)
    
