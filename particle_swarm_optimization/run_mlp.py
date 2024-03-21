import random
import pandas as pd
import itertools
import time
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('/home/rouf/logger/pso_mlp.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

data_path = "/home/rouf/evo-algo/real_data/PC.csv"
optimization_weights = [
    [34, 33, 33],
] # f1, inference_time, memory
objective_functions = ["max-normalized", "range-normalized", "non-normalized"]

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




def generate_random_position():
    random_param_set = {}
    for key in param_grid:
        random_param_set[key] = random.randint(0, len(param_grid[key]) - 1)
    return random_param_set

def generate_random_velocity(low, high):
    random_velocity = {}
    for key in param_grid:
        random_velocity[key] = random.uniform(low, high)
    return random_velocity

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


# particle swarm algorithm parameters
particle_nums = [5, 10, 15, 20, 25, 30] 
num_iterations = [10, 20, 30, 40, 50]
w_range = (0.5, 0.9)
c1 = 2
c2 = 2
counter = 0



class Particle:
    def __init__(self):
        # (accuracy, time, memory, hidden_layer_size, number_of_hidden_layer, learning_rate)
        self.position = generate_random_position()
        self.velocity = generate_random_velocity(-1, 1)
        self.best_position = self.position
        self.best_fitness = float('inf')

# def calculate_fitness(x): # x is a position
#     global counter
#     counter += 1
#     utilities = calculate_utility_from_position(x)
#     f1 = utilities['f1']
#     memory = utilities['memory']
#     inference_time = utilities['inference_time']
#     x = f1_weight * (f1/data['max']['f1']) + inference_time_weight * (inference_time/data['max']['inference_time']) + memory_weight * (memory/data['max']['memory'])
#     return (1/(1 + x)) if x > 0 else (1 + abs(x))

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

def update_velocity(particle, global_best_position, w):
    for key in particle.velocity:
        r1, r2 = random.random(), random.random()
        cognitive_component = c1 * r1 * (particle.best_position[key] - particle.position[key])
        social_component = c2 * r2 * (global_best_position[key] - particle.position[key])
        particle.velocity[key] = w * particle.velocity[key] + cognitive_component + social_component

def update_position(particle):
    for key in particle.position:
        particle.position[key] = particle.position[key] + particle.velocity[key]
        particle.position[key] = int(round(particle.position[key]))
        
        if particle.position[key] > (len(param_grid[key]) - 1): 
            particle.position[key] = (len(param_grid[key]) - 1)
        if particle.position[key] < 0:
            particle.position[key] = 0

def interpolate_w(iteration_no, total_iteration): 
    return w_range[1] - (((iteration_no + 1)/total_iteration) * (w_range[1] - w_range[0]))

def pso_algorithm(weights, nums_part, nums_iter, obj_fun):
    global counter
    counter = 0
    particles = [Particle() for _ in range(nums_part + 1)]
    global_best_particle = min(particles, key=lambda x: fitness_function(calculate_utility_from_position(x.position), weights, obj_fun))
    counter += nums_part
    global_best_position = global_best_particle.position

    for _ in range(nums_iter):
        for particle in particles:
            fitness = fitness_function(calculate_utility_from_position(particle.position), weights, obj_fun)
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position

            if fitness < fitness_function(calculate_utility_from_position(global_best_position), weights, obj_fun):
                global_best_position = particle.position
        counter += (nums_part + 1) 
        w = interpolate_w(_, nums_iter)
        for particle in particles:
            update_velocity(particle, global_best_position, w)
            update_position(particle)

    counter += 1
    return global_best_position, fitness_function(calculate_utility_from_position(global_best_position), weights, obj_fun)

if __name__ == "__main__":
    # best_position, best_fitness = pso_algorithm()
    # print("Best fitness: ", best_fitness)
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
        opt_fitness, opt_params, opt_utility = optimal_algorithm(optimization_weights[0], objective_fun)
        opt_params = {
            'hidden_layer_sizes': opt_params[0], 
            'activation': opt_params[1],
            'solver': opt_params[2],
            'alpha': opt_params[3],
            'learning_rate': opt_params[4],
            'warm_start': opt_params[5]
        }
        for particle_num in particle_nums:
            for iter_num in num_iterations:
                s_time = time.time() * 1000
                pso_best_individual, pso_best_fitness = pso_algorithm(optimization_weights[0], particle_num, iter_num, objective_fun)
                pso_best_params = get_paramset_from_position(pso_best_individual)
                pso_best_utility = calculate_utility_from_position(pso_best_individual)
                pso_count = counter
                result = [
                    i,
                    optimization_weights[0][0],
                    optimization_weights[0][1],
                    optimization_weights[0][2],
                    objective_fun,
                    particle_num, 
                    iter_num,
                    pso_best_params, 
                    pso_best_utility['idx'],
                    pso_best_fitness,
                    pso_best_utility['f1'],
                    pso_best_utility['inference_time'],
                    pso_best_utility['memory'],
                    pso_count,
                    opt_params,
                    opt_utility['idx'],
                    opt_utility['f1'],
                    opt_utility['inference_time'],
                    opt_utility['memory'],
                    abs(opt_utility['f1'] - pso_best_utility['f1']),
                    abs(opt_utility['inference_time'] - pso_best_utility['inference_time']),
                    abs(opt_utility['memory'] - pso_best_utility['memory'])
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
        'number_of_particle', 
        'number_of_iteration',
        'pso_best_params', 
        'pso_best_param_idx',
        'pso_best_fitness',
        'pso_best_f1',
        'pso_best_inference_time',
        'pso_best_memory',
        'pso_iteration_count',
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
    df.to_csv('pso_PC.csv', index=False)