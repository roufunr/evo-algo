import random
import pandas as pd
from itertools import product

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
particle_nums = 6 
num_iterations = 30
w_range = (0.5, 0.9)
c1 = 2
c2 = 2
counter = 0
f1_weight = 30/100
memory_weight = (-1) * (50/100)
inference_time_weight = (-1) * (20/100)



class Particle:
    def __init__(self):
        # (accuracy, time, memory, hidden_layer_size, number_of_hidden_layer, learning_rate)
        self.position = generate_random_position()
        self.velocity = generate_random_velocity(-1, 1)
        self.best_position = self.position
        self.best_fitness = float('inf')

def calculate_fitness(x): # x is a position
    global counter
    counter += 1
    utilities = calculate_utility_from_position(x)
    f1 = utilities['f1']
    memory = utilities['memory']
    inference_time = utilities['inference_time']
    x = f1_weight * (f1/data['max']['f1']) + inference_time_weight * (inference_time/data['max']['inference_time']) + memory_weight * (memory/data['max']['memory'])
    return (1/(1 + x)) if x > 0 else (1 + abs(x))

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

def interpolate_w(iteration_no): 
    return w_range[1] - (((iteration_no + 1)/num_iterations) * (w_range[1] - w_range[0]))

def pso_algorithm():
    particles = [Particle() for _ in range(particle_nums + 1)]
    global_best_particle = min(particles, key=lambda x: calculate_fitness(x.position))
    global_best_position = global_best_particle.position

    for _ in range(num_iterations):
        for particle in particles:
            fitness = calculate_fitness(particle.position)
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position

            if fitness < calculate_fitness(global_best_position):
                global_best_position = particle.position
        
        w = interpolate_w(_)
        for particle in particles:
            update_velocity(particle, global_best_position, w)
            update_position(particle)

    return global_best_position, calculate_fitness(global_best_position)

if __name__ == "__main__":
    best_position, best_fitness = pso_algorithm()
    print("Best fitness: ", best_fitness)
    best_utility = calculate_utility_from_position(best_position)
    print("Optimal Accuracy: ", round(best_utility['f1'] * 100, 3), "%")
    print("Optimal inference time: ", round(best_utility['inference_time'], 3), "s")
    print("Optimal required memory: ", round(best_utility['memory'], 3), "MiB")
    print("Optimal idx: ", round(best_utility['idx'], 3), "th")
    print("Optimal param: ", get_paramset_from_position(best_position))
    print("Total inference:", counter)