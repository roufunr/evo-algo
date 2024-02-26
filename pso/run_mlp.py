import random
import pandas as pd
from itertools import product

param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (50, 100), (100, 50), (100, 100), (50, 50, 50), (50, 50, 100), (50, 100, 50), (50, 100, 100), (100, 50, 50), (100, 50, 100), (100, 100, 50), (100, 100, 100)], 
    'activation': ['identity', 'relu', 'tanh'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': [0.01, 0.001, 0.0001],
    'learning_rate': ['constant', 'adaptive', 'invscaling'],
    'warm_start': [True, False]
}

param_types = {
    'hidden_layer_sizes': 'tuple_list',
    'activation': 'string_list',
    'solver': 'string_list',
    'alpha': 'float_list',
    'learning_rate': 'float_list',
    'warm_start': 'boolean_list'
}

search_dict = {}
for hidden_layer_sizes, activation, solver, alpha, learning_rate, warm_start in product(*param_grid.values()):
    param_tuple = (hidden_layer_sizes, activation, solver, alpha, learning_rate, warm_start)
    search_dict[param_tuple] = {
        'f1': 0,
        'inference_time': 0,
        'memory': 0
    }

print(search_dict)


# ----------------------------------
# synthetic data section ends here


# # particle swarm algorithm parameters
# particle_nums = 10
# num_iterations = 10
# w_range = (0.5, 0.9)
# c1 = 2
# c2 = 2
# counter = 0


# class Particle:
#     def __init__(self, start_range, end_range):
#         # (accuracy, time, memory, hidden_layer_size, number_of_hidden_layer, learning_rate)
#         self.position = [random.randint(start_range, end_range), random.randint(start_range, end_range), random.randint(start_range, end_range)]
#         self.velocity = [random.randint(-1, 1), random.randint(-1, 1), random.randint(-1, 1)]
#         self.best_position = self.position
#         self.best_fitness = float('inf')

# def objective_function(x):
#     global counter
#     counter += 1
#     accuracy, time, memory = data_param_to_metric[(x[0] * 10, x[1], x[2]/1000)]
#     x = (0.34) * (accuracy * 100 * 100) -  (0.33) * time - (0.33) * memory
#     return (1/(1 + x)) if x > 0 else (1 + abs(x))

# def update_velocity(particle, global_best_position, w):
#     for i in range(len(particle.velocity)):
#         r1, r2 = random.random(), random.random()
#         cognitive_component = c1 * r1 * (particle.best_position[i] - particle.position[i])
#         social_component = c2 * r2 * (global_best_position[i] - particle.position[i])
#         particle.velocity[i] = w * particle.velocity[i] + cognitive_component + social_component

# def update_position(particle):
#     for i in range(len(particle.position)):
#         particle.position[i] = particle.position[i] + particle.velocity[i]
#         particle.position[i] = int(round(particle.position[i]))
#         if particle.position[i] > _range_: 
#             particle.position[i] = _range_
#         if particle.position[i] < 1:
#             particle.position[i] = 1

# def interpolate_w(iteration_no): 
#     return w_range[1] - (((iteration_no + 1)/num_iterations) * (w_range[1] - w_range[0]))

# def pso_algorithm():
#     particles = [Particle(1, _range_) for _ in range(particle_nums + 1)]
#     global_best_particle = min(particles, key=lambda x: objective_function(x.position))
#     global_best_position = global_best_particle.position

#     for _ in range(num_iterations):
#         for particle in particles:
#             fitness = objective_function(particle.position)
#             if fitness < particle.best_fitness:
#                 particle.best_fitness = fitness
#                 particle.best_position = particle.position

#             if fitness < objective_function(global_best_position):
#                 global_best_position = particle.position
        
#         w = interpolate_w(_)
#         for particle in particles:
#             update_velocity(particle, global_best_position, w)
#             update_position(particle)

#     return global_best_position, objective_function(global_best_position)

# if __name__ == "__main__":
    # generate_synthetic_data()
    # best_position, best_fitness = pso_algorithm()
    # print("Best fitness: ", best_fitness)
    # accuracy, time, memory = data_param_to_metric[(best_position[0] * 10, best_position[1], best_position[2]/1000)]
    # print("Optimal Accuracy: ", round(accuracy, 3), "%")
    # print("Optimal inference time: ", round(time, 3), "s")
    # print("Optimal required memory: ", round(memory, 3), "MiB")
    # print("FOR")
    # print("hidden_layer_size: ", best_position[0] * 10)
    # print("num_hidden_layer: ", best_position[1])
    # print("learning_rate: ", best_position[2] / 1000)
    # print("total overhead: ", counter)