import random
_range_ = 100
hidden_layer_sizes = [x * 10 for x in range(1, _range_ + 1)]
number_of_hidden_layers = [x for x in range(1, _range_ + 1)]
learning_rates = [x/1000 for x in range(1, _range_ + 1)]
particle_nums = 100
data_param_to_metric = {}
data_metric_to_param = {}

def generate_synthetic_data():
    global data_param_to_metric
    global data_metric_to_param
    min = 1
    max = 0
    min_time = 100000
    max_time = 0
    min_mem = 100000
    max_mem = 0
    for hidden_layer_size in hidden_layer_sizes:
        for number_of_hidden_layer in number_of_hidden_layers:
            for learning_rate in learning_rates:
                accuracy = abs(1 - ((abs(705 - hidden_layer_size)/750) * (abs(60.5 - number_of_hidden_layer)/64) * (abs(0.0525 - learning_rate)/0.049)))
                time = (((hidden_layer_size * number_of_hidden_layer)/learning_rate)/(100 * 1000 / 0.001)) * (100 * 1000)
                memory = hidden_layer_size * number_of_hidden_layer
                data_param_to_metric[(hidden_layer_size, number_of_hidden_layer, learning_rate)] = (accuracy, time, memory)
                data_metric_to_param[(accuracy, time, memory, hidden_layer_size, number_of_hidden_layer, learning_rate)] = (hidden_layer_size, number_of_hidden_layer, learning_rate)
                if accuracy < min :
                    min = accuracy
                if accuracy > max:
                    max = accuracy
                if time < min_time :
                    min_time = time
                if time > max_time :
                    max_time = time
                if memory < min_mem :
                    min_mem = memory
                if memory > max_mem :
                    max_mem = memory
    print('min-accuracy:', min, 'max-accuracy:', max, 'min-inference-time:', min_time,'max-inference-time:', max_time,'min-memory:', min_mem, 'max-memory:', max_mem)
    return data_metric_to_param

class Particle:
    def __init__(self, start_range, end_range):
        # (accuracy, time, memory, hidden_layer_size, number_of_hidden_layer, learning_rate)
        self.position = [random.randint(start_range, end_range), random.randint(start_range, end_range), random.randint(start_range, end_range)]
        self.velocity = [random.randint(-1, 1), random.randint(-1, 1), random.randint(-1, 1)]
        self.best_position = self.position
        self.best_fitness = float('inf')

def objective_function(x):
    accuracy, time, memory = data_param_to_metric[(x[0] * 10, x[1], x[2]/1000)]
    return (time * memory) / (accuracy * accuracy * accuracy)



def update_velocity(particle, global_best_position, w=0.5, c1=1.5, c2=1.5):
    for i in range(len(particle.velocity)):
        r1, r2 = random.random(), random.random()
        cognitive_component = c1 * r1 * (particle.best_position[i] - particle.position[i])
        social_component = c2 * r2 * (global_best_position[i] - particle.position[i])
        particle.velocity[i] = w * particle.velocity[i] + cognitive_component + social_component
        

def update_position(particle):
    for i in range(len(particle.position)):
        particle.position[i] = particle.position[i] + particle.velocity[i]
        particle.position[i] = int(round(particle.position[i]))
        if particle.position[i] > _range_: 
            particle.position[i] = _range_
        if particle.position[i] < 1:
            particle.position[i] = 1
        

def pso_algorithm(num_iterations):
    particles = [Particle(1, _range_) for _ in range(particle_nums + 1)]
    global_best_particle = min(particles, key=lambda x: objective_function(x.position))
    global_best_position = global_best_particle.position

    for _ in range(num_iterations):
        for particle in particles:
            fitness = objective_function(particle.position)
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position

            if fitness < objective_function(global_best_position):
                global_best_position = particle.position

        for particle in particles:
            update_velocity(particle, global_best_position)
            update_position(particle)

    return global_best_position, objective_function(global_best_position)

if __name__ == "__main__":
    generate_synthetic_data()
    num_iterations = 10000
    best_position, best_fitness = pso_algorithm(num_iterations)
    print("Best fitness: ", best_fitness)
    accuracy, time, memory = data_param_to_metric[(best_position[0] * 10, best_position[1], best_position[2]/1000)]
    print("Optimal Accuracy: ", accuracy)
    print("Optimal inference time: ", time)
    print("Optimal required memory: ", memory)
    print("FOR")
    print("hidden_layer_size: ", best_position[0] * 10)
    print("num_hidden_layer: ", best_position[1])
    print("learning_rate: ", best_position[2] / 1000)
