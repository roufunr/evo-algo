import random
import copy

_range_ = 100
hidden_layer_sizes = [x * 10 for x in range(1, _range_ + 1)]
number_of_hidden_layers = [x for x in range(1, _range_ + 1)]
learning_rates = [x/1000 for x in range(1, _range_ + 1)]
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
    print('min-accuracy:', round(min, 2), 'max-accuracy:', round(max, 2), 'min-inference-time:', round(min_time, 2),'max-inference-time:', round(max_time, 2),'min-memory:', round(min_mem, 2), 'max-memory:', round(max_mem, 2))

# abc parameters
dimensions = 3                                              # number of parameters
food_sources_nums = 10
particle_nums = dimensions * food_sources_nums              # multiple of 3 because we have three parameters to be tuned (N)
iteration_nums = 10                                         # you can set any number
abandoned_solution_limit = 1
counter = 0

def calculate_fitness(x):
    return ((1/(1 + x)) if x > 0 else (1 + abs(x))) 

def objective_function(x):
    global counter
    counter += 1
    accuracy, time, memory = data_param_to_metric[(x[0] * 10, x[1], x[2]/1000)]
    return  (0.34) * (accuracy * 100 * 100) -  (0.33) * time - (0.33) * memory

def perform_employed_phase(food_sources_data, f_x, fitnesses, trials):
    for i in range(food_sources_nums):
        X = food_sources_data[i]
        choices_idx = [j for j in range(food_sources_nums)]
        choices_idx.remove(i)
        Xp = food_sources_data[random.choice(choices_idx)]
        dim_idx_choice = random.choice([j for j in range(dimensions)])
        phi = random.uniform(1/_range_, _range_/_range_)
        x_new = X[dim_idx_choice] + phi * (X[dim_idx_choice] - Xp[dim_idx_choice])
        x_new = round(x_new)
        if x_new < 1: 
            x_new = 1
        if x_new > _range_: 
            x_new = _range_
        X_new = copy.deepcopy(X)
        X_new[dim_idx_choice] = x_new
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
            dim_idx_choice = random.choice([j for j in range(dimensions)])
            phi = random.uniform(1/_range_, _range_/_range_)
            x_new = X[dim_idx_choice] + phi * (X[dim_idx_choice] - Xp[dim_idx_choice])
            x_new = round(x_new)
            if x_new < 1: 
                x_new = 1
            if x_new > _range_: 
                x_new = _range_
            X_new = copy.deepcopy(X)
            X_new[dim_idx_choice] = x_new
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
            food_sources_data[i] = [random.randint(1, _range_) for _ in range(dimensions)]
            f_x[i] = objective_function(food_sources_data[i])
            fitnesses[i] = calculate_fitness(f_x[i])
            trials[i] = 0


def run_abc(): 
    food_sources_data = [[random.randint(1, _range_) for i in range(dimensions)] for _ in range(food_sources_nums)]
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
    generate_synthetic_data()
    best_position = run_abc()
    accuracy, time, memory = data_param_to_metric[(best_position[0] * 10, best_position[1], best_position[2]/1000)]
    
    print("Optimal Accuracy: ", round(accuracy, 3), "%")
    print("Optimal inference time: ", round(time, 3), "s")
    print("Optimal required memory: ", round(memory, 3), "MiB")
    print("FOR")
    print("hidden_layer_size: ", best_position[0] * 10)
    print("num_hidden_layer: ", best_position[1])
    print("learning_rate: ", best_position[2] / 1000)
    print("total overhead: ", counter)
