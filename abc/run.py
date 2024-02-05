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
    print('min-accuracy:', round(min, 2), 'max-accuracy:', round(max, 2), 'min-inference-time:', round(min_time, 2),'max-inference-time:', round(max_time, 2),'min-memory:', round(min_mem, 2), 'max-memory:', round(max_mem, 2))

generate_synthetic_data()