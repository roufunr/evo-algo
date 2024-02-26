import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Importing Axes3D for 3D plotting
import pandas as pd
import numpy as np
import statistics
import matplotlib.lines as mlines
import os
import pandas as pd
import copy

base_path = "/home/rouf-linux/mlp_4d_plot/"
device = "ORIN"        #PC or ORIN
data_path = f"{base_path}/data/mlp_{device}.csv"
plot_height = 36
plot_width = 30

layer_info_to_idx_mapper = {
    "50_0_0":  1,
    "100_0_0": 2,
    "50_50_0":  3,
    "50_100_0": 4,
    "100_50_0":  5,
    "100_100_0": 6,
    "50_50_50":  7,
    "50_50_100": 8,
    "50_100_50":  9,
    "50_100_100": 10,
    "100_50_50":  11,
    "100_50_100": 12,
    "100_100_50":  13,
    "100_100_100": 14,
}
id_to_layer_info_mapper = {
    1: '50_0_0', 
    2: '100_0_0', 
    3: '50_50_0', 
    4: '50_100_0', 
    5: '100_50_0', 
    6: '100_100_0', 
    7: '50_50_50', 
    8: '50_50_100', 
    9: '50_100_50', 
    10: '50_100_100', 
    11: '100_50_50', 
    12: '100_50_100', 
    13: '100_100_50', 
    14: '100_100_100'
}

parameter_dic = {}
value_dic = {}

def normalize(data):
    if ((np.max(data) - np.min(data))==0):
        return [1]*len(data)

    returnvalue=(data - np.min(data)) / (np.max(data) - np.min(data))
    returnvalue = returnvalue + 0.1
    
    return returnvalue

def draw(xyz_axis_names, value_axis_name, aggregation):
    x_value=xyz_axis_names[0]
    y_value=xyz_axis_names[1]
    z_value=xyz_axis_names[2]

    d = [parameter_dic.get(x_value), parameter_dic.get(y_value), parameter_dic.get(z_value)]
    r=value_dic.get(value_axis_name)

    layer_infomation_idx = -1
    if "layer_information" in xyz_axis_names:
        layer_infomation_idx = xyz_axis_names.index("layer_information")
    
    if layer_infomation_idx != -1:
        layer_information_data = d[layer_infomation_idx]
        trainable_param_list = []
        for lid in layer_information_data:
            trainable_param_list.append(layer_info_to_idx_mapper[lid])
        d[layer_infomation_idx] = trainable_param_list
    u = []
    ind = []
    for i in range(3):
        u_, ind_ = np.unique(d[i], return_inverse=True)
        if i == layer_infomation_idx:
            u_list = list(u_)
            new_u_list = [id_to_layer_info_mapper[x] for x in u_list]
            u_ = np.array(new_u_list)
        u.append(u_)
        ind.append(ind_)
    
    u1 = u[0]
    u2 = u[1]
    u3 = u[2]
    
    ind1 = ind[0]
    ind2 = ind[1]
    ind3 = ind[2]
    
    dic={}
    for i in range(0, len(ind1)):
        key=(ind1[i], ind2[i], ind3[i])
        if(key in dic):
            value=dic.get(key)
            value.append(r[i])
            dic[key]=value
        else:
            dic[key]=[r[i]]

    r=[]
    xs=[]
    ys=[]
    zs=[]
    for key in dic:
        if(aggregation=="mean"):
            r.append(statistics.mean(dic.get(key)))
        elif(aggregation=="max"):
            r.append(max(dic.get(key)))
        elif(aggregation=="min"):
            r.append(min(dic.get(key)))
        elif(aggregation=="median"):
            r.append(statistics.median(dic.get(key)))
        else:
            r.append(statistics.mean(dic.get(key)))

        xs.append(key[0])
        ys.append(key[1])
        zs.append(key[2])

    
    r_normalized = normalize(r) if value_axis_name != "f1" else r
    
    color_code=[]
    max_z = max(r_normalized) - 0.1
    for z in r_normalized:
        z = z - 0.1
        if(z==max_z):
            color_code.append("red")######
        elif(z<0.5):
            color_code.append("darkcyan")
        elif(0.50<=z and z<0.70):
            color_code.append("darkorange")
        elif(0.70<=z and z<0.80):
            color_code.append("magenta")
        elif(0.80<=z and z<0.90):
            color_code.append("green")
        elif(0.90<=z):
            color_code.append("blue")

    marker_code=[]
    for z in r_normalized:
        z = z - 0.1
        if(z == max_z):
            marker_code.append("*")

        elif(z<0.5):
            marker_code.append("+")
        elif(0.50<=z and z<0.70):
            marker_code.append("s")
        elif(0.70<=z and z<0.80):
            marker_code.append("^")
        elif(0.80<=z and z<0.90):
            marker_code.append("x")
        elif(0.90<=z):
            marker_code.append("o")

    fig = plt.figure(figsize=(plot_height, plot_width))  # Adjust the figsize here for larger size
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(r)): 
        ax.scatter(xs[i], ys[i], zs[i], color=color_code[i], s=60, marker=marker_code[i]) 

    
    ax.set_title(value_axis_name)
    ax.set_xlabel(x_value, labelpad=5)
    ax.set_ylabel(y_value, labelpad=10)
    ax.set_zlabel(z_value, labelpad=1)


    ax.set_xticks(list(set(ind1)))
    ax.set_xticklabels(u1)

    ax.set_yticks(list(set(ind2)))
    ax.set_yticklabels(u2)

    ax.set_zticks(list(set(ind3)))##3
    ax.set_zticklabels(u3)##3

    marker1 = mlines.Line2D([], [], color='red', marker='*', linestyle='None', markersize=6, label='max')
    marker2 = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=6, label='[0.9 - 1.0]')
    marker3 = mlines.Line2D([], [], color='green', marker='x', linestyle='None', markersize=6, label='[0.8 - 0.9)')
    marker4 = mlines.Line2D([], [], color='magenta', marker='^', linestyle='None', markersize=6, label='[0.7 - 0.8)')
    marker5 = mlines.Line2D([], [], color='darkorange', marker='s', linestyle='None', markersize=6, label='[0.5 - 0.7)')
    marker6 = mlines.Line2D([], [], color='darkcyan', marker='+', linestyle='None', markersize=6, label='[0 - 0.5)')			

    plt.legend(handles=[marker1, marker2, marker3, marker4, marker5, marker6],  bbox_to_anchor=(0, 0.6), loc=1, borderaxespad=0.1)

    path = f"{base_path}/plot/{xyz_axis_names[0]}_{xyz_axis_names[1]}_{xyz_axis_names[2]}_{value_axis_name}_{aggregation}_{device}.png"
    plt.savefig(path)
    plt.show()
    plt.clf()
    plt.close()

    print(f"Done plotting {path}")

def load_data():
    df = pd.read_csv(data_path)
    new_df_list = []
    total_rows = len(df)
    for i in range(total_rows):
        old_row = df.loc[i]
        if old_row['l1'] == 150 or old_row['l2'] == 150 or old_row['l3'] == 150:
            continue
        new_df_list.append(old_row)
    new_df_len = len(new_df_list)
    new_df = pd.DataFrame(new_df_list, index = [x for x in range(new_df_len)])
    
    
    sorting_column = []
    for i in range(new_df_len):
        l1 = new_df.loc[i]['l1']
        l2 = new_df.loc[i]['l2']
        l3 = new_df.loc[i]['l3']
        l1 = 1 if l1 == 0 else l1
        l2 = 1 if l2 == 0 else l2
        l3 = 1 if l3 == 0 else l3
        sorting_column.append(l1 * l2 * l3)
    
    new_df["sort"] = sorting_column

    sorted_df = new_df.sort_values(by="sort", ascending=True)

    data_dict = {}
    for column_name in list(sorted_df.columns):
        data_dict[column_name] = list(sorted_df[column_name])
    
    total_new_data_points = len(data_dict['l1'])
    
    X = {}
    X["layer_information"] = [f"{data_dict['l1'][i]}_{data_dict['l2'][i]}_{data_dict['l3'][i]}" for i in range(total_new_data_points)]
    X["activation"] = data_dict["activation"]
    X["solver"] = data_dict["solver"]
    X["learning_rate"] = data_dict["learning_rate"]
    X["alpha"] = data_dict["alpha"]
    X["warm_start"] = data_dict["warm_start"]

    Y = {}
    Y["f1"] = data_dict["f1"]
    Y["inference_time"] = data_dict["inference time (ms)"]
    Y["memory"] = data_dict["total_addr_space(mem)(MiB)"]

    global parameter_dic
    parameter_dic = copy.deepcopy(X)

    global value_dic
    value_dic = copy.deepcopy(Y)

load_data()


if __name__=="__main__":
    # layer_information
    # activation
    # solver
    # learning_rate
    # alpha
    # warm_start

    # f1
    # inference_time
    # memory

    # min
    # max
    # median
    # mean

    draw(['layer_information', 'activation', 'warm_start'], 'memory', 'mean')