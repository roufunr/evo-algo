import pandas as pd
data_path = "/home/rouf/evo-algo/genetic_algorithm/gen_PC.csv"
df = pd.read_csv(data_path)
d1 = []
d2 = []
d3 = []
total_data = len(df)
for i in range(total_data):
    row = df.iloc[i]
    weights = [row['f1-weight'], row['inference-time-weight'], row['memory-weight']]
    zero_count = 0
    for w in weights:
        if w == 0:
            zero_count += 1
    
    if zero_count == 0:
        d3.append(row)
    elif zero_count == 1:
        d2.append(row)
    else:
        d1.append(row)
    

d1_df = pd.DataFrame(d1)
d1_df.drop(columns=['idx'], inplace=True)
d1_df.to_csv("1D_pc.csv", index=False)

d2_df = pd.DataFrame(d2)
d2_df.drop(columns=['idx'], inplace=True)
d2_df.to_csv("2D_pc.csv", index=False)

d3_df = pd.DataFrame(d3)
d3_df.drop(columns=['idx'], inplace=True)
d3_df.to_csv("3D_pc.csv", index=False)
