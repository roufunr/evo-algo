import pandas as pd
from itertools import product
import time

df = pd.read_csv("/home/rouf/Documents/code/evo-algo/real_data/mlp_ORIN_main.csv")





total_df_len = len(df)
new_df_list = []
idx = 1
for i in range(total_df_len):
    df_row = df.iloc[i]
    if df_row['l1'] == 150 or df_row['l2'] == 150 or df_row['l3'] == 150:
        continue

    df_row['idx'] = idx
    new_df_list.append(df_row)
    idx+=1

new_df = pd.DataFrame(new_df_list)
new_df.to_csv('ORIN.csv', index=False)