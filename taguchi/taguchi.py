import pandas as pd
import numpy as np
import pdb
import math

df = pd.read_csv('magcup.csv')

n = 10 # This is the assumed sampling number
x_labels = [c for c in df.columns if 'x' in c]
w_labels = [c for c in df.columns if 'w' in c]
df_error = df[x_labels]
df_control = df[w_labels]
N = len(df) # 8
N_x = len(x_labels)
N_w = len(w_labels)

SN = [0]*N
for i, row in df_error.iterrows():
    SN[i] = 10*math.log10(row.mean()**2/row.var()-1/n)

w_mean_difference = [0]*N_w
for i in range(N_w):
    w = w_labels[i]
    df_w = df_control[w]
    w_mean_a = 0
    w_mean_b = 0
    for j, sn in enumerate(SN):
        if df_w[j]==1:
            w_mean_a += SN[j]
        else:
            w_mean_b += SN[j]
    w_mean_a = w_mean_a / (N/2)
    w_mean_b = w_mean_b / (N/2)
    w_mean_difference[i] = w_mean_a - w_mean_b
    
print(w_mean_difference)
pdb.set_trace()

