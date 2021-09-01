import pandas as pd
import numpy as np
import pdb

df = pd.read_csv('health.csv')

x1_Y = df[df.healthy=='Y'].x1
x2_Y = df[df.healthy=='Y'].x2
x3_Y = df[df.healthy=='Y'].x3
x4_Y = df[df.healthy=='Y'].x4
x5_Y = df[df.healthy=='Y'].x5
x6_Y = df[df.healthy=='Y'].x6

x1_Y_base = (x1_Y-x1_Y.mean())/x1_Y.std(ddof=0)
x2_Y_base = (x2_Y-x2_Y.mean())/x2_Y.std(ddof=0)
x3_Y_base = (x3_Y-x3_Y.mean())/x3_Y.std(ddof=0)
x4_Y_base = (x4_Y-x4_Y.mean())/x4_Y.std(ddof=0)
x5_Y_base = (x5_Y-x5_Y.mean())/x5_Y.std(ddof=0)
x6_Y_base = (x6_Y-x6_Y.mean())/x6_Y.std(ddof=0)

x1_N = df[df.healthy=='N'].x1
x2_N = df[df.healthy=='N'].x2
x3_N = df[df.healthy=='N'].x3
x4_N = df[df.healthy=='N'].x4
x5_N = df[df.healthy=='N'].x5
x6_N = df[df.healthy=='N'].x6

x1_N_base = (x1_N-x1_Y.mean())/x1_Y.std(ddof=0)
x2_N_base = (x2_N-x2_Y.mean())/x2_Y.std(ddof=0)
x3_N_base = (x3_N-x3_Y.mean())/x3_Y.std(ddof=0)
x4_N_base = (x4_N-x4_Y.mean())/x4_Y.std(ddof=0)
x5_N_base = (x5_N-x5_Y.mean())/x5_Y.std(ddof=0)
x6_N_base = (x6_N-x6_Y.mean())/x6_Y.std(ddof=0)

x_Y_base = pd.DataFrame({'x1':x1_Y_base, 'x2':x2_Y_base, 'x3':x3_Y_base, 'x4':x4_Y_base, 'x5':x5_Y_base, 'x6':x6_Y_base})
S = np.cov(x_Y_base,rowvar=0,bias=True)
V = np.linalg.inv(S)

M_Y = [0]*len(x1_Y)
xm_Y = [0]*len(x1_Y)
for i in range(len(x1_Y)):
    xm_Y[i] = np.vstack( \
       (x1_Y_base.tolist()[i],
        x2_Y_base.tolist()[i],
        x3_Y_base.tolist()[i],
        x4_Y_base.tolist()[i],
        x5_Y_base.tolist()[i],
        x6_Y_base.tolist()[i]))
    M_Y[i] = (xm_Y[i].transpose() @ V @ xm_Y[i]) / 6 #評価項目数6で割る

M_N = [0]*len(x1_N)
xm_N = [0]*len(x1_N)
for i in range(len(x1_N)):
    xm_N[i] = np.vstack( \
       (x1_N_base.tolist()[i],
        x2_N_base.tolist()[i],
        x3_N_base.tolist()[i],
        x4_N_base.tolist()[i],
        x5_N_base.tolist()[i],
        x6_N_base.tolist()[i]))
    M_N[i] = (xm_N[i].transpose() @ V @ xm_N[i]) / 6 #評価項目数6で割る

pdb.set_trace()

