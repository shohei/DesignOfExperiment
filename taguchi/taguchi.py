import pandas as pd
import numpy as np
import pdb
import math
import matplotlib.pyplot as plt

df = pd.read_csv('magcup.csv')

n = 10 # This is the assumed sampling number
# wは制御因子, xは誤差因子
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

# 改善すべき因子の抽出 (第一段階: SN比を利用)
SN_mean_difference = [0]*N_w
sn_mean_a = [0]*N_w
sn_mean_b = [0]*N_w
for i in range(N_w):
    w = w_labels[i]
    df_w = df_control[w]
    for j, sn in enumerate(SN):
        if df_w[j]==1:
            sn_mean_a[i] += SN[j]
        else:
            sn_mean_b[i] += SN[j]
    sn_mean_a[i] = sn_mean_a[i] / (N/2)
    sn_mean_b[i] = sn_mean_b[i] / (N/2)
    SN_mean_difference[i] = sn_mean_a[i] - sn_mean_b[i]
    
# 目標値のあわせこみ(第二段階: 平均値を利用)
Mean = df_error.mean(axis=1).tolist()
Mean_mean_difference = [0]*N_w
mean_mean_a = [0]*N_w
mean_mean_b = [0]*N_w
for i in range(N_w):
    w = w_labels[i]
    df_w = df_control[w]
    for j, sn in enumerate(Mean):
        if df_w[j]==1:
            mean_mean_a[i] += Mean[j]
        else:
            mean_mean_b[i] += Mean[j]
    mean_mean_a[i] = mean_mean_a[i] / (N/2)
    mean_mean_b[i] = mean_mean_b[i] / (N/2)
    Mean_mean_difference[i] = mean_mean_a[i] - mean_mean_b[i]

# 要因効果図の描画
offset = 150
initial_x = 20
width = 50
x_a = [initial_x]
x_b = [initial_x+width]
# x座標生成
for i in range(N_w):
    x_a.append(x_a[-1]+offset)
    x_b.append(x_b[-1]+offset)

plt.figure(1)
plt.title("SN ratio")
plt.figure(2)
plt.title("Mean evaluation")
# 要因効果図の描画
for i in range(N_w):
    sn_a = sn_mean_a[i]
    sn_b = sn_mean_b[i]
    plt.figure(1)
    plt.plot(x_a[i],sn_a,'bo')
    plt.plot(x_b[i],sn_b,'bo')
    plt.plot((x_a[i],x_b[i]),(sn_a,sn_b),'r')

    mean_a = mean_mean_a[i]
    mean_b = mean_mean_b[i]
    plt.figure(2)
    plt.plot(x_a[i],mean_a,'bo')
    plt.plot(x_b[i],mean_b,'bo')
    plt.plot((x_a[i],x_b[i]),(mean_a,mean_b),'r')

# 平均値の描画
sn_average = pd.DataFrame(SN).mean()
mean_average = pd.DataFrame(Mean).mean()
plt.figure(1)
plt.plot((x_a[0],x_b[-1]),(sn_average,sn_average),'k--')
plt.figure(2)
plt.plot((x_a[0],x_b[-1]),(mean_average,mean_average),'k--')

# ラベルの描画
for i in range(N_w):
    plt.figure(1)
    axes = plt.gca()
    yl = axes.get_ylim()[0]
    plt.text(x_a[i], yl, 'w'+str(i+1), fontsize=12)

    plt.figure(2)
    axes = plt.gca()
    yl = axes.get_ylim()[0]
    plt.text(x_a[i], yl, 'w'+str(i+1), fontsize=12)

plt.show()

# パラメータの最適化(重回帰分析)



