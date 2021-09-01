import pandas as pd
import numpy as np
import pdb
import math

def mahalanobis(df, labels=[1,2,3,4,5,6]):
    N = len(labels)
    x_Y = [[]]*N
    for i in range(N):
        x_Y[i] = df[df.healthy=='Y']["x"+str(labels[i])]
    
    x_Y_base = [[]]*N
    for i in range(N):
        x_Y_base[i] = (x_Y[i]-x_Y[i].mean())/x_Y[i].std(ddof=0)
    
    x_N = [[]]*N
    for i in range(N):
        x_N[i] = df[df.healthy=='N']["x"+str(labels[i])]

    x_N_base = [[]]*N
    for i in range(N):
        x_N_base[i] = (x_N[i]-x_Y[i].mean())/x_Y[i].std(ddof=0)

    hash = {}
    for i in range(N):
        hash['x'+str(labels[i])] = x_Y_base[i]

    x_Y_base = pd.DataFrame(hash)
    S = np.cov(x_Y_base,rowvar=0,bias=True)
    V = np.linalg.inv(S)
    
    N_sample_Y = len(x_Y[0])
    M_Y = [0]*N_sample_Y
    xm_Y = [0]*N_sample_Y
    for i in range(N_sample_Y):
        xm_Y_temp = 0
        for j in range(N):
            x_Y_base_j = x_Y_base["x"+str(labels[j])]
            xm_Y_temp = np.vstack((xm_Y_temp,x_Y_base_j.tolist()[i]))
        xm_Y[i] = xm_Y_temp[1:]

        M_Y[i] = (xm_Y[i].transpose() @ V @ xm_Y[i]) / N #評価項目数Nで割る
    
    N_sample_N = len(x_N[0])
    M_N = [0]*N_sample_N
    xm_N = [0]*N_sample_N
    for i in range(N_sample_N):
        xm_N_temp = 0
        for j in range(N):
            # x_N_base_j = x_N_base["x"+str(labels[j])]
            x_N_base_j = x_N_base[j]
            xm_N_temp = np.vstack((xm_N_temp,x_N_base_j.tolist()[i]))

        xm_N[i] = xm_N_temp[1:]
        M_N[i] = (xm_N[i].transpose() @ V @ xm_N[i]) / N #評価項目数Nで割る

    return (M_Y, M_N)    

if __name__=="__main__":
    df = pd.read_csv('health.csv')
    labels = [[1,2,3,4,5,6],[1,2,3],[1,4,5],[1,6],[2,4,6],[2,5],[3,4],[3,5,6]]
    N = len(labels) #8

    # Calculate Mahalanobis distance
    M_N = [0]*N
    for i in range(N):
        _, M_N[i] = mahalanobis(df, labels[i])
        print(M_N[i])

    # Calculate SN ratio
    SN = [0]*N
    N_labels = len(df[df["healthy"]=='N'])
    for i in range(N):
        sumD = sum([1/m for m in M_N[i]])
        SN[i] = -10*math.log10(1/N_labels*sumD)

    SN_labels = [[1,2,3,4],[1,2,5,6],[1,2,7,8],[1,3,5,7],[1,3,6,8],[1,4,5,8]]
    SN_difference = [0]*6
    for i in range(6): # 6 is for x1,x2,x3,x4,x5,x6
        SN_use = []
        SN_no_use = [] 
        for j in range(N): 
            if (j+1) in SN_labels[i]:
                SN_use.append(SN[j])
            else:
                SN_no_use.append(SN[j])
        SN_use_average = sum(SN_use)/len(SN_use)
        SN_no_use_average = sum(SN_no_use)/len(SN_no_use)
        SN_difference[i] = SN_use_average-SN_no_use_average

    print(SN_difference)

    SN_difference_with_feature =[[]]*6
    for i in range(6):
        SN_difference_with_feature[i] = (SN_difference[i],'x'+str(i+1))

    # Sort by value
    SN_difference_with_feature = list(reversed(sorted(SN_difference_with_feature)))

    SN_difference_with_feature_selected = []
    for i in range(6):
        if SN_difference_with_feature[i][0] > 0:
            SN_difference_with_feature_selected.append(SN_difference_with_feature[i])

    selected_labels = []
    for i in range(len(SN_difference_with_feature_selected)):
        feature_name = SN_difference_with_feature_selected[i][1]
        selected_labels.append(int(feature_name.replace('x','')))

    # Recalculate Mahalanobis distance
    M_Y_selected, M_N_selected = mahalanobis(df, selected_labels)
    print(M_Y_selected)






