import pandas as pd
import numpy as np
import scipy.stats
import pdb
import sys

def flatten(t):
    return [item for sublist in t for item in sublist]

if __name__=="__main__":

    if len(sys.argv)!=2:
        print("Usage: $ python two_way_anova.py <CSV_FILE.csv>")
        exit()
    csv = str(sys.argv[1])
    df = pd.read_csv(csv)
    nA = max(df["A"])
    nB = max(df["B"])
    data = []
    for i in range(nA):
        data.append(([[]]*nB))
    
    for i in range(nA):
        for j in range(nB):
            _df = df[df["A"]==(i+1)][df["B"]==(j+1)]
            data[i][j] = _df.value.tolist()
    
    flatten_data = flatten(flatten(data))
    N = len(flatten_data)
    mu = sum(flatten_data)/N
    
    data = np.array(data)
    
    # S: Total sum of squares of deviation
    S = 0
    for i in range(nA):
        for j in range(nB):
            d = data[i][j]
            for v in d:
                S += (v-mu)**2
    
    # S_A: Sum of squares of deviation for factor A
    S_A = 0
    for i in range(nA):
        d = flatten(data[i,:].tolist())
        n_a = len(d)
        mean_a = sum(d)/n_a
        S_A += n_a * (mean_a-mu)**2
    
    # S_B: Sum of squares of deviation for factor B
    S_B = 0
    for j in range(nB):
        d = flatten(data[:,j].tolist())
        n_b = len(d)
        mean_b = sum(d)/n_b
        S_B += n_b * (mean_b-mu)**2
    
    # S_AB: Sum of squares of deviation for interaction between A and B
    S_AB = 0
    for i in range(nA):
        for j in range(nB):
            d = data[i][j].tolist()
            n_ab = len(d)
            mean_ab = sum(d)/n_ab
            S_AB += n_ab*(mean_ab-mu)**2
    S_AB = S_AB - S_A - S_B
    
    # S_E: Sum of squares of deviation for erros
    S_E = S - S_A - S_B - S_AB
    
    # Degree of freedom
    f = N-1
    f_A = nA - 1
    f_B = nB - 1
    f_AB = (nA-1)*(nB-1)
    f_E = N - nA*nB
    
    # Invariant variance
    V_A = S_A/f_A
    V_B = S_B/f_B
    V_AB = S_AB / f_AB
    V_E =  S_E / f_E
    
    # Ratio of variances
    F_A = V_A/V_E
    F_B = V_B/V_E
    F_AB = V_AB/V_E
    
    # Caluculation of F distribution
    print("****************")
    print("Result of ANOVA")
    print("****************")
    if F_A > scipy.stats.f.isf(0.05, f_A, f_E):
        if F_A > scipy.stats.f.isf(0.01, f_A, f_E):
           print("Significant difference among A (p=0.01)")
        else:
           print("Significant difference among A (p=0.05)")
    else:
           print("No significant difference among A")
    
    if F_B > scipy.stats.f.isf(0.05, f_B, f_E):
        if F_B > scipy.stats.f.isf(0.01, f_B, f_E):
           print("Significant difference among B (p=0.01)")
        else:
           print("Significant difference among B (p=0.05)")
    else:
           print("No significant difference among B")
    
    if F_AB > scipy.stats.f.isf(0.05, f_AB, f_E):
        if F_AB > scipy.stats.f.isf(0.01, f_AB, f_E):
           print("Significant difference among interaction AxB (p=0.01)")
        else:
           print("Significant difference among interaction AxB (p=0.05)")
    else:
           print("No significant difference among interaction AxB")
    
