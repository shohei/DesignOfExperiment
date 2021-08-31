# coding: utf-8
# 無作為抽出の場合の直交表(乱塊法ではない)
import pandas as pd
import numpy as np
import pdb
import scipy.stats
import inspect
import re
import sys

def flatten(t):
    if not type(t)==type([]): #t is numpy array 
        t = t.tolist()
    return [item for sublist in t for item in sublist]

def prinfo(*args):
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    vnames = r.split(", ")
    for i,(var,val) in enumerate(zip(vnames, args)):
        print(f"{var} = {val}")

def report_result(pvalue, label):
    if pvalue < 0.05:
        if pvalue < 0.01:
            print("There is a significant effect by {} (p=0.01).".format(label))
        else:
            print("There is a significant effect by {} (p=0.05).".format(label))
    else:
        print("There is no significant effect by {}.".format(label))

if __name__=="__main__":
    if len(sys.argv)!=2:
        print("Usage: $ python orthogonal.py <CSV_FILE.csv>")
        exit()
    csv = str(sys.argv[1])
    df = pd.read_csv(csv)
    
    # df = pd.read_csv('sashimi.csv')
    # df = pd.read_csv('buhin.csv')
    # df = pd.read_csv('buhin_randomized.csv')
    # df = pd.read_csv('buhin2.csv')
    
    alldata= df.value.tolist()
    mean = sum(alldata)/len(alldata)
    N = len(alldata)
    
    levels = df.keys().tolist()
    levels.remove('value')
    
    # Transform A(A1,A2),B(B1,B2),C(C1,C2),D(D1,D2) to N (N1 to N8)
    data2 = np.vstack( \
        (np.array(df[df["A"]==1][df["B"]==1][df["C"]==1][df["D"]==1].value.tolist()),\
        np.array(df[df["A"]==1][df["B"]==1][df["C"]==2][df["D"]==2].value.tolist()),\
        np.array(df[df["A"]==1][df["B"]==2][df["C"]==1][df["D"]==2].value.tolist()),\
        np.array(df[df["A"]==1][df["B"]==2][df["C"]==2][df["D"]==1].value.tolist()),\
        np.array(df[df["A"]==2][df["B"]==1][df["C"]==1][df["D"]==2].value.tolist()),\
        np.array(df[df["A"]==2][df["B"]==1][df["C"]==2][df["D"]==1].value.tolist()),\
        np.array(df[df["A"]==2][df["B"]==2][df["C"]==1][df["D"]==1].value.tolist()),\
        np.array(df[df["A"]==2][df["B"]==2][df["C"]==2][df["D"]==2].value.tolist()))).transpose()
    
    alldata2 = flatten(data2)
    N2 = len(alldata2)
    mean2 = sum(alldata2)/N2
    S_T = sum([(x-mean2)**2 for x in alldata2])
    
    (m2,n2) = data2.shape
    S_temp = [0]*n2
    for j in range(n2):
        S_temp[j] = sum(data2[:,j])
    S_temp_average = sum(S_temp)/len(S_temp)
    S_N = 0 
    for j in range(n2):
        S_N += (sum(data2[:,j])-S_temp_average)**2 / len(data2[:,j])
    # S_e: 純誤差
    S_e = S_T - S_N
    
    # levels = ["A","B","C","D"]
    S = [0]*len(levels)
    s = [0,0]
    
    S_T = sum([(x-mean)**2 for x in df.value.tolist()])
    
    for i in range(len(levels)):
        for j in range(2):
            level = levels[i]
            xs = df[df[level]==(j+1)].value.tolist()
            s[j] = sum(xs)
        S[i] = (s[0]-s[1])**2 / N
        
    S_A = S[levels.index("A")]
    S_B = S[levels.index("B")]
    S_C = S[levels.index("C")]
    S_D = S[levels.index("D")] if "D" in levels else 0
    S_AB = S[levels.index("AB")] if "AB" in levels else 0
    S_AC = S[levels.index("AC")] if "AC" in levels else 0
    S_BC = S[levels.index("BC")] if "BC" in levels else 0
    # S_E: 誤差
    S_E = S_T-S_A-S_B-S_C-S_D-S_AB-S_AC-S_BC
    # S_e_prime: 不適合 
    S_e_prime = S_E - S_e 
    
    # Degree of freedom
    fT = N-1
    fA = 2-1
    fB = 2-1
    fC = 2-1
    fD = 2-1
    fAB = 2-1
    fAC = 2-1
    fBC = 2-1
    fN = 8-1
    fe = fT - fN
    fe_prime = 2-1
    
    if S_E==0:
        # Run pooling
        print("Pooling")
        S_E = S_AB + S_AC + S_BC + S_E
        fe = fAB + fAC + fBC + fe
    
    # Invariant variance
    V_T = S_T/fT
    V_A = S_A/fA
    V_B = S_B/fB
    V_C = S_C/fC
    V_D = S_D/fD
    V_E = S_E/fe
    V_AB = S_AB/fAB
    V_AC = S_AC/fAC
    V_BC = S_BC/fBC
    V_e = S_e / fe
    V_e_prime = S_e_prime / fe_prime
    
    # Variance ratio
    F_A = V_A/V_E 
    F_B = V_B/V_E
    F_C = V_C/V_E
    F_D = V_D/V_E
    F_AB = V_AB/V_E
    F_AC = V_AC/V_E
    F_BC = V_AC/V_E
    F_e_prime = V_e_prime / V_e
    
    p_A = 1-scipy.stats.f.cdf(F_A,fA,fe)
    p_B = 1-scipy.stats.f.cdf(F_B,fB,fe)
    p_C = 1-scipy.stats.f.cdf(F_C,fC,fe)
    p_D = 1-scipy.stats.f.cdf(F_D,fD,fe)
    p_AB = 1-scipy.stats.f.cdf(F_AB,fAB,fe)
    p_AC = 1-scipy.stats.f.cdf(F_AC,fAC,fe)
    p_BC = 1-scipy.stats.f.cdf(F_BC,fBC,fe)
    p_e_prime = 1-scipy.stats.f.cdf(F_e_prime,fe_prime,fe)
    
    # prinfo(pA, pB, pC, pD)
    
    report_result(p_A,"A")
    report_result(p_B,"B")
    report_result(p_C,"C")
    if "D" in levels:
        report_result(p_D,"D") 
    if "AB" in levels:
        report_result(p_AB,"AB")
    if "AC" in levels:
        report_result(p_AC,"AC")
    if "BC" in levels:
        report_result(p_BC,"BC")
    report_result(p_e_prime,"e_prime")
    
    # prinfo(F_A, F_B, F_C, F_D, F_AB, F_AC, F_e_prime)