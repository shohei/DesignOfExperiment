import pandas as pd
import numpy as np
import scipy.stats
import pdb
import sys
import inspect
import re
from itertools import combinations
import math

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

def one_way_anova(df):
    nA = max(df["A"])
    data = []
    for i in range(nA):
        data.append(([]))
    
    for i in range(nA):
        _df = df[df["A"]==(i+1)]
        data[i] = _df.value.tolist()
        if len(data[i])==0:
            print("Cannot run ANOVA since no data found for A{}",i+1)
            exit()

    flatten_data = flatten(data)
    N = len(flatten_data)
    mu = sum(flatten_data)/N
    
    # S: Total sum of squares of deviation
    S = 0
    for i in range(nA):
        d = data[i]
        for v in d:
            S += (v-mu)**2
    
    # S_A: Sum of squares of deviation for factor A
    S_A = 0
    for i in range(nA):
        d = data[i]
        n_a = len(d)
        mean_a = sum(d)/n_a
        S_A += n_a * (mean_a-mu)**2

    # S_E: Sum of squares of deviation for erros
    S_E = S - S_A 

    # Degree of freedom
    f = N-1
    f_A = nA - 1
    f_E = N - nA

    # Invariant variance
    V_A = S_A / f_A
    V_E =  S_E / f_E
    
    # Ratio of variances
    F_A = V_A/V_E

    # Caluculation of F distribution
    print("********************************")
    print("Result of ANOVA:",csv)
    print("********************************")

    if F_A > scipy.stats.f.isf(0.05, f_A, f_E):
        if F_A > scipy.stats.f.isf(0.01, f_A, f_E):
           print("Significant difference among A (p=0.01)")
        else:
           print("Significant difference among A (p=0.05)")
        multiple_comparison_bonferroni(data, V_E, pvalue=0.05)
    else:
           print("No significant difference among A")

def two_way_anova(df):
    nA = max(df["A"])
    nB = max(df["B"])
    data = []
    for i in range(nA):
        data.append(([[]]*nB))
    
    has_no_repetition = True 
    for i in range(nA):
        for j in range(nB):
            _df = df[df["A"]==(i+1)][df["B"]==(j+1)]
            data[i][j] = _df.value.tolist()
            if len(data[i][j])==0:
                print("Cannot run ANOVA since no data found for A{},B{}",i+1,j+1)
                exit()
            if len(data[i][j])>1:
                has_no_repetition = False

    # Ignore interaction if each data has only one sample 
    if has_no_repetition:
        ignore_interaction = True
    else:
        ignore_interaction = False

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
        d = flatten(data[i,:])
        n_a = len(d)
        mean_a = sum(d)/n_a
        S_A += n_a * (mean_a-mu)**2
    
    # S_B: Sum of squares of deviation for factor B
    S_B = 0
    for j in range(nB):
        d = flatten(data[:,j])
        n_b = len(d)
        mean_b = sum(d)/n_b
        S_B += n_b * (mean_b-mu)**2
    
    # S_AB: Sum of squares of deviation for interaction between A and B
    S_AB = 0
    for i in range(nA):
        for j in range(nB):
            d = data[i][j]
            if not type(d)==type([]): #d is numpy array 
                d = d.tolist()
            n_ab = len(d)
            mean_ab = sum(d)/n_ab
            S_AB += n_ab*(mean_ab-mu)**2
    S_AB = S_AB - S_A - S_B
    
    # S_E: Sum of squares of deviation for erros
    if ignore_interaction:
        S_AB = 0
    S_E = S - S_A - S_B - S_AB
    
    # Degree of freedom
    f = N-1
    f_A = nA - 1
    f_B = nB - 1
    f_AB = (nA-1)*(nB-1)
    f_E = N - nA*nB
    if ignore_interaction:
        f_E = (nA-1)*(nB-1)
    
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
    print("********************************")
    print("Result of ANOVA:",csv)
    print("********************************")

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
    
    if ignore_interaction:
        print("*The interaction AxB is ignored for this input data*")
    else:
        if F_AB > scipy.stats.f.isf(0.05, f_AB, f_E):
            if F_AB > scipy.stats.f.isf(0.01, f_AB, f_E):
               print("Significant difference among interaction AxB (p=0.01)")
            else:
               print("Significant difference among interaction AxB (p=0.05)")
        else:
               print("No significant difference among interaction AxB")
   
    #prinfo(N, nA, nB)
    #prinfo(S, S_A, S_B, S_AB, S_E)
    #prinfo(f, f_A, f_B, f_AB, f_E) 
    #prinfo(V_A, V_B, V_AB, V_E)
    #prinfo(F_A, F_B, F_AB)

def multiple_comparison_bonferroni(data, V_E, pvalue):
    nA = len(data)
    flatten_data = flatten(data)
    N = len(flatten_data)
    mu = sum(flatten_data)/N
    means = [[]]*nA
    for i in range(nA):
        means[i] = sum(data[i])/len(data[i])

    pvalue_prime = pvalue / (nA*(nA-1)/2)  
    f_prime = N - nA 
    # Critical value T0
    # p-value is divided by 2 since TINV() function of Excel uses two-sided tail probability
    # https://scipy-user.scipy.narkive.com/j7pH5qnd/how-does-scipy-stats-t-isf-work
    T0 = scipy.stats.t.isf(pvalue_prime/2, f_prime) 

    combi =  [x for x in combinations(range(nA),2)]
    T = [[]]*len(combi)
    for i,(u,v) in enumerate(combi):
        n_u = len(data[u])
        n_v = len(data[v])
        T[i] = abs(means[u]-means[v])/math.sqrt(V_E*(1/n_u+1/n_v))
        if T[i] < T0:
            print(" A{} and A{} don't have a significant difference.".format(u,v))
        else:
            print(" A{} and A{} have a significant difference.".format(u,v))


def check_normality(df, pvalue):
    if scipy.stats.shapiro(df.value).pvalue < pvalue:
        print("The given data doesn't comply with the normal distribution.")
        print("Thus the multiple comparison test cannot be applied.")
        print("Use a non-parametric multiple comparison test instead.")
        return False 
    else:
        return True

def check_homoscedasticity_one_way(df, pvalue):
    nA = max(df["A"])
    valueA = [df[df["A"]==i].value.tolist() for i in range(1,nA+1)]
    if scipy.stats.bartlett(*valueA).pvalue < pvalue:
        print("The given data doesn't comply with the homoscedasticity.")
        print("Thus the multiple comparison test cannot be applied.")
        print("Apply 1) Non-parametric test, 2) Kruskal-Wallis test,")
        print("and finally 3) Non-parametric multiple comparison test.")
        return False
    else:
        return True

def check_homoscedasticity_two_way(df, pvalue):
    nA = max(df["A"])
    nB = max(df["B"])
    valueA = [df[df["A"]==i].value.tolist() for i in range(1,nA+1)]
    valueB = [df[df["B"]==i].value.tolist() for i in range(1,nB+1)]
    isHomoscedasticity = True
    if scipy.stats.bartlett(*valueA).pvalue < pvalue:
        print("The cateogry A doesn't comply with the homoscedasticity.")
        print("Thus the multiple comparison test cannot be applied.")
        print("Apply 1) Non-parametric test, 2) Kruskal-Wallis test,")
        print("and finally 3) Non-parametric multiple comparison test.")
        isHomoscedasticity = False
    if scipy.stats.bartlett(*valueB).pvalue < pvalue:
        print("The category B doesn't comply with the homoscedasticity.")
        print("Thus the multiple comparison test cannot be applied.")
        print("Apply 1) Non-parametric test, 2) Kruskal-Wallis test,")
        print("and finally 3) Non-parametric multiple comparison test.")
        isHomoscedasticity = False
    return isHomoscedasticity

if __name__=="__main__":
    if len(sys.argv)!=2:
        print("Usage: $ python anova.py <CSV_FILE.csv>")
        exit()
    csv = str(sys.argv[1])
    df = pd.read_csv(csv)

    if "B" in df.keys():
        # Two way ANOVA
        print("Two way ANOVA")
        if not check_normality(df, pvalue=0.05) \
           or not check_homoscedasticity_two_way(df, pvalue=0.05):
            exit()
        two_way_anova(df)
    else:
        # One way ANOVA
        print("One way ANOVA")
        if not check_normality(df, pvalue=0.05) \
           or not check_homoscedasticity_one_way(df, pvalue=0.05):
            exit()
        one_way_anova(df)

