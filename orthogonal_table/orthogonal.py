import pandas as pd
import numpy as np
import pdb
import scipy.stats
import inspect
import re

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


df = pd.read_csv('sashimi.csv')
# df = pd.read_csv('buhin.csv')
# df = pd.read_csv('buhin2.csv')

# dic = {[]}

all= df.value.tolist()
mean = sum(all)/len(all)
N = len(all)

levels = df.keys().tolist()
levels.remove('value')
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
S_E = S_T-S_A-S_B-S_C-S_D-S_AB-S_AC-S_BC

 # Degree of freedom
fT = N-1
fA = 2-1
fB = 2-1
fC = 2-1
fD = 2-1
fAB = 2-1
fAC = 2-1
fBC = 2-1
fe = fT - len(levels) 

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

# Variance ratio
F_A = V_A/V_E 
F_B = V_B/V_E
F_C = V_C/V_E
F_D = V_D/V_E
F_AB = V_AB/V_E
F_AC = V_AC/V_E
F_BC = V_AC/V_E

p_A = 1-scipy.stats.f.cdf(F_A,fA,fe)
p_B = 1-scipy.stats.f.cdf(F_B,fB,fe)
p_C = 1-scipy.stats.f.cdf(F_C,fC,fe)
p_D = 1-scipy.stats.f.cdf(F_D,fD,fe)
p_AB = 1-scipy.stats.f.cdf(F_AB,fAB,fe)
p_AC = 1-scipy.stats.f.cdf(F_AC,fAC,fe)
p_BC = 1-scipy.stats.f.cdf(F_BC,fBC,fe)

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
