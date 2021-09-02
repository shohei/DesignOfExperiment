import pandas as pd
import pdb
import math

df = pd.read_csv('golf.csv')
mu_suzuki = df['suzuki'].mean()
mu_sato = df['sato'].mean()
std_suzuki = df['suzuki'].std()
std_sato = df['sato'].std()

SN_ratio_suzuki_large_best = -10*math.log10(1/len(df['suzuki'])*sum([1/d**2 for d in df['suzuki'].tolist()]))
SN_ratio_sato_large_best = -10*math.log10(1/len(df['sato'])*sum([1/d**2 for d in df['sato'].tolist()]))

SN_ratio_suzuki_nominal_best = 10*math.log10(mu_suzuki**2/std_suzuki**2 - 1/len(df['suzuki']))
SN_ratio_sato_nominal_best = 10*math.log10(mu_sato**2/std_sato**2 - 1/len(df['sato']))
pdb.set_trace()

