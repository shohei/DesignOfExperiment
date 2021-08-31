from scipy import stats
import pandas as pd
import pdb

a10ml = pd.Series([10,10,5,15])
a20ml = pd.Series([15,25,35,25])
a40ml = pd.Series([40,50,20,20])
res = stats.f_oneway(a10ml,a20ml,a40ml)
pdb.set_trace()






