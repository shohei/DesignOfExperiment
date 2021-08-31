import pandas as pd
import numpy as np
from sklearn import linear_model
import pdb
from sklearn import preprocessing

df = pd.read_csv("baseball.csv")

model = linear_model.LinearRegression()
Y = df['error']
X = df.loc[:,("temperature","humidity")]
model.fit(X,Y)
print("coef",model.coef_)
print("icept",model.intercept_)
R2 = model.score(X, Y)

sscaler = preprocessing.StandardScaler()
sscaler.fit(X)
X2 = sscaler.transform(X) 
# sscaler.fit(Y)
# Y2 = sscaler.transform(Y)
model = linear_model.LinearRegression()
model.fit(X2,Y)
print("coef",model.coef_)
print("icept",model.intercept_)
R2 = model.score(X2, Y)
print(R2)
