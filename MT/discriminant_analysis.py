import pandas as pd
import numpy as np
import pdb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.linear_model import LinearRegression 

# df = pd.read_csv('health.csv')
df = pd.read_csv('drinking.csv')
X = df.loc[:,("days","cigar")]
Y = df.loc[:,("healthy")]
lda = LinearDiscriminantAnalysis()
lda.fit(X,Y)
print("Discriminant Analsis")
print(lda.coef_)
print(lda.intercept_)
w = {"days":30,"cigar":40}
W = pd.DataFrame(w,index=[0])
print(lda.predict(W))

# model = LinearRegression()
# Y2 = [1 if y=='Y' else 0 for y in Y]
# model.fit(X,Y2)
# print("Linear multiple regression")
# print("coef",model.coef_)
# print("icept",model.intercept_)
# print("R2",model.score(X,Y2))

pdb.set_trace()