import pandas as pd
import pdb
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
from itertools import product

df = pd.read_csv("questionnaire.csv")
# Z1 = df[df['X1']==0][df['X2']==0][df['X3']==0][df['X4']==0][df['X5']==0][df['X6']==0]
# for i in range(len(df)):
#     df[i,:]
arr = np.array(df.loc[:,("Y1","Y2","Y3")])
Y1 = arr[:,0]
Y2 = arr[:,1]
Y3 = arr[:,2]
Y = np.hstack((Y1,Y2,Y3)).reshape((len(Y1)+len(Y2)+len(Y3),1))

df2 = df.loc[:,("X1","X2","X3","X4","X5","X6")]
arr2 = np.array(df2)
X = np.vstack((arr2,arr2,arr2))

merged = np.hstack((X,Y))
df = pd.DataFrame(merged,columns=('X1','X2','X3','X4','X5','X6','Y'))

model = linear_model.LinearRegression()
Y = df['Y']
X = df.loc[:,('X1','X2','X3','X4','X5','X6')]
model.fit(X,Y)
print("coef",model.coef_)
print("icept",model.intercept_)
R2 = model.score(X, Y)
print('R2=',R2)

patterns = list(product((0,1),(0,1),(0,1),(0,1),(0,1),(0,1)))
x = np.array(patterns)
y = model.predict(x)
y = y.reshape(len(y),1)
complete = np.hstack((x,y))
df_complete = pd.DataFrame(complete,columns=('X1','X2','X3','X4','X5','X6','Y'))
df_complete = df_complete.sort_values(by=['Y'],ascending=False)

print('**********************')
print('Optimal solution')
print('**********************')
print(df_complete.iloc[0])