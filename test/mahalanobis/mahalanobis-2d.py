import pandas as pd
import pdb
import numpy as np

df = pd.read_csv('sales.csv')
team1_df = df[df['team']==1]
team2_df = df[df['team']==2]
covmat_team1 = np.cov(team1_df['item1'],team1_df['item2'],rowvar=0,bias=0)
covmat_team2 = np.cov(team2_df['item1'],team2_df['item2'],rowvar=0,bias=0)
mu_item1_team1 = team1_df['item1'].mean()
mu_item2_team1 = team1_df['item2'].mean()
mu_item1_team2 = team2_df['item1'].mean()
mu_item2_team2 = team2_df['item2'].mean()

x_A = np.vstack((np.array(df[df['name']=='A'].item1),np.array((df[df['name']=='A'].item2))))
x_mean_A = np.vstack((np.array(mu_item1_team1),np.array((mu_item2_team1))))
mn_A = (x_A - x_mean_A).transpose() @ np.linalg.inv(covmat_team1) @ (x_A - x_mean_A)

x_H = np.vstack((np.array(df[df['name']=='H'].item1),np.array((df[df['name']=='H'].item2))))
x_mean_H = np.vstack((np.array(mu_item1_team2),np.array((mu_item2_team2))))
mn_H = (x_H - x_mean_H).transpose() @ np.linalg.inv(covmat_team2) @ (x_H - x_mean_H)

print('mahalanobis distance A:',float(mn_A))
print('mahalanobis distance H:',float(mn_H))
pdb.set_trace()
