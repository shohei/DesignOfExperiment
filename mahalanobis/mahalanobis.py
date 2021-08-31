import pandas as pd
import pdb

df = pd.read_csv('sales.csv')
team1_df = df[df['team']==1]
team2_df = df[df['team']==2]

sigma_item1_team1 = team1_df['item1'].std()
sigma_item1_team2 = team2_df['item1'].std()
mu_item1_team1 = team1_df['item1'].mean()
mu_item1_team2 = team2_df['item1'].mean()
mahalanobis_E = ((df[df['name']=='E'].item1 - mu_item1_team1) / sigma_item1_team1)**2
mahalanobis_K = ((df[df['name']=='K'].item1 - mu_item1_team2) / sigma_item1_team2)**2
print('mahalanobis distance E:',float(mahalanobis_E))
print('mahalanobis distance K:',float(mahalanobis_K))

