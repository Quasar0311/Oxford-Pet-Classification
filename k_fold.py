# Divide data into 5-fold
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
# from main import df
import pandas as pd


df = pd.read_csv('./data/annotations/list.txt', skiprows = 6, delimiter = ' ', header = None)
df.columns = ['file_name', 'id', 'species', 'breed']

kf = KFold(n_splits = 5, shuffle = True, random_state = 42)

df['fold'] = -1
for idx, (t, v) in enumerate(kf.split(df), 1):
    # t - Train data split , v - Validation data split
    print(t, v, len(v)) 
    df.loc[v, 'fold'] = idx

# Number of validation data in fold 1
print(len(df[df['fold'] == 1]))
# Number of train data in fold 1
print(len(df[df['fold'] != 1]))

# Split fairly by 'id' item
from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

df['fold'] = -1
for idx, (t, v) in enumerate(kf.split(df, df['id']), 1):
    # t - Train data split , v - Validation data split
    print(t, v, len(v)) 
    df.loc[v, 'fold'] = idx


value_counts = df[df['fold'] != 5]['id'].value_counts().sort_index()

plt.bar(range(len(value_counts)), value_counts.values, align = 'center')
plt.xticks(range(len(value_counts)), value_counts.index.values)

plt.tight_layout()
plt.show()


df.to_csv('data/kfolds.csv', index = False)