import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')

df = pd.read_csv('./data/annotations/list.txt', skiprows = 6, delimiter = ' ', header = None)
df.columns = ['file_name', 'id', 'species', 'breed']

# print(df['species'].value_counts().sort_index())

value_counts = df['species'].value_counts().sort_index()

plt.bar(range(len(value_counts)), value_counts.values, align = 'center')
plt.xticks(range(len(value_counts)), value_counts.index.values)

# plt.show()

value_counts = df['id'].value_counts().sort_index()

plt.bar(range(len(value_counts)), value_counts.values, align = 'center')
plt.xticks(range(len(value_counts)), value_counts.index.values)

# plt.show()

value_counts = df[df['species'] == 1]['breed'].value_counts().sort_index()

plt.bar(range(len(value_counts)), value_counts.values, align = 'center')
plt.xticks(range(len(value_counts)), value_counts.index.values)

plt.tight_layout()
# plt.show()



