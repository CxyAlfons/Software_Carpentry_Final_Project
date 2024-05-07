import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

act_pep = pd.read_csv("D:\\Py_Workspace\\Software Carpentry\\Software_Carpentry_Final_Project\\AMP0_data.csv")
act_pep = act_pep[act_pep['Target species'] == 'Escherichia coli']
act_pep = act_pep.drop_duplicates(subset=['Sequence'])

ina_file = "D:\\Py_Workspace\\Software Carpentry\\Software_Carpentry_Final_Project\\AMPlify_non_AMP_imbalanced.fa"
ina_pep = []
with open(ina_file, "r") as f:
    for line in f:
        if not line.startswith(">"):
            ina_pep.append(line.split("\n")[0])

act_len = []
for index, row in act_pep.iterrows():
    seq = row['Sequence']
    act_len.append(len(seq))

ina_len = []
for pep in ina_pep:
    ina_len.append(len(pep))

sns.histplot(ina_len, color="blue")
sns.histplot(act_len, color="red")
fig = plt.gcf()
fig.savefig('Peptide_Length_Distribution.png')
plt.clf()

mask = act_pep['Sequence'].str.len() <= 50
act_pep_fil = act_pep[mask]

ina_pep_fil = [s for s in ina_pep if len(s) <= 50]

act_len_fil = []
for index, row in act_pep_fil.iterrows():
    seq = row['Sequence']
    act_len_fil.append(len(seq))

ina_len_fil = []
for pep in ina_pep_fil:
    ina_len_fil.append(len(pep))

sns.histplot(ina_len_fil, color="blue")
sns.histplot(act_len_fil, color="red")
fig = plt.gcf()
fig.savefig('Filtered_Peptide_Length_Distribution.png')
plt.clf()

stats = [[-0.58, -0.65], [0.57, 0.58], [-9e-05, -4e-05], [0.62, 0.67], [0.75, 0.78], [0.58, 0.66]]
models = ['Linear', 'Ridge', 'Lasso', 'RF', 'SVR', 'KNN']

group_labels = []
categories = []
for i in range(6):
    group_label = models[i]
    group_labels.extend([group_label] * 2)
    categories.append('CV_mean')
    categories.append('FT')
flat_stats = [value for group in stats for value in group]
stat_data = pd.DataFrame({
    'Model': group_labels,
    'r2': flat_stats,
    'Category': categories
})

stat_data_new = pd.DataFrame({
    'Model': ['MLP', 'MLP', 'CRNN', 'CRNN'],
    'r2': [0.82, 0.85, 0.60, 0.67],
    'Category': ['CV_mean', 'FT', 'CV_mean', 'FT']
})
stat_data_all = pd.concat([stat_data, stat_data_new], ignore_index=True)

sns.barplot(x='Model', y='r2', hue='Category', data=stat_data_all, errorbar=None)
#plt.xticks(rotation=-60, fontsize=10)
plt.xlabel('Models', fontsize=14)
plt.ylabel('R2', fontsize=14)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, title_fontsize=12)
plt.tight_layout()
plt.savefig('Performance_Comparison.png', bbox_inches='tight')