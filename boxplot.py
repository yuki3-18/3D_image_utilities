import os
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# setting
data = "CT"
index = "topo"
file_in = "./input/{}/{}_pre.csv".format(data, index)
file_in2 = "./input/{}/{}_fine.csv".format(data, index)

pro = 'Pre.'
con = 'Fine.'

# load data
if index == "topo":
    df = pd.read_csv(file_in, sep=',', na_values=".", header=None)
    df.columns = ['T0', 'T1', 'T2', 'T0*', 'T1*', 'T2*']
    df2 = pd.read_csv(file_in2, sep=',', na_values=".", header=None)
    df2.columns = ['T0', 'T1', 'T2', 'T0*', 'T1*', 'T2*']
if index == "ph":
    df = pd.read_csv(file_in, sep=',', na_values=".", header=None)
    df2 = pd.read_csv(file_in2, sep=',', na_values=".", header=None)
    df.columns = df2.columns = ['Topological Loss']
elif index == "int":
    df = pd.read_csv(file_in, sep=',', na_values=".", header=None)
    df.columns = ['Generalization', 'Specificity']
    df2 = pd.read_csv(file_in2, sep=',', na_values=".", header=None)
    df2.columns = ['Generalization', 'Specificity']

# hypothesis testing
for i in range(len(df.columns)):
    if df[df.columns[i]].all() != 0:
        w1 = stats.wilcoxon(df[df.columns[i]], df2[df2.columns[i]])
        w2 = stats.wilcoxon(df[df.columns[i]], df2[df2.columns[i]], correction=True)
        m = stats.mannwhitneyu(df[df.columns[i]], df2[df2.columns[i]], alternative='two-sided')
        print(df.columns[i])
        print(w1)
        print(w2)
        print(m)

if index == "gen":
    title = "Generalization"
    p = w1[1]
elif index == "spe":
    title = "Specificity"
    p = m[1]
elif index == "int":
    title = "Intensity error"
    p = m[1]
else:
    title = "Topology error"
    p = w2[1]

# merge the two data frames to one data frame
df_melt = pd.melt(df)
df_melt['method'] = pro
df2_melt = pd.melt(df2)
df2_melt['method'] = con

df3 = pd.concat([df_melt, df2_melt], axis=0)
print(df3.head())

# visualize
file_out = "./output/{}/".format(data)
os.makedirs(file_out, exist_ok=True)
# df.plot.box()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
sns.boxplot(x='variable', y='value', data=df3, hue='method', palette='Dark2', ax=ax)
plt.title(title, fontname='SegoeUI')
# plt.ylabel('L2-Norm')
# plt.savefig(file_out + "{}_{:.3f}.png".format(index, p))
plt.show()
