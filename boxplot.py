import os
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# setting
data = "bif"
index = "int"

if data == "hole":
    pro = 'Pre.'
    con = 'Fine.'
    con2 = 'Fine. (Erosion)'
    file_in = "./input/{}/{}_pre.csv".format(data, index)
    file_in2 = "./input/{}/{}_fine.csv".format(data, index)
    file_in3 = "./input/{}/{}_fine_e.csv".format(data, index)
else:
    pro = 'Pre.'
    con = 'Fine.'
    file_in = "./input/{}/{}_pre.csv".format(data, index)
    file_in2 = "./input/{}/{}_fine.csv".format(data, index)
    file_in3 = None

df = pd.read_csv(file_in, sep=',', na_values=".", header=None)
df2 = pd.read_csv(file_in2, sep=',', na_values=".", header=None)

# load data
if index == "topo":
    if data == "hole":
        df.columns = df2.columns = ['T0', 'T1', 'T2', 'T0*']
    else:
        df.columns = ['T0', 'T1', 'T2', 'T0*', 'T1*', 'T2*']
        df2.columns = ['T0', 'T1', 'T2', 'T0*', 'T1*', 'T2*']
elif index == "BD" or index == "BD_gs":
    df.columns = ['k=0', 'k=1', 'k=2']
    df2.columns = ['k=0', 'k=1', 'k=2']
elif index == "BD_sum" or index == "BD_sum_gs":
    df.columns = ['Bottleneck distance']
    df2.columns = ['Bottleneck distance']
elif index == "ph":
    df.columns = df2.columns = ['Topological Loss']
elif index == "int" or index == "int_gs":
    df.columns = ['Generalization', 'Specificity']
    df2.columns = ['Generalization', 'Specificity']

if file_in3:
    df3 = pd.read_csv(file_in3, sep=',', na_values=".", header=None)
    df3.columns = df.columns

# hypothesis testing
p = []
for i in range(len(df.columns)):
    if df.columns[i] != 'Specificity':
        if (df[df.columns[i]] - df2[df2.columns[i]]).all() != 0:
            w1 = stats.wilcoxon(df[df.columns[i]], df2[df2.columns[i]])
            w2 = stats.wilcoxon(df[df.columns[i]], df2[df2.columns[i]], correction=True)
            # m = stats.mannwhitneyu(df[df.columns[i]], df2[df2.columns[i]], alternative='two-sided')
            # print(df.columns[i])
            print(w1)
            print(w2)
            p.append(w1[1])
        if file_in3:
            w1 = stats.wilcoxon(df[df.columns[i]], df3[df3.columns[i]])
            w2 = stats.wilcoxon(df2[df2.columns[i]], df3[df3.columns[i]])
            print(w1)
            print(w2)
            p.append(w1[1])
    else:
        m = stats.mannwhitneyu(df[df.columns[i]], df2[df2.columns[i]], alternative='two-sided')
        print("1-2: ", m)
        p.append(m[1])
        if file_in3:
            m1 = stats.mannwhitneyu(df[df.columns[i]], df3[df3.columns[i]], alternative='two-sided')
            m2 = stats.mannwhitneyu(df2[df.columns[i]], df3[df3.columns[i]], alternative='two-sided')
            print("1-3: ", m1)
            print("2-3: ", m2)
            p.append(m[1])

if index == "int":
    title = "Intensity error"
elif index == "int_gs":
    title = "Intensity error (no hole)"
elif index == "BD_gs" or index == "BD_sum_gs":
    title = "Topology error (no hole)"
else:
    title = "Topology error"

# merge the two data frames to one data frame
df_melt = pd.melt(df)
df_melt['method'] = pro
df2_melt = pd.melt(df2)
df2_melt['method'] = con

if file_in3:
    df3_melt = pd.melt(df3)
    df3_melt['method'] = con2
    df_a = pd.concat([df_melt, df2_melt, df3_melt], axis=0)
else:
    df_a = pd.concat([df_melt, df2_melt], axis=0)
print(df_a.head())

# visualize
file_out = "./output/{}/".format(data)
os.makedirs(file_out, exist_ok=True)
# df.plot.box()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
x = sns.boxplot(x='variable', y='value', data=df_a, hue='method', palette='Dark2', ax=ax)
plt.title(title, fontname='SegoeUI')

if index == "int" or index == "int_gs":
    plt.ylabel('L2-Norm')
elif index == "BD" or index == "BD_gs":
    plt.ylabel('Bottleneck distance')

# averages = df.mean()
# print(averages)
# num_boxes = len(df.mean())
# pos = np.arange(num_boxes)
# upper_labels = [str(np.round(s, 5)) for s in averages]
# print(upper_labels)
min = df.min().min()
max = df.max().max()
plt.ylim([-0.05, 0.7])
# plt.ylim([min / 2, max * 1.2])
#
# for tick, label in zip(range(num_boxes), ax.get_xticklabels()):
#     k = tick % 2
#     ax.text(pos[tick], .95, upper_labels[tick],
#             transform=ax.get_xaxis_transform(),
#             horizontalalignment='center')

plt.savefig(file_out + "{}_{:.3f}.png".format(index, p[0]))
plt.show()
plt.close()
