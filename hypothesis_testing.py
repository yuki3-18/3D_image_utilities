import os
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# setting
data = "CT"
index = "spe"
file_in = "./input/{}/{}.csv".format(data, index)

pro = 'Pre.'
con = 'Fine.'

# load data
df = pd.read_csv(file_in, sep=',', na_values=".", header=None)
df.columns = [pro, con]

# hypothesis testing
w1 = stats.wilcoxon(df[pro], df[con])
w2 = stats.wilcoxon(df[pro], df[con], correction=True)
m = stats.mannwhitneyu(df[pro], df[con], alternative='two-sided')

if index == "gen":
    title = "Generalization"
    p = w1[1]
elif index == "spe":
    title = "Specificity"
    p = m[1]
else:
    title = index
    p = w2[1]

print(w1)
print(w2)
print(m)

# visualize
file_out = "./output/{}/".format(data)
os.makedirs(file_out, exist_ok=True)
# df.plot.box()
fig = plt.figure()
averages = df.mean()

ax = fig.add_subplot(1, 1, 1)
x = sns.boxplot(data=df, palette='Dark2', ax=ax)

num_boxes = len(df.columns)
pos = np.arange(num_boxes)
upper_labels = [str(np.round(s, 5)) for s in averages]
min = df.min().min()
max = df.max().max()
plt.ylim([0, 0.08])

if averages[0] < averages[1]:
    weights = ['semibold', 'normal']
else:
    weights = ['normal', 'semibold']

for tick, label in zip(range(num_boxes), ax.get_xticklabels()):
    k = tick % 2
    print(k)
    ax.text(pos[tick], .95, upper_labels[tick],
            transform=ax.get_xaxis_transform(),
            horizontalalignment='center', weight=weights[k])

# plt.text(df.mean()[0], x, x)
plt.savefig(file_out + "{}_{:.3f}.png".format(index, p))
plt.title(title)
plt.ylabel('L2-Norm')
plt.show()
plt.close()