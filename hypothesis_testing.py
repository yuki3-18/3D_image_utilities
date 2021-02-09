import os
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


# setting
data = "CT"
index = "spe"
file_in = "./input/{}/{}.csv".format(data, index)

pro='Pre.'
con='Fine.'

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
df.plot.box()
plt.title(title)
plt.ylabel('L2-Norm')
plt.savefig(file_out + "{}_{:.3f}.png".format(index, p))
plt.show()