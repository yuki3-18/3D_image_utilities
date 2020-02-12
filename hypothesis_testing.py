import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# setting

file_in = "E:\spe_otsu_0.1.csv"
file_out = "E:\spe_otsu_0.1.png"

pro='new'
con='old'


# load data
df = pd.read_csv(file_in, sep=',', na_values=".", header=None)
df.columns = [pro, con]

# hypothesis testing
w1 = stats.wilcoxon(df[pro], df[con])
w2 = stats.wilcoxon(df[pro], df[con], correction=True)
m = stats.mannwhitneyu(df[pro], df[con], alternative='two-sided')

print(w1)
print(w2)
print(m)

# visualize
df.plot.box()
plt.title('Specificity')
plt.ylabel('L1-Norm')
plt.savefig(file_out)
plt.show()