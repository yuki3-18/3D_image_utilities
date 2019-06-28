import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# setting

file_in = "E:/result/specificity/new_vae_spe.csv"
file_out = "E:/result/hypothesis_test/new_vae_gen.png"

# load data
df = pd.read_csv(file_in, sep=',', na_values=".", header=None)
df.columns = ['β-VAE', 'PCA']

# hypothesis testing
w1 = stats.wilcoxon(df['β-VAE'], df['PCA'])
w2 = stats.wilcoxon(df['β-VAE'], df['PCA'], correction=True)
m = stats.mannwhitneyu(df['β-VAE'], df['PCA'], alternative='two-sided')

print(w1)
print(w2)
print(m)

# visualize
df.plot.box()
plt.title('Generalization')
plt.ylabel('L1-Norm')
plt.savefig(file_out)
plt.show()