import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pylab as plb
from scipy.optimize import curve_fit
from scipy import stats
from scipy.stats import norm
import seaborn as sns
import ioFunction_version_4_3 as IO

img = IO.read_raw("E:/data/bkg/bkg2.raw", 'short')
img = np.reshape(img, (-1,1))
img = np.reshape(img,(9,9,9))
img = img[2:7,2:7,2:7]
img = np.reshape(img,(125,))
print(img.shape)
print(np.var(img))
print(np.std(img))
print(np.mean(img))
# plt.plot(img)
plt.hist(img,bins=60)
plt.show()
plt.plot(img)
plt.show()

# best fit of data
# (mu, sigma) = norm.fit(img)

# the histogram of the data
# n, bins, patches = plt.hist(img, 60, normed=1, facecolor='green', alpha=0.75, label="Histogram")

# add a 'best fit' line
# y = mlab.normpdf(bins, mu, sigma)
# l = plt.plot(bins, y, 'r--', linewidth=2, label="Gaussian")

#plot
# plt.xlabel('CT value')
# plt.ylabel('Probability')
# plt.title(r'$\mathrm{Parameters:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))
# plt.legend()
# plt.grid(True)
# plt.savefig("E:/data/bkg/bkg4.png")
# plt.show()


