import matplotlib.pyplot as plt
from skimage import filters
from skimage import exposure
import dataIO as io
import numpy as np
from utils import get_dataset


def plot_hist(g_hist, th_v):
    plt.plot(g_hist)
    plt.axvline(x=th_v, color='red', label='otsu')
    plt.legend(loc='upper right')
    plt.title("histogram, otsu and ave value")
    plt.xlabel("brightness")
    plt.ylabel("frequency")
    plt.show()


print("load data")
path = "E:/data/Tokushima/Lung/t0000190_6.mhd"
img = io.read_mhd_and_raw(path)

target = np.where(img == -2000, False, img)
val = filters.threshold_otsu(target)
# val = -700

# hist, bins_center = exposure.histogram(img, nbins=10)
print(img.shape)
print(val)

slice = np.argmax(np.average(np.average(img, axis=2), axis=1), axis=0)
print(slice)
plt.figure(figsize=(9, 4))
plt.subplot(131)
plt.imshow(img[slice], cmap='gray', interpolation=None)
plt.axis('off')
plt.subplot(132)
plt.imshow(img[slice] > val, cmap='gray', interpolation=None)
plt.axis('off')
plt.subplot(133)
# plt.plot(bins_center, hist, lw=2)
plt.hist(img.flatten(), bins=100, range=(-1500, 500))
# plt.ylim(0, 1.4)
plt.axvline(val, color='k', ls='--')
plt.title("Histogram")
plt.xlabel("CT value")
plt.ylabel("frequency")
plt.tight_layout()
plt.show()

# plt.hist(img.flatten(), range=(-1500, 500))
# plt.axvline(val, color='k', ls='--')
# plt.show()
# hist = np.histogram(img.flatten())
# plot_hist(hist, val)
