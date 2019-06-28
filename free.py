import dataIO as io
import numpy as np
import utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def func1(x, y):
        return np.cos(y)

x = np.arange(-1.57, 1.57, 0.1)
y = np.arange(-1.57, 1.57, 0.1)

X, Y = np.meshgrid(x, y)
Z = func1(X, Y)

fig = plt.figure()
ax = Axes3D(fig)

ax.set_xlabel("θ")
ax.set_ylabel("φ")
ax.set_zlabel("f(θ, φ)")

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, alpha=0.9)
# fig.colorbar(surf)
ax.plot_wireframe(X, Y, Z)
# ax.scatter(X, Y, Z, s=1)
plt.show()
# setting
# x_size = y_size = z_size = 9
# size = x_size * y_size * z_size
# in1 = "E:/git/TFRecord_example/in/new/patch/th_150/size_9/patch_73_176_41.mhd"
# in2 = "E:/git/beta-VAE/output/CT/patch/model2/z24/alpha_1e-5/beta_0.1/gen/EUDT/recon_104.mhd"
# in3 = "E:/git/pca/output/CT/patch/z24/reconstruction/EUDT/recon_105.mhd"
# #
# # in2 = "E:/git/beta-VAE/output/CT/patch/model2/z24/alpha_1e-5/beta_0.1/spe/EUDT/3215.mhd"
# # in3 = "E:/git/pca/output/CT/patch/z24/generate/EUDT/gen_4130.mhd"
# outdir = "E:/result"
#
# # load data
# print('load data')
# img1 = np.zeros((9, 9, 9))
# img2 = np.zeros((9, 9, 9))
# img3 = np.zeros((9, 9, 9))
# img1[:] = io.read_mhd_and_raw(in1)
# img2[:] = io.read_mhd_and_raw(in2)
# img3[:] = io.read_mhd_and_raw(in3)
#
# # print(img3.shape)
#
#
# utils.display_image(img1, img2, img3, z_size, outdir)
# utils.display_image2(img2, img3, z_size, outdir)
