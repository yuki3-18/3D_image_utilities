import SimpleITK
import dataIO as io
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import ndimage
import utils

# setting
size = 9

num_data = 100

# "E:/git/beta-VAE/output/CT/patch/model2/z24/alpha_1e-5/beta_0.1/gen/EUDT/filename.txt""C:/Users/saeki/PycharmProjects/3D/input/list.txt" "E:/git/TFRecord_example/in/axis1/noise/test/filename.txt""E:/git/TFRecord_example/input/shift/val/filename.txt"
# input_file = "E:/git/3d-glow/data/1axis/filename.txt"
input_file = "E:/git/TFRecord_example/in/new/patch/th_150/size_9/filename.txt"
# load data
print('load data')
case = np.zeros((num_data, size, size, size))

with open(input_file, 'rt') as f:
    i = 0
    for line in f:
        if i >= num_data:
            break
        line = line.split()
        case[i, :] = io.read_mhd_and_raw(line[0])
        i += 1

# fig, axes = plt.subplots(ncols=9, nrows=8, figsize=(8, 10))

min = np.min(case)
max = np.max(case)

# img = []
# img.append(case[438])
# img.append(case[466])
# img.append(case[473])
# img.append(case[560])
# img.append(case[573])
# img = np.reshape(img, [5, size, size, size])

# utils.display_slices(img, size, 5)

utils.display_center_slices(case, size, 5)

# for i in range(num_data - 1):
#     for j in range(x_size - 1):
#         # print(X[i, :].shape)
#         # print(X[i, :])
#         axes[j, i].imshow(case[j, :, :, i].reshape(9, 9), cmap=cm.Greys_r, vmin = min, vmax= max, interpolation='none')
#         # axes[0,i].imshow(X1[i, :].reshape(9, 9), cmap=cm.Greys_r)
#         # axes[1,i].imshow(X2[i, :].reshape(9, 9), cmap=cm.Greys_r)
#         axes[j, i].set_title('x = %d' % i)
#         axes[j ,i].get_xaxis().set_visible(False)
#         axes[j ,i].get_yaxis().set_visible(False)
#
# # plt.savefig(FLAGS.outdir + "reconstruction.png")
# plt.show()
#
#
# for i in range(num_data - 1):
#     for j in range(y_size - 1):
#         axes[j, i].imshow(case[j, :, i, :].reshape(9, 9), cmap=cm.Greys_r, vmin = min, vmax= max, interpolation='none')
#         axes[j, i].set_title('y = %d' % i)
#         axes[j ,i].get_xaxis().set_visible(False)
#         axes[j ,i].get_yaxis().set_visible(False)
#
# plt.show()
#
#
# for i in range(num_data - 1):
#     for j in range(z_size - 1):
#         axes[j, i].imshow(case[j, i, :, :].reshape(9, 9), cmap=cm.Greys_r, vmin = min, vmax= max, interpolation='none')
#         axes[j, i].set_title('z = %d' % i)
#         axes[j ,i].get_xaxis().set_visible(False)
#         axes[j ,i].get_yaxis().set_visible(False)
#
# plt.show()
