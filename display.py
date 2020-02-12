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

num_data = 6

# "E:/git/beta-VAE/output/CT/patch/model2/z24/alpha_1e-5/beta_0.1/gen/EUDT/filename.txt""C:/Users/saeki/PycharmProjects/3D/input/list.txt" "E:/git/TFRecord_example/in/axis1/noise/test/filename.txt""E:/git/TFRecord_example/input/shift/val/filename.txt"
# input_file = "E:/git/3d-glow/data/1axis/filename.txt"
input_file = "E:/git/pytorch/vae/input/s0/filename.txt"

# get data
data_set = utils.get_dataset(input_file, size, num_data)


# fig, axes = plt.subplots(ncols=9, nrows=8, figsize=(8, 10))

# min = np.min(data_set)
# max = np.max(data_set)

# img = []
# img.append(case[438])
# img.append(case[466])
# img.append(case[473])
# img.append(case[560])
# img.append(case[573])
# img = np.reshape(img, [5, size, size, size])

utils.display_slices(data_set, size, 3)
#
# utils.display_center_slices(data_set, size, 7)
