import SimpleITK
import dataIO as io
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import ndimage
from sklearn.utils import shuffle
import os


from extract import SNR
import utils

# setting
size = 9

num_data = 517

# "E:/git/beta-VAE/output/CT/patch/model2/z24/alpha_1e-5/beta_0.1/gen/EUDT/filename.txt""C:/Users/saeki/PycharmProjects/3D/input/list.txt" "E:/git/TFRecord_example/in/axis1/noise/test/filename.txt""E:/git/TFRecord_example/input/shift/val/filename.txt"
# input_file = "E:/git/3d-glow/data/1axis/filename.txt""E:/git/pytorch/vae/input/s200/filename.txt"
input_file = "E:/git/pytorch/vae/input/th_150/rank/filename.txt"
# input_file = "E:/git/pytorch/vae/input/test/CT/filename.txt"
out_dir = os.path.join(os.path.dirname(input_file), 'check')
os.makedirs(out_dir, exist_ok=True)
# get data
data_set = utils.get_dataset(input_file, size, num_data)
data = shuffle(data_set, random_state=2)

# data_set = utils.min_max(data_set)
patch = data_set[10]

# print(np.average(patch))
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

# SNR(data_set[1]-np.min(data_set[1]))

# utils.display_slices(data_set, size, 3)
# utils.display_image(patch, 9, np.min(patch), np.max(patch))

# utils.display_center_slices(data_set, size, 5, np.min(data_set), np.max(data_set))
# utils.display_center_slices(data, size, 5, np.min(data), np.max(data))

for s in range(size):
    utils.display_any_slices(data, size, 5, 5, np.min(data_set), np.max(data_set), s, out_dir, 'slice{}'.format(s))
    # utils.display_any_slices(data, size, 10, 5, np.min(data_set), np.max(data_set), s)