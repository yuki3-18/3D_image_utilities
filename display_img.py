import SimpleITK
import dataIO as io
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import ndimage
import utils, os
import torch.nn

# outdir1 = "E:/result/cars/generalization/L1"
outdir2 = "./"
#
# # check folder
# if not (os.path.exists(outdir1)):
#     os.makedirs(outdir1)
# img = io.read_mhd_and_raw("E:/from_kubo/vector_rotation/x64/Release/output/output_5_5_2.mhd")
#
img1 = io.read_mhd_and_raw("E:/git/pytorch/vae/results/artificial/hole/z_6/B_0.1/batch128/L_60000/gen/ori/0001.mhd")
# img2 = io.read_mhd_and_raw("E:/git/pytorch/vae/results/artificial/tip/z_24/B_0.1/L_0/gen/rec/0000.mhd")
# # "E:/git/pca/output/CT/patch/z24/EUDT/recon_104.mhd"
#
# img1 = (img1 - np.min(img1))/ (np.max(img1) - np.min(img1))
# img3 = abs(img1 -img2)
# print(img3)
# print(np.max(img3))
# print(np.min(img3))

# ori=np.reshape(img1, [9,9,9])
# preds=np.reshape(img2, [9,9,9])
#
# # plot reconstruction

# utils.display_image(img1, img2, img3, 9, outdir1)
# utils.display_image2(img1, img2, 9, outdir2)

img = io.read_mhd_and_raw("E:/git/pytorch/vae/input/hole0/std/0000.mhd")
footprint = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                     [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                     [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
result = ndimage.minimum_filter(img, footprint=footprint)
th = 0.5
img = (img > th) * 1
result = (result > th) * 1


utils.display_image(img, 9)
utils.display_image(result, 9)