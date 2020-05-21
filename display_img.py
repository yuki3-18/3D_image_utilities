import SimpleITK
import dataIO as io
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import ndimage
import utils, os

# outdir1 = "E:/result/cars/generalization/L1"
# outdir2 = "E:/result/ct_shift/pca"
#
# # check folder
# if not (os.path.exists(outdir1)):
#     os.makedirs(outdir1)

# img1 = io.read_mhd_and_raw("E:/git/TFRecord_example/in/new/patch/th_150/size_9/patch_73_176_41.mhd")
# img2 = io.read_mhd_and_raw("E:/git/beta-VAE/output/CT/patch/model2/z24/alpha_1e-5/beta_0.1/gen/EUDT/recon_104.mhd")
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
# utils.display_image2(img1, img3, 9, outdir1)

img = io.read_mhd_and_raw("E:/from_kubo/vector_rotation/x64/Release/output/output_3_7_2.mhd")
utils.display_image(img, 9)