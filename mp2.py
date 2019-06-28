import os
import dataIO as io
import numpy as np
import SimpleITK as sitk
import ioFunction_version_4_3 as IO
from scipy import ndimage
import argparse

parser = argparse.ArgumentParser(description='py, x, y, z')
parser.add_argument('--x', '-x', default='', help='x')
parser.add_argument('--y', '-y', default='', help='y')
parser.add_argument('--z', '-z', default='', help='z')
args = parser.parse_args()

z_size = 250
y_size = 300
x_size = 300
size = x_size * y_size * z_size
patch_side = 9
w = (patch_side - 1) / 2
threshold = 200

path_w = "E:/data/data1_patch/sigma_0.9/th_{}/size_{}/".format(threshold, patch_side)

# load data
print('load data')
img = io.read_mhd_and_raw("E:/data/data1.mhd")
mask = io.read_mhd_and_raw("E:/itk_hessian/x64/Release/OUTPUT/all/0.900000/lineness3/data1.mhd")

img = np.reshape(img, [z_size, y_size, x_size])
mask = np.reshape(mask, [z_size, y_size, x_size])

# check folder
if not (os.path.exists(path_w)):
    os.makedirs(path_w)

# print(img)
# print(mask)

file = open(path_w + "filename.txt", mode='w')
count = 0

if mask[args.z, args.y, args.x] > threshold and mask[args.z, args.y, args.x] > mask[args.z, args.y, args.x - 1] and mask[args.z, args.y, args.x] > mask[args.z, args.y, args.x + 1] \
        and mask[args.z, args.y, args.x] > mask[args.z - 1, args.y, args.x] and mask[args.z, args.y, args.x] > mask[args.z + 1, args.y, args.x]\
        and mask[args.z, args.y, args.x] > mask[args.z, args.y - 1, args.x] and mask[args.z, args.y, args.x] > mask[args.z, args.y + 1, args.x]:
    # if 5 < args.x < 295 & 5 < args.y < 295 & 5 < args.z < 245:
    patch = img[args.z-4:args.z+5, args.y-4:args.y+5, args.x-4:args.x+5]
    # print(patch)
    # patch = img[args.z-2:args.z+3, args.y-2:args.y+3, args.x-2:args.x+3]
    patch = patch.reshape([patch_side, patch_side, patch_side])
    eudt_image = sitk.GetImageFromArray(patch)
    eudt_image.SetOrigin([patch_side, patch_side, patch_side])
    eudt_image.SetSpacing([0.885, 0.885, 1])
    io.write_mhd_and_raw(eudt_image, os.path.join(path_w , "data1_patch_{}_{}_{}.mhd".format(args.x, args.y, args.z)))
    # patch = patch.reshape([9 *  9 * 9])
    # io.write_mhd_and_raw(patch, "E:/data/data1_patch/data1_patch_{}".format(args.x) + "_{}".format(args.y) + "_{}.raw".format(args.z))
    file.write(os.path.join(path_w ,"data1_patch_{}_{}_{}.mhd".format(args.x, args.y, args.z) + "\n"))
    count += 1
    print(count)

file.close()
# with open(path_w + "filename.txt") as f:
#     print(f.read())
# with open(os.path.join(path_w , "number.txt"), "r") as f:
#     f.write(count)