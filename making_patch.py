import os
import dataIO as io
import numpy as np
import SimpleITK as sitk
import ioFunction_version_4_3 as IO
from scipy import ndimage
import argparse
import random

def main():
    parser = argparse.ArgumentParser(description='py, case_name, z_size, xy_resolution')

    parser.add_argument('--case_name', '-i1', default='t0000597_4', help='case_name')

    parser.add_argument('--z_size', '-i2', type=int, default=851, help='z_size')

    parser.add_argument('--res', '-i3', type=float, default=0.976, help='xy_resolution')

    args = parser.parse_args()

    z_size = 250
    y_size = 300
    x_size = 300
    size = x_size * y_size * z_size
    patch_side = 9
    w = (patch_side - 1) / 2
    threshold = 150

    path_w = "E:/git/TFRecord_example/input/CT/shift/th_{}/".format(threshold)

    # load data
    print('load data')
    img = io.read_mhd_and_raw("E:/data/data1.mhd")
    mask = io.read_mhd_and_raw("E:/itk_hessian/x64/Release/OUTPUT/all/0.900000/lineness3/data1.mhd")

    img = np.reshape(img, [z_size, y_size, x_size])
    mask = np.reshape(mask, [z_size, y_size, x_size])

    # check folder
    if not (os.path.exists(path_w)):
        os.makedirs(path_w)

    d = 2.688
    dx = dy = dz = 0

    file = open(path_w + "filename.txt", mode='w')
    count = 0
    for z in range(z_size-1):
        for y in range(y_size-1):
            for x in range(x_size-1):
                if mask[z, y, x] > threshold and mask[z, y, x] > mask[z, y, x - 1] and mask[z, y, x] > mask[z, y, x + 1] \
                        and mask[z, y, x] > mask[z - 1, y, x] and mask[z, y, x] > mask[z + 1, y, x]\
                        and mask[z, y, x] > mask[z, y - 1, x] and mask[z, y, x] > mask[z, y + 1, x]:
                    # if 5 < x < 295 & 5 < y < 295 & 5 < z < 245:
                    random.seed(a=None, version=2)
                    dx = random.uniform(-d, d)
                    if dx >= 0:
                        dx = int(dx + 0.5)
                    elif dx < 0:
                        dx = int(dx - 0.5)

                    random.seed(a=None, version=2)
                    dy = random.uniform(-d, d)
                    if dy >= 0:
                        dy = int(dy + 0.5)
                    elif dy < 0:
                        dy = int(dy - 0.5)

                    random.seed(a=None, version=2)
                    dz = int(random.uniform(-d, d) + 0.5)
                    if dz >= 0:
                        dz = int(dz + 0.5)
                    elif dz < 0:
                        dz = int(dz - 0.5)

                    patch = img[z-4+dz:z+5+dz, y-4+dy:y+5+dy, x-4+dx:x+5+dx]
                    # print(patch)
                    # patch = img[z-2:z+3, y-2:y+3, x-2:x+3]
                    patch = patch.reshape([patch_side, patch_side, patch_side])
                    if np.all(patch!=0):
                        eudt_image = sitk.GetImageFromArray(patch)
                        eudt_image.SetOrigin([0, 0, 0])
                        eudt_image.SetSpacing([0.885, 0.885, 1])
                        eudt_image.SetSpacing([args.res, args.res, 1])
                        io.write_mhd_and_raw(eudt_image, os.path.join(path_w , "patch_{}_{}_{}.mhd".format(x+dx, y+dy, z+dz)))
                        # patch = patch.reshape([9 *  9 * 9])
                        # io.write_mhd_and_raw(patch, "E:/data/data1_patch/data1_patch_{}".format(x) + "_{}".format(y) + "_{}.raw".format(z))
                        file.write(os.path.join(path_w ,"patch_{}_{}_{}.mhd".format(x+dx, y+dy, z+dz) + "\n"))
                        count += 1
                        print(count)

    file.close()
    # with open(path_w + "filename.txt") as f:
    #     print(f.read())
    # with open(os.path.join(path_w , "number.txt"), "r") as f:
    #     f.write(count)

if __name__ == '__main__':
    main()