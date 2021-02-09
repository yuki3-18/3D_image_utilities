import os
import numpy as np
import SimpleITK as sitk
import argparse
import random
from tqdm import trange
import dataIO as io


def main():
    parser = argparse.ArgumentParser(description='py, case_name, z_size, xy_resolution')
    parser.add_argument('--root', type=str, default="E:/data/Tokushima/", help='root path')
    parser.add_argument('--patch_side', '-i1', type=int, default=9, help='patch size')
    parser.add_argument('--th', '-i2', type=int, default=150, help='threshold of hessian')
    parser.add_argument('--is_shift', '-i3', type=bool, default=False, help='shift')
    args = parser.parse_args()

    patch_side = args.patch_side
    b = int(patch_side / 2)
    e = int(patch_side / 2 + 0.5)
    out_value = -2000
    hessian = []
    median = []
    average = []

    # check folder
    path_w = "E:/data/Tokushima/Lung/all/patch/"
    # path_w = "E:/git/TFRecord_example/input/CT/size{}th{}/".format(args.patch_side, args.th)
    if not (os.path.exists(path_w)):
        os.makedirs(path_w)

    case_list = io.load_list(args.root + 'filename.txt')

    # load data
    count = 0

    # shift param
    d = 2.688
    dx = dy = dz = 0

    file = open(path_w + "filename.txt", mode='w')

    for i in case_list:
        print('load data')
        img_path = os.path.join(args.root, "Lung/all/CT", i)
        mask_path = os.path.join(args.root, "Lung/all/hessian", i)
        if os.path.isfile(img_path):
            sitkimg = sitk.ReadImage(img_path, sitk.sitkInt16)
            img = sitk.GetArrayFromImage(sitkimg)
            spacing = sitkimg.GetSpacing()
            origin = sitkimg.GetOrigin()
            size = sitkimg.GetSize()
            sitkmask = sitk.ReadImage(mask_path)
            mask = sitk.GetArrayFromImage(sitkmask)
            x_size, y_size, z_size = size
            img = np.reshape(img, [z_size, y_size, x_size])
            mask = np.reshape(mask, [z_size, y_size, x_size])

            # make patch
            for z in trange(z_size - 1):
                for y in range(y_size - 1):
                    for x in range(x_size - 1):
                        if mask[z, y, x] >= args.th and mask[z, y, x] > mask[z, y, x - 1] and mask[z, y, x] > mask[
                            z, y, x + 1] \
                                and mask[z, y, x] > mask[z - 1, y, x] and mask[z, y, x] > mask[z + 1, y, x] \
                                and mask[z, y, x] > mask[z, y - 1, x] and mask[z, y, x] > mask[z, y + 1, x]:
                            # if 5 < x < 295 & 5 < y < 295 & 5 < z < 245:
                            if args.is_shift == True:
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

                            # patch = img[z-4+dz:z+5+dz, y-4+dy:y+5+dy, x-4+dx:x+5+dx]
                            patch = img[z - b + dz:z + e + dz, y - b + dy:y + e + dy, x - b + dx:x + e + dx]
                            if np.all(patch != out_value) and patch.size != 0:
                                max = np.max(patch)
                                med = np.median(patch)
                                avr = np.average(patch)
                                if max <= 80 and med <= -720:
                                    # save info
                                    median.append(med)
                                    # average.append(avr)
                                    hessian.append(mask[z, y, x])
                                    # save patch
                                    patch_img = np.array(patch, dtype='int16')
                                    patch_img = np.reshape(patch_img, [patch_side, patch_side, patch_side])
                                    eudt_image = sitk.GetImageFromArray(patch_img)
                                    eudt_image.SetSpacing(spacing)
                                    eudt_image.SetOrigin(origin)
                                    # sitk.WriteImage(eudt_image, os.path.join(path_w,
                                    #                                          "i_{}_{}_{}.mhd".format(x + dx, y + dy,
                                    #                                                                      z + dz)))
                                    sitk.WriteImage(eudt_image, os.path.join(path_w, "{}.mhd".format(str(count).zfill(4))))
                                    # file.write(os.path.join(path_w, "i_{}.mhd".format(x + dx, y + dy, z + dz) + "\n"))
                                    file.write(os.path.join(path_w, "{}.mhd".format(str(count).zfill(4)) + "\n"))
                                    count += 1
            # np.save(path_w + 'average.npy', average)

            print(str(count).zfill(4))
    file.close()
    np.save(path_w + 'hessian.npy', hessian)
    median_array = np.array(median, dtype='int16')
    np.save(path_w + 'median.npy', median_array)


if __name__ == '__main__':
    main()
