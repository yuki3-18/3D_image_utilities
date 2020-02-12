import os
import numpy as np
import SimpleITK as sitk
import argparse
import random

def main():
    parser = argparse.ArgumentParser(description='py, case_name, z_size, xy_resolution')
    parser.add_argument('--patch_side', '-i1', type=int, default=9, help='patch size')
    parser.add_argument('--th', '-i2', type=int, default=150, help='threshold of hessian')
    parser.add_argument('--is_shift', '-i3', type=bool, default=False, help='shift')
    args = parser.parse_args()

    patch_side = args.patch_side
    b = int(patch_side/2)
    e = int(patch_side/2 + 0.5)

    # check folder
    path_w = "E:/git/TFRecord_example/input/CT/patch/size{}/".format(patch_side)
    # path_w = "E:/git/TFRecord_example/input/CT/size{}th{}/".format(args.patch_side, args.th)
    if not (os.path.exists(path_w)):
        os.makedirs(path_w)

    # load data
    print('load data')
    sitkimg = sitk.ReadImage("E:/data/data1.mhd")
    img = sitk.GetArrayFromImage(sitkimg)
    sitkmask = sitk.ReadImage("E:/itk_hessian/x64/Release/OUTPUT/all/0.900000/lineness3/data1.mhd")
    mask = sitk.GetArrayFromImage(sitkmask)
    size = sitkimg.GetSize()
    x_size = size[0]
    y_size = size[1]
    z_size = size[2]
    img = np.reshape(img, [z_size, y_size, x_size])
    mask = np.reshape(mask, [z_size, y_size, x_size])

    # shift param
    d = 2.688
    dx = dy = dz = 0

    # make patch
    file = open(path_w + "filename.txt", mode='w')
    count = 0
    for z in range(z_size-1):
        for y in range(y_size-1):
            for x in range(x_size-1):
                if mask[z, y, x] > args.th and mask[z, y, x] > mask[z, y, x - 1] and mask[z, y, x] > mask[z, y, x + 1] \
                        and mask[z, y, x] > mask[z - 1, y, x] and mask[z, y, x] > mask[z + 1, y, x]\
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
                    patch = img[z-b+dz:z+e+dz, y-b+dy:y+e+dy, x-b+dx:x+e+dx]
                    patch = patch.reshape([patch_side, patch_side, patch_side])
                    if np.all(patch!=0):
                        eudt_image = sitk.GetImageFromArray(patch)
                        eudt_image.SetSpacing(sitkimg.GetSpacing())
                        eudt_image.SetOrigin(sitkimg.GetOrigin())
                        # sitk.WriteImage(eudt_image, os.path.join(path_w , "patch_{}_{}_{}.mhd".format(x+dx, y+dy, z+dz)))
                        sitk.WriteImage(eudt_image, os.path.join(path_w, "{}.mhd".format(str(count).zfill(4))))
                        # file.write(os.path.join(path_w ,"patch{}.mhd".format(x+dx, y+dy, z+dz) + "\n"))
                        file.write(os.path.join(path_w ,"{}.mhd".format(str(count).zfill(4)) + "\n"))
                        count += 1
                        print(str(count).zfill(4))
    file.close()


if __name__ == '__main__':
    main()