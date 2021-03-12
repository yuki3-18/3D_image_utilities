import os
import numpy as np
import SimpleITK as sitk
import argparse
import utils

def main():
    parser = argparse.ArgumentParser(description='side')
    parser.add_argument('--patch_side', '-i1', type=int, default=9, help='patch size')
    args = parser.parse_args()

    patch_side = args.patch_side
    b = int(patch_side/2)
    e = int(patch_side/2 + 0.5)

    # check folder
    # path_w = "E:/git/TFRecord_example/input/CT/patch/size{}/".format(patch_side)
    path_w = "E:/itk_hessian/x64/Release/INPUT/"

    if not (os.path.exists(path_w)):
        os.makedirs(path_w)

    # load data
    print('load data')
    sitkimg = sitk.ReadImage("E:/data/Ai/Lung/Fukui_Ai-CT_Sept2015_01-2.mhd")
    img = sitk.GetArrayFromImage(sitkimg)
    size = sitkimg.GetSize()
    x_size = size[0]
    y_size = size[1]
    z_size = size[2]
    img = np.reshape(img, [z_size, y_size, x_size])

    c1 = [225, 120, 60]
    c2 = [97, 219, 188]
    c3 = [78, 53, 193]
    c4 = [73, 43, 191]
    c5 = [80, 47, 203]
    c6 = [98, 184, 39]
    c7 = [98, 200, 33]
    c8 = [40, 134, 68]
    c9 = [66, 150, 44]
    c10 = [78, 53, 193]
    # list = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10]
    list = [c7]
    # list = [c8, c9, c10]

    # file = open(path_w + "filename.txt", mode='w')

    count = 0
    for c in list:
        x, y, z = c
        patch = img[z-b:z+e, y-b:y+e, x-b:x+e]
        patch = patch.reshape([patch_side, patch_side, patch_side])
        eudt_image = sitk.GetImageFromArray(patch)
        eudt_image.SetSpacing(sitkimg.GetSpacing())
        eudt_image.SetOrigin(sitkimg.GetOrigin())
        # sitk.WriteImage(eudt_image, os.path.join(path_w , "{}_{}_{}.mhd".format(x, y, z)))
        # sitk.WriteImage(eudt_image, os.path.join(path_w, "{}.mhd".format(str(count).zfill(4))))
        # file.write(os.path.join(path_w ,"{}_{}_{}.mhd".format(x, y, z) + "\n"))
        # file.write(os.path.join(path_w ,"{}.mhd".format(str(count).zfill(4)) + "\n"))
        count += 1
        print(str(count).zfill(4))
        SNR(patch)
        utils.display_image(patch, patch_side)
    # file.close()

def SNR(data):
    avr = np.average(data)
    std = np.std(data)
    SNR = avr/std
    print("SNR={}, avr={}, std={}".format(abs(SNR), avr, std))
    print(10*np.log10(SNR**2), "[dB]")


if __name__ == '__main__':
    main()