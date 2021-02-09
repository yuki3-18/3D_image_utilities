import os
import argparse
import numpy as np
import SimpleITK as sitk
import dataIO as io
from utils import cropping
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Settings')
    parser.add_argument('--root', type=str,
                        default="E:/data/Tokushima/",
                        help='root path')
    parser.add_argument('--data_n', type=int, default=1,
                        help='index of data')
    parser.add_argument('--org', type=str,
                        default="float/",
                        help='target organ')

    args = parser.parse_args()

    # settings
    # data_n = str(args.data_n).zfill(2)
    # img_path = os.path.join(args.root, "Fukui_Ai-CT_Sept2015/Fukui_Ai-CT_Sept2015_{}-2.mhd".format(data_n))
    # mask_path = os.path.join(args.root, "Fukui_Ai-CT_2015_Label/{}/Fukui_Ai-CT_Sept2015_{}-2_{}.mhd".format(args.org, data_n, args.org))

    # check folder
    w_path = os.path.join(args.root, "{}/".format(args.org))
    os.makedirs(w_path, exist_ok=True)

    case_list = io.load_list(args.root + 'filename.txt')

    for i in case_list:
        img_path = os.path.join(args.root, "CT", i)
        out_path = os.path.join(w_path, os.path.basename(img_path))
        mask_path = os.path.join(args.root, "Label", i)

        # loading data
        print("-" * 20, 'Loading data', "-" * 20)
        print(img_path)
        if os.path.isfile(img_path):

            sitkimg = sitk.ReadImage(img_path, sitk.sitkInt16)
            img = sitk.GetArrayFromImage(sitkimg)
            mask = io.read_mhd_and_raw(mask_path)

            # masking
            img = np.where((mask == 1) | (mask == 2), img, -2000)
            img = np.array(img, dtype='float32')

            # cropping
            idx = np.where(img != -2000)
            z, y, x = idx
            img = cropping(img, np.min(x), np.max(x), np.min(y), np.max(y), np.min(z), np.max(z))

            # plt.imshow(img[int(np.mean(z))], cmap='gray', interpolation=None)
            # plt.show()

            # saving img
            eudt_image = sitk.GetImageFromArray(img)
            eudt_image.SetSpacing(sitkimg.GetSpacing())
            eudt_image.SetOrigin(sitkimg.GetOrigin())
            sitk.WriteImage(eudt_image, out_path)
            print(out_path)


if __name__ == '__main__':
    main()
    print("-" * 20, 'Completed', "-" * 20)
