import os
import argparse
import numpy as np
import SimpleITK as sitk
import dataIO as IO
from utils import cropping

def main():
    parser = argparse.ArgumentParser(description='Settings')
    parser.add_argument('--root', type=str,
                        default="E:/data/Fukui_Ai-CT/",
                        help='root path')
    parser.add_argument('--data_n', type=int, default=1,
                        help='index of data')
    parser.add_argument('--org', type=str,
                        default="Lung",
                        help='target organ')
    parser.add_argument('--w_path', type=str,
                        default="E:/data/Ai/",
                        help='writing path')
    args = parser.parse_args()

    # settings
    data_n = str(args.data_n).zfill(2)
    img_path = os.path.join(args.root, "Fukui_Ai-CT_Sept2015/Fukui_Ai-CT_Sept2015_{}-2.mhd".format(data_n))
    mask_path = os.path.join(args.root, "Fukui_Ai-CT_2015_Label/{}/Fukui_Ai-CT_Sept2015_{}-2_{}.mhd".format(args.org, data_n, args.org))
    w_path = os.path.join(args.w_path, "{}/".format(args.org))

    # check folder
    if not (os.path.exists(w_path)):
        os.makedirs(w_path)

    # loading data
    print("-"*20, 'Loading data', "-"*20)
    sitkimg = sitk.ReadImage(img_path)
    img = sitk.GetArrayFromImage(sitkimg)
    mask = IO.read_mhd_and_raw(mask_path)

    # masking
    img = np.where(mask == 0, -2000, img)

    # cropping
    idx = np.where(img != -2000)
    z, y, x = idx
    img = cropping(img, np.min(x), np.max(x), np.min(y), np.max(y), np.min(z), np.max(z))

    # saving img
    eudt_image = sitk.GetImageFromArray(img)
    eudt_image.SetSpacing(sitkimg.GetSpacing())
    eudt_image.SetOrigin(sitkimg.GetOrigin())
    sitk.WriteImage(eudt_image, os.path.join(w_path, os.path.basename(img_path)))


if __name__ == '__main__':
    main()
    print("-"*20, 'Completed', "-"*20)