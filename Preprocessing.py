import SimpleITK as sitk
import ioFunction_version_4_3 as IO
import dataIO as io
import numpy as np
from scipy import ndimage
from utils import masking, cropping320, cropping400
import argparse
import os


def main():

    parser = argparse.ArgumentParser(description='py, case_name, z_size, xy_resolution, outdir')

    # parser.add_argument('--img_path', '-i1', default='E:/from_miyagawa/Tokushima/CT/{}.mhd'.format(), help='img_path')
    #
    # parser.add_argument('--label_path', '-i2', default='E:/from_miyagawa/Tokushima/Label/t0000190_6.mhd', help='label_path')

    parser.add_argument('--case_name', '-i1', default='t0000597_4', help='case_name')

    parser.add_argument('--z_size', '-i2', type=int, default=851, help='z_size')

    parser.add_argument('--res', '-i3', type=float, default=0.976, help='xy_resolution')

    parser.add_argument('--outdir', '-i4', default='E:/data/lung/', help='outdir')


    args = parser.parse_args()

    # check folder
    if not (os.path.exists(args.outdir)):
        os.makedirs(args.outdir)

    img_path = 'E:/from_miyagawa/Tokushima/CT/{}.mhd'.format(args.case_name)
    label_path = 'E:/from_miyagawa/Tokushima/Label/{}.mhd'.format(args.case_name)

    # setting
    case_num = 2
    x_size = y_size = 512
    z_size = args.z_size
    # cropping size
    xc_size = yc_size = zc_size = 400
    # resolution
    xr = yr = args.res
    zr = 1
    # crop center
    xc, yc, zc = 256, 256, 200

    case_size = int(x_size * y_size * z_size)
    cp_size = int(zc_size * yc_size * xc_size)

    # reading
    img = IO.read_mhd_and_raw(img_path)
    label = IO.read_mhd_and_raw(label_path)

    # print(img.shape)
    masked_img = masking(img, x_size, y_size, z_size, label, 1, 2)
    # print(masked_img.shape)

    eudt_image = sitk.GetImageFromArray(masked_img)
    eudt_image.SetOrigin([0, 0, 0])
    eudt_image.SetSpacing([xr, yr, zr])
    io.write_mhd_and_raw(eudt_image, os.path.join(args.outdir, "masking/lung_{}.mhd".format(args.case_name)))

    cropped = cropping400(masked_img, x_size, y_size, z_size, xc, yc, zc)
    eudt_image = sitk.GetImageFromArray(cropped)
    eudt_image.SetOrigin([0, 0, 0])
    eudt_image.SetSpacing([xr, yr, zr])
    io.write_mhd_and_raw(eudt_image, os.path.join(args.outdir, "lung_{}.mhd".format(args.case_name)))

if __name__ == '__main__':
    main()