import SimpleITK as sitk
import ioFunction_version_4_3 as IO
import dataIO as io
import numpy as np
from scipy import ndimage
from utils import cropping9
import argparse
import os


def main():

    parser = argparse.ArgumentParser(description='py, case_name, outdir')

    parser.add_argument('--in_txt', '-i1', default="E:/git/TFRecord_example/input/shift/train/filename.txt", help='img_text')

    parser.add_argument('--outdir', '-i2', default="E:/git/TFRecord_example/input/shift/train/", help='outdir')

    args = parser.parse_args()

    # check folder
    if not (os.path.exists(args.outdir)):
        os.makedirs(args.outdir)

    # setting
    case_num = 10000
    x_size = y_size = z_size = 15

    # resolution
    xr = yr = 0.885
    zr = 1
    # crop center
    xc = yc = zc = int((x_size) / 2)

    # reading
    data_set = np.zeros((case_num, z_size, y_size, x_size))

    line = []
    with open(args.in_txt, 'rt') as f:
        i = 0
        for line in f:
            if i >= case_num:
                break
            line = line.split()
            data = np.reshape(IO.read_mhd_and_raw(line[0]), [z_size, y_size, x_size])
            data_set[i, :] = data
            cropped = cropping9(data_set[i], x_size, y_size, z_size, xc, yc, zc)
            eudt_image = sitk.GetImageFromArray(cropped)
            eudt_image.SetOrigin([0, 0, 0])
            eudt_image.SetSpacing([xr, yr, zr])
            io.write_mhd_and_raw(eudt_image, os.path.join(args.outdir, line[0]))
            i += 1

if __name__ == '__main__':
    main()