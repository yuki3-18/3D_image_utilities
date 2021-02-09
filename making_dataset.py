import os
import numpy as np
import SimpleITK as sitk
import dataIO as io
import argparse
from tqdm import trange
import pandas as pd
from utils import rank_norm

def main():
    parser = argparse.ArgumentParser(description='py, in, out, num')
    parser.add_argument('--indir', '-i1', default="E:/git/TFRecord_example/input/CT/patch/size9/", help='input directory')
    parser.add_argument('--side', '-i2', type=int, default=9, help='patch side size')
    parser.add_argument('--num_of_data', '-i3', type=int, default=3039, help='number of the input data')
    args = parser.parse_args()

    # check folder
    indir = args.indir
    outdir = os.path.join(indir, "cc1/")
    if not (os.path.exists(outdir)):
        os.makedirs(outdir + 'test')
        os.makedirs(outdir + 'val')
        os.makedirs(outdir + 'train')
    num_of_data = args.num_of_data
    side = args.side

    # load data
    print('load data')
    data_set = np.zeros((num_of_data, side, side, side))
    # file = open(outdir + 'filename.txt', 'w')
    list = []
    with open(indir + 'filename.txt', 'rt') as f:
        i = 0
        for line in f:
            if i >= num_of_data:
                break
            line = line.split()
            sitkdata = sitk.ReadImage(line[0])
            # data = sitk.GetArrayFromImage(sitkdata)
            data = np.reshape(io.read_mhd_and_raw(line[0]), [side, side, side])
            data_set[i, :] = data
            list.append(line[0])
            i += 1
            print(i)
            # for x in line:
            #     file.write(str(x) + "/n")
    # file.close()
    # print(list)

    # load mask
    # topo = pd.read_csv(indir + "topo.csv", header=None).values.tolist()
    # print(topo)
    # topo = np.loadtxt(indir + "topo.csv", delimiter=",", dtype="unicode")
    # topo = [flatten for inner in topo for flatten in inner]

    # file = open(outdir + "filename.txt", mode='w')

    # Normalization
    data_set = rank_norm(data_set.reshape(num_of_data, side * side * side))
    data_set = data_set.reshape(num_of_data, side, side, side)

    for i in trange(len(data_set)):
        # if topo[i] == 1:
            # print(i)
            eudt_image = sitk.GetImageFromArray(data_set[i].reshape(side, side, side))
            eudt_image.SetSpacing(sitkdata.GetSpacing())
            eudt_image.SetOrigin(sitkdata.GetOrigin())
            if i <= 602: folder = "test/"
            elif i <= 602*2: folder = "val/"
            else: folder = "train/"
            sitk.WriteImage(eudt_image, os.path.join(outdir , folder, "{}.mhd".format(str(i).zfill(4))))
            file = open(outdir + folder + "filename.txt", mode='a')
            file.write(os.path.join(outdir , folder, "{}.mhd".format(str(i).zfill(4)) + "\n"))

    file.close()


if __name__ == '__main__':
    main()