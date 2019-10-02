import os
import numpy as np
import SimpleITK as sitk
import dataIO as io
import argparse
from tqdm import trange

def main():
    parser = argparse.ArgumentParser(description='py, in, out, num')
    parser.add_argument('--indir', '-i1', default="E:/git/TFRecord_example/input/CT/th_150/", help='input directory')
    parser.add_argument('--side', '-i2', type=int, default=9, help='patch side size')
    parser.add_argument('--num_of_data', '-i3', type=int, default=3039, help='number of the input data')
    args = parser.parse_args()

    # check folder
    indir = args.indir
    outdir = os.path.join(indir, "normalize/")
    if not (os.path.exists(outdir)):
        os.makedirs(outdir)
    num_of_data = args.num_of_data
    side = args.side
    
    # load data
    print('load data')
    data_set = np.zeros((num_of_data, side, side, side))
    file = open(outdir + 'filename.txt', 'w')
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
            for x in line:
                file.write(str(x) + "\n")
    file.close()
    print(list)
    img = data_set.reshape(num_of_data, side * side * side)
    # file = open(outdir + "filename.txt", mode='w')
    img_max = np.max(img,axis=1)
    img_min = np.min(img,axis=1)
    # print('max = ',img_max)
    # print('min = ',img_min)

    # Normalization
    for i in trange(num_of_data):
        img[i] = (img[i] - img_min[i]) / (img_max[i] - img_min[i])

        eudt_image = sitk.GetImageFromArray(img[i].reshape(side, side, side))
        eudt_image.SetSpacing(sitkdata.GetSpacing())
        eudt_image.SetOrigin(sitkdata.GetOrigin())
        sitk.WriteImage(eudt_image, "{}".format(list[i]))
        # file.write(os.path.join(outdir ,"patch_{}_{}_{}.mhd" + "\n"))

    # file.close()

if __name__ == '__main__':
    main()