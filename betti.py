from utils import *
import dataIO as io
import argparse
from skimage import filters
import torch
import torch.nn as nn

parser = argparse.ArgumentParser(description='Betti')
parser.add_argument('--input', type=str,
                    default="E:/data/Tokushima/Lung/all/patch/rank/filename.txt",
                    help='File path of input images')
parser.add_argument('--patch_side', type=int, default=9,
                    help='how long patch side for input')
parser.add_argument('--num_of_data', type=int, default=500,
                    help='number of dataset')
args = parser.parse_args()

# set path
num = args.num_of_data
patch_side = args.patch_side
data_path = args.input
out_path = os.path.dirname(args.input)

os.makedirs(out_path, exist_ok=True)

# get data
data = get_dataset(data_path, patch_side, num)
tensor = torch.from_numpy(data)
m = nn.ConstantPad3d(1, 1.0)
frm_t = m(tensor.view(num, 1, patch_side, patch_side, patch_side))
frm_d = frm_t.view(num, patch_side + 2, patch_side + 2, patch_side + 2).numpy()

c = 0

for i in trange(num):
    # img
    img = data[i]
    frm = frm_d[i]
    # thresholding
    th = filters.threshold_otsu(img)
    # th = 0.5
    bin = thresh(img, th)
    frm_bin = thresh(frm, th)
    # compute betti
    betti = compute_Betti_bumbers(bin, th)
    betti_f = compute_Betti_bumbers(frm_bin, th)
    if betti == [1, 0, 0, 0] and betti_f == [1, 2, 1, 0]:
        c += 1
        # print(c)
        print(i)
        display_slices(img)