import SimpleITK
import ioFunction_version_4_3 as IO
import numpy as np
from scipy import ndimage

# load data
print('load data')
# img = np.zeros((original_dim))
# mask = np.zeros((train_data, original_dim))

size = 1072*512*512
img = IO.read_mhd_and_raw("E:/from_miyagawa/Tokushima/CT/t0000190_6.mhd")
mask = IO.read_mhd_and_raw("E:/from_miyagawa/Tokushima/Label/t0000190_6.mhd")

print(img)
print(mask.shape)
img = img.reshape(size)
mask = mask.reshape(size)

for i in range(size):
    if mask[i]==2 or mask[i]==1:
        img[i] = img[i]
    else:
        img[i] = np.random.normal(-913.6, 22.2)

# img = img.reshape(1072,512,512)
# mask = mask.reshape(1072,512,512)
print(img)
print(mask.shape)

IO.write_raw(img,"E:/data/masking2.raw")
print('write img')