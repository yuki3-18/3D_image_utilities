import SimpleITK
import ioFunction_version_4_3 as IO
import numpy as np
from scipy import ndimage

# load data
print('load data')

z_size = 250
y_size = 512
x_size = 512
size = x_size*y_size*z_size
img = IO.read_raw("E:/data/lung.raw",dtype="short")

org = np.zeros(x_size*y_size*z_size)
org[:] = img.reshape(size)
print(img.shape)

org = org.reshape(z_size,y_size,x_size)

sizel = 35
# startl = (sizel - 1) / 2
# endl = (sizel + 1) / 2
x1,y1,z1 = 164, 243, 81
x2,y2,z2 = 181, 173, 124
x3,y3,z3 = 189, 232, 132

sizem = 25
# startm = (sizem - 1) / 2
# endm = (sizem + 1) / 2
x4,y4,z4 = 151, 244, 148
x5,y5,z5 = 151, 182, 121
x6,y6,z6 = 189, 193, 96

sizes = 9
# starts = (sizes - 1) / 2
# ends = (sizes + 1) / 2
x7,y7,z7 = 185, 136, 160
x8,y8,z8 = 147, 148, 143
x9,y9,z9 = 145, 152, 131

# x,y,z = 145, 187, 104
# x,y,z = 140, 150, 136
x,y,z = 245, 215, 125

# # large
# patch1 = org[z1-17:z1+18,y1-17:y1+18,x1-17:x1+18]
# patch1 = patch1.reshape(sizel*sizel*sizel)
#
# patch2 = org[z2-17:z2+18,y2-17:y2+18,x2-17:x2+18]
# patch2 = patch2.reshape(sizel*sizel*sizel)
#
# patch3 = org[z3-17:z3+18,y3-17:y3+18,x3-17:x3+18]
# patch3 = patch3.reshape(sizel*sizel*sizel)
#
#
# # medium
# patch4 = org[z4-12:z4+13,y4-12:y4+13,x4-12:x4+13]
# patch4 = patch4.reshape(sizem*sizem*sizem)
#
# patch5 = org[z5-12:z5+13,y5-12:y5+13,x5-12:x5+13]
# patch5 = patch5.reshape(sizem*sizem*sizem)
#
# patch6 = org[z6-12:z6+13,y6-12:y6+13,x6-12:x6+13]
# patch6 = patch6.reshape(sizem*sizem*sizem)


# small
# patch7 = org[z3-7:z3+8,y3-7:y3+8,x7-7:x7+8]
# patch7 = patch7.reshape(sizes*sizes*sizes)
#
# patch8 = org[z8-7:z8+8,y8-7:y8+8,x8-7:x8+8]
# patch8 = patch8.reshape(sizes*sizes*sizes)
#
# patch9 = org[z9-7:z9+8,y9-7:y9+8,x9-7:x9+8]
# patch9 = patch9.reshape(sizes*sizes*sizes)

# patch = org[z-4:z+5,y-4:y+5,x-4:x+5]
# patch = patch.reshape(sizes*sizes*sizes)

patch = org[z-125:z+125,y-150:y+150,x-150:x+150]
patch = patch.reshape(z_size*300*300)
# print(patch1)
print(patch.shape)
print(patch.dtype)
IO.write_raw(patch.astype('short'),"E:/data/data1.raw")
# IO.write_raw(patch1.astype('short'),"E:/data/patch/patch_L1.raw")
# IO.write_raw(patch2.astype('short'),"E:/data/patch/patch_L2.raw")
# IO.write_raw(patch3.astype('short'),"E:/data/patch/patch_L3.raw")
# IO.write_raw(patch4.astype('short'),"E:/data/patch/patch_M1.raw")
# IO.write_raw(patch5.astype('short'),"E:/data/patch/patch_M2.raw")
# IO.write_raw(patch6.astype('short'),"E:/data/patch/patch_M3.raw")
# IO.write_raw(patch7.astype('short'),"E:/data/patch/patch_S1.raw")
# IO.write_raw(patch8.astype('short'),"E:/data/patch/patch_S2.raw")
# IO.write_raw(patch9.astype('short'),"E:/data/patch/patch_S3.raw")

print('write img')