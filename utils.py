import os
import numpy as np
import SimpleITK as sitk
import ioFunction_version_4_3 as IO
import dataIO as io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable
from tqdm import trange

from scipy import ndimage

def masking(img, x_size, y_size, z_size, mask, mask_value1, mask_value2):
    size = x_size * y_size * z_size

    img = np.reshape(img, [z_size, y_size, x_size])
    mask = np.reshape(mask, [z_size, y_size, x_size])

    for z in range(z_size - 1):
        for y in range(y_size - 1):
            for x in range(x_size - 1):
                if mask[z, y, x] == mask_value1 or mask[z, y, x] == mask_value2:
                    img[z, y, x] = img[z, y, x]
                else:
                    img[z, y, x] = np.random.normal(-913.6, 22.2)

    masking_img = np.reshape(img, [z_size, y_size, x_size])

    return  masking_img

def cropping9(img, x_size, y_size, z_size, x, y, z):
    img = np.reshape(img, [z_size, y_size, x_size])
    cropped_img = img[z - 4:z + 5, y - 4:y + 5, x - 4:x + 5]
    cropped_img = cropped_img.reshape([9, 9, 9])

    return cropped_img

def cropping320(img, x_size, y_size, z_size, x, y, z):
    img = np.reshape(img, [z_size, y_size, x_size])
    cropped_img = img[z - 160:z + 160, y - 160:y + 160, x - 160:x + 160]
    print(cropped_img.shape)
    cropped_img = cropped_img.reshape([400, 400, 400])
    print(cropped_img.shape)

    return cropped_img

def cropping400(img, x_size, y_size, z_size, x, y, z):
    img = np.reshape(img, [z_size, y_size, x_size])
    cropped_img = img[z - 200:z + 200, y - 200:y + 200, x - 200:x + 200]
    print(cropped_img.shape)
    cropped_img = cropped_img.reshape([400, 400, 400])
    print(cropped_img.shape)

    return cropped_img


def making_patch(num, img_path, mask_path, patch_side, threshold):

    z_size = 320
    y_size = 320
    x_size = 320
    w = int (patch_side  / 2)

    path_w = "E:/data/data{}_patch/sigma_0.9/th_{}/size_{}/".format(num, threshold, patch_side)

    # load data
    print('load data')
    img = io.read_mhd_and_raw(img_path)
    mask = io.read_mhd_and_raw(mask_path)

    img = np.reshape(img, [z_size, y_size, x_size])
    mask = np.reshape(mask, [z_size, y_size, x_size])

    # check folder
    if not (os.path.exists(path_w)):
        os.makedirs(path_w)

    file = open(path_w + "filename.txt", mode='w')
    count = 0
    for z in range(z_size - 1):
        for y in range(y_size - 1):
            for x in range(x_size - 1):
                if mask[z, y, x] > threshold and mask[z, y, x] > mask[z, y, x - 1] and mask[z, y, x] > mask[z, y, x + 1] \
                        and mask[z, y, x] > mask[z - 1, y, x] and mask[z, y, x] > mask[z + 1, y, x] \
                        and mask[z, y, x] > mask[z, y - 1, x] and mask[z, y, x] > mask[z, y + 1, x]:
                    patch = img[z - w:z + w + 1, y - w:y + w + 1, x - w:x + w + 1]
                    patch = patch.reshape([patch_side, patch_side, patch_side])
                    eudt_image = sitk.GetImageFromArray(patch)
                    eudt_image.SetOrigin([patch_side, patch_side, patch_side])
                    eudt_image.SetSpacing([0.885, 0.885, 1])
                    io.write_mhd_and_raw(eudt_image, os.path.join(path_w, "patch_{}_{}_{}.mhd".format(x, y, z)))
                    file.write(os.path.join(path_w, "data1_patch_{}_{}_{}.mhd".format(x, y, z) + "\n"))
                    count += 1
                    print(count)

    return 0

def display_image(img, side, outdir):
    img = np.reshape(img, [side, side, side])
    max = np.max(img)
    min = np.min(img)
    fig, axes = plt.subplots(ncols=9, nrows=1, figsize=(10, 10))
    for i in range(side):
        axes[i].imshow(img[i].reshape(side, side), vmin=min, vmax=max, interpolation='none')
        axes[i].set_title('z = %d' % i)
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)

    # カラーバーの設定
    axpos = axes[1].get_position()
    cbar_ax = fig.add_axes([0.87, axpos.y0, 0.02, axpos.height])
    # norm = colors.Normalize(vmin=df['price'].min(), vmax=df['price'].max())
    mappable = ScalarMappable()
    mappable._A = []
    fig.colorbar(mappable, cax=cbar_ax)
    # 余白の調整
    plt.subplots_adjust(right=0.85)
    plt.subplots_adjust(wspace=0.1)
    # plt.savefig(outdir + "/axial_generate.png")
    plt.show()

    # fig, axes = plt.subplots(ncols=9, nrows=1, figsize=(6, 2))
    # for i in range(side):
    #     axes[i].imshow(img[:, i].reshape(side, side))
    #     axes[i].set_title('y = %d' % i)
    #     axes[i].get_xaxis().set_visible(False)
    #     axes[i].get_yaxis().set_visible(False)
    #
    # fig.colorbar(img)
    # # plt.savefig(outdir + "/coronal_generate.png")
    # plt.show()
    #
    # fig,axes = plt.subplots(ncols=9, nrows=1, figsize=(6, 2))
    # for i in range(side):
    #     axes[i].imshow(img[:, :, i].reshape(side, side))
    #     axes[i].set_title('x = %d' % i)
    #     axes[i].get_xaxis().set_visible(False)
    #     axes[i].get_yaxis().set_visible(False)
    #
    # fig.colorbar(img)
    # # plt.savefig(outdir + "/sagital_generate.png")
    # plt.show()


def display_image2(img1, img2, side, outdir):
    fig, axes = plt.subplots(ncols=9, nrows=2, figsize=(6, 2))
    for i in range(side):
        axes[0, i].imshow(img1[i, :, :].reshape(side, side), cmap=cm.Greys_r)
        axes[0, i].set_title('z = %d' % i)
        axes[0, i].get_xaxis().set_visible(False)
        axes[0, i].get_yaxis().set_visible(False)

        axes[1, i].imshow(img2[i, :, :].reshape(side, side), cmap=cm.Greys_r)
        # axes[1, i].set_title('vae z=%d' % i)
        axes[1, i].get_xaxis().set_visible(False)
        axes[1, i].get_yaxis().set_visible(False)

    plt.savefig(outdir + "/axial_generate.png")
    plt.show()

    fig, axes = plt.subplots(ncols=9, nrows=2, figsize=(6, 2))
    for i in range(side):
        axes[0, i].imshow(img1[:, i, :].reshape(side, side), cmap=cm.Greys_r)
        axes[0, i].set_title('y = %d' % i)
        axes[0, i].get_xaxis().set_visible(False)
        axes[0, i].get_yaxis().set_visible(False)

        axes[1, i].imshow(img2[:, i, :].reshape(side, side), cmap=cm.Greys_r)
        # axes[1, i].set_title('vae z=%d' % i)
        axes[1, i].get_xaxis().set_visible(False)
        axes[1, i].get_yaxis().set_visible(False)

    plt.savefig(outdir + "/coronal_generate.png")
    plt.show()

    fig,axes = plt.subplots(ncols=9, nrows=2, figsize=(6, 2))
    for i in range(side):
        axes[0, i].imshow(img1[:, :, i].reshape(side, side), cmap=cm.Greys_r)
        axes[0, i].set_title('x = %d' % i)
        axes[0, i].get_xaxis().set_visible(False)
        axes[0, i].get_yaxis().set_visible(False)

        axes[1, i].imshow(img2[:, :, i].reshape(side, side), cmap=cm.Greys_r)
        # axes[1, i].set_title('vae z=%d' % i)
        axes[1, i].get_xaxis().set_visible(False)
        axes[1, i].get_yaxis().set_visible(False)

    plt.savefig(outdir + "/sagital_generate.png")
    plt.show()

def display_image3(img1, img2, img3, side, outdir='./output'):
    fig, axes = plt.subplots(ncols=9, nrows=3, figsize=(10, 3))
    min1 = np.min(img1)
    max1 = np.max(img1)
    min2 = np.min(img2)
    max2 = np.max(img2)
    min3 = np.min(img3)
    max3 = np.max(img3)
    for i in range(side):
        axes[0, i].imshow(img1[i, :, :].reshape(side, side), cmap=cm.Greys_r, vmin=min1, vmax=max1, interpolation='none')
        axes[0, i].set_title('z = %d' % i)
        axes[0, i].get_xaxis().set_visible(False)
        axes[0, i].get_yaxis().set_visible(False)

        axes[1, i].imshow(img2[i, :, :].reshape(side, side), cmap=cm.Greys_r, vmin=min2, vmax=max2, interpolation='none')
        # axes[1, i].set_title('vae z=%d' % i)
        axes[1, i].get_xaxis().set_visible(False)
        axes[1, i].get_yaxis().set_visible(False)

        axes[2, i].imshow(img3[i, :, :].reshape(side, side), cmap=cm.Greys_r, vmin=min3, vmax=max3, interpolation='none')
        # axes[2, i].set_title('pca z=%d' % i)
        axes[2, i].get_xaxis().set_visible(False)
        axes[2, i].get_yaxis().set_visible(False)
    plt.savefig(outdir + "/axial_reconstruction.png")

    for i in range(side):
        axes[0, i].imshow(img1[:, i, :].reshape(side, side), cmap=cm.Greys_r, vmin=min1, vmax=max1, interpolation='none')
        axes[0, i].set_title('y = %d' % i)
        axes[0, i].get_xaxis().set_visible(False)
        axes[0, i].get_yaxis().set_visible(False)

        axes[1, i].imshow(img2[:, i, :].reshape(side, side), cmap=cm.Greys_r, vmin=min2, vmax=max2, interpolation='none')
        # axes[1, i].set_title('vae z=%d' % i)
        axes[1, i].get_xaxis().set_visible(False)
        axes[1, i].get_yaxis().set_visible(False)

        axes[2, i].imshow(img3[:, i, :].reshape(side, side), cmap=cm.Greys_r, vmin=min3, vmax=max3, interpolation='none')
        # axes[2, i].set_title('pca z=%d' % i)
        axes[2, i].get_xaxis().set_visible(False)
        axes[2, i].get_yaxis().set_visible(False)
    plt.savefig(outdir + "/coronal_reconstruction.png")

    for i in range(side):
        axes[0, i].imshow(img1[:, :, i].reshape(side, side), cmap=cm.Greys_r, vmin=min1, vmax=max1, interpolation='none')
        axes[0, i].set_title('x = %d' % i)
        axes[0, i].get_xaxis().set_visible(False)
        axes[0, i].get_yaxis().set_visible(False)

        axes[1, i].imshow(img2[:, :, i].reshape(side, side), cmap=cm.Greys_r, vmin=min2, vmax=max2, interpolation='none')
        # axes[1, i].set_title('vae z=%d' % i)
        axes[1, i].get_xaxis().set_visible(False)
        axes[1, i].get_yaxis().set_visible(False)

        axes[2, i].imshow(img3[:, :, i].reshape(side, side), cmap=cm.Greys_r, vmin=min3, vmax=max3, interpolation='none')
        # axes[2, i].set_title('pca z=%d' % i)
        axes[2, i].get_xaxis().set_visible(False)
        axes[2, i].get_yaxis().set_visible(False)
    # plt.savefig(outdir + "/sagital_reconstruction.png")
    plt.show()

def display_slices(case, size, num_data):
    # case: image data, num_data: number of data, size: length of a side
    min = np.min(case)
    max = np.max(case)
    # sagital
    fig, axes = plt.subplots(ncols=size, nrows=num_data, figsize=(size - 2, num_data), dpi=150)
    for i in range(size):
        for j in range(num_data):
            axes[j, i].imshow(case[j, :, :, i].reshape(size, size), cmap=cm.Greys_r, vmin=min, vmax=max, interpolation='none')
            axes[j, i].set_title('x = %d' % i)
            axes[j, i].get_xaxis().set_visible(False)
            axes[j, i].get_yaxis().set_visible(False)
    plt.show()
    # coronal
    fig, axes = plt.subplots(ncols=size, nrows=num_data, figsize=(size - 2, num_data), dpi=150)
    for i in range(size):
        for j in range(num_data):
            axes[j, i].imshow(case[j, :, i, :].reshape(size, size), cmap=cm.Greys_r, vmin=min, vmax=max, interpolation='none')
            axes[j, i].set_title('y = %d' % i)
            axes[j, i].get_xaxis().set_visible(False)
            axes[j, i].get_yaxis().set_visible(False)
    plt.show()
    # axial
    fig, axes = plt.subplots(ncols=size, nrows=num_data, figsize=(size - 2, num_data), dpi=150)
    for i in range(size):
        for j in range(num_data):
            axes[j, i].imshow(case[j, i, :, :].reshape(size, size), cmap=cm.Greys_r, vmin=min, vmax=max, interpolation='none')
            axes[j, i].set_title('z = %d' % i)
            axes[j, i].get_xaxis().set_visible(False)
            axes[j, i].get_yaxis().set_visible(False)
    plt.show()

def visualize_slices(X, Xe, outdir):
    # plot reconstruction
    fig, axes = plt.subplots(ncols=10, nrows=2, figsize=(18, 4))
    minX = np.min(X)
    maxX = np.max(X)
    minXe = np.min(Xe)
    maxXe = np.max(Xe)
    for i in range(10):
        axes[0, i].imshow(X[i, :].reshape(9, 9), cmap=cm.Greys_r, vmin=minX, vmax=maxX,
                          interpolation='none')
        axes[0, i].set_title('original %d' % i)
        axes[0, i].get_xaxis().set_visible(False)
        axes[0, i].get_yaxis().set_visible(False)

        axes[1, i].imshow(Xe[i, :].reshape(9, 9), cmap=cm.Greys_r, vmin=minXe, vmax=maxXe,
                          interpolation='none')
        axes[1, i].set_title('reconstruction %d' % i)
        axes[1, i].get_xaxis().set_visible(False)
        axes[1, i].get_yaxis().set_visible(False)
    plt.savefig(outdir + "reconstruction.png")
    plt.show()

def display_center_slices(case, size, num_data):
    # case: image data, num_data: number of data, size: length of a side
    min = np.min(case)
    max = np.max(case)
    patch_center = size//2
    x = num_data
    # axial
    fig, axes = plt.subplots(ncols=x, nrows=x, figsize=(x, x))
    for j in range(x):
        for i in range(x):
            axes[j, i].imshow(case[i + x*j, :, :, patch_center].reshape(size, size), cmap=cm.Greys_r, vmin=min, vmax=max, interpolation='none')
            axes[j, i].get_xaxis().set_visible(False)
            axes[j, i].get_yaxis().set_visible(False)
    # plt.savefig(outdir + "/.png")
    plt.show()

# calculate L1
def L1norm(im1, im2):
    im1 = np.asarray(im1)
    im2 = np.asarray(im2)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    return np.double(np.mean(abs(im1 - im2)))

def get_dataset(input, patch_side, num_of_test):
    print('load data')
    list = io.load_list(input)
    data_set = np.zeros((num_of_test, patch_side, patch_side, patch_side))
    for i in trange(num_of_test):
        data_set[i, :] = np.reshape(io.read_mhd_and_raw(list[i]), [patch_side, patch_side, patch_side])
    return data_set