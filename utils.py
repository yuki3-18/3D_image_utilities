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
import gudhi as gd
from topologylayer.util.construction import unique_simplices
from topologylayer.nn.features import get_start_end
from topologylayer.nn.levelset import LevelSetLayer
from scipy.spatial import Delaunay
import torch


def masking(img, x_size, y_size, z_size, mask, mask_value1, mask_value2):
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

    return masking_img


def cropping(img, xs, xe, ys, ye, zs, ze):
    cropped_img = img[zs:ze + 1, ys:ye + 1, xs:xe + 1]

    return cropped_img


def cropping9(img, x_size, y_size, z_size, x, y, z):
    img = np.reshape(img, [z_size, y_size, x_size])
    cropped_img = img[z - 4:z + 5, y - 4:y + 5, x - 4:x + 5]
    cropped_img = cropped_img.reshape([9, 9, 9])

    return cropped_img


def cropping320(img, x_size, y_size, z_size, x, y, z):
    img = np.reshape(img, [z_size, y_size, x_size])
    cropped_img = img[z - 160:z + 160, y - 160:y + 160, x - 160:x + 160]
    print(cropped_img.shape)
    cropped_img = cropped_img.reshape([320, 320, 320])
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
    w = int(patch_side / 2)

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


def display_image(img, side, min=0, max=1):
    img = np.reshape(img, [side, side, side])
    fig, axes = plt.subplots(ncols=side, nrows=1, figsize=(side - 2, 2), constrained_layout=True)

    for i in range(side):
        axes[i].imshow(img[i].reshape(side, side), vmin=min, vmax=max, interpolation='none', cmap=cm.Greys_r)
        axes[i].set_title('z = %d' % i)
        axes[i].get_xaxis().set_visible(False)
        # axes[i].get_yaxis().set_visible(False)
        axes[0].set_ylabel('k', rotation=0)
        axes[i].axes.yaxis.set_ticks([])

    plt.show()

    fig, axes = plt.subplots(ncols=side, nrows=1, figsize=(side - 2, 2), constrained_layout=True)
    for i in range(side):
        axes[i].imshow(img[:, i].reshape(side, side), vmin=min, vmax=max, interpolation='none', cmap=cm.Greys_r)
        axes[i].set_title('y = %d' % i)
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)

    plt.show()
    #
    fig, axes = plt.subplots(ncols=side, nrows=1, figsize=(side - 2, 2), constrained_layout=True)
    for i in range(side):
        axes[i].imshow(img[:, :, i].reshape(side, side), vmin=min, vmax=max, interpolation='none', cmap=cm.Greys_r)
        axes[i].set_title('x = %d' % i)
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)

    plt.show()
    plt.close()


def display_image2(img1, img2, side, outdir):
    fig, axes = plt.subplots(ncols=9, nrows=2, figsize=(6, 2))
    min = 0
    max = 1
    for i in range(side):
        axes[0, i].imshow(img1[i, :, :].reshape(side, side), vmin=min, vmax=max, interpolation='none', cmap=cm.Greys_r)
        axes[0, i].set_title('z = %d' % i)
        axes[0, i].get_xaxis().set_visible(False)
        axes[0, i].get_yaxis().set_visible(False)

        axes[1, i].imshow(img2[i, :, :].reshape(side, side), vmin=min, vmax=max, interpolation='none', cmap=cm.Greys_r)
        # axes[1, i].set_title('vae z=%d' % i)
        axes[1, i].get_xaxis().set_visible(False)
        axes[1, i].get_yaxis().set_visible(False)

    plt.savefig(outdir + "/axial_generate.png")
    plt.show()

    fig, axes = plt.subplots(ncols=9, nrows=2, figsize=(6, 2))
    for i in range(side):
        axes[0, i].imshow(img1[:, i, :].reshape(side, side), vmin=min, vmax=max, interpolation='none', cmap=cm.Greys_r)
        axes[0, i].set_title('y = %d' % i)
        axes[0, i].get_xaxis().set_visible(False)
        axes[0, i].get_yaxis().set_visible(False)

        axes[1, i].imshow(img2[:, i, :].reshape(side, side), vmin=min, vmax=max, interpolation='none', cmap=cm.Greys_r)
        # axes[1, i].set_title('vae z=%d' % i)
        axes[1, i].get_xaxis().set_visible(False)
        axes[1, i].get_yaxis().set_visible(False)

    plt.savefig(outdir + "/coronal_generate.png")
    plt.show()

    fig, axes = plt.subplots(ncols=9, nrows=2, figsize=(6, 2))
    for i in range(side):
        axes[0, i].imshow(img1[:, :, i].reshape(side, side), vmin=min, vmax=max, interpolation='none', cmap=cm.Greys_r)
        axes[0, i].set_title('x = %d' % i)
        axes[0, i].get_xaxis().set_visible(False)
        axes[0, i].get_yaxis().set_visible(False)

        axes[1, i].imshow(img2[:, :, i].reshape(side, side), vmin=min, vmax=max, interpolation='none', cmap=cm.Greys_r)
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
        axes[0, i].imshow(img1[i, :, :].reshape(side, side), cmap=cm.Greys_r, vmin=min1, vmax=max1,
                          interpolation='none')
        axes[0, i].set_title('z = %d' % i)
        axes[0, i].get_xaxis().set_visible(False)
        axes[0, i].get_yaxis().set_visible(False)

        axes[1, i].imshow(img2[i, :, :].reshape(side, side), cmap=cm.Greys_r, vmin=min2, vmax=max2,
                          interpolation='none')
        # axes[1, i].set_title('vae z=%d' % i)
        axes[1, i].get_xaxis().set_visible(False)
        axes[1, i].get_yaxis().set_visible(False)

        axes[2, i].imshow(img3[i, :, :].reshape(side, side), cmap=cm.Greys_r, vmin=min3, vmax=max3,
                          interpolation='none')
        # axes[2, i].set_title('pca z=%d' % i)
        axes[2, i].get_xaxis().set_visible(False)
        axes[2, i].get_yaxis().set_visible(False)
    plt.savefig(outdir + "/axial_reconstruction.png")

    for i in range(side):
        axes[0, i].imshow(img1[:, i, :].reshape(side, side), cmap=cm.Greys_r, vmin=min1, vmax=max1,
                          interpolation='none')
        axes[0, i].set_title('y = %d' % i)
        axes[0, i].get_xaxis().set_visible(False)
        axes[0, i].get_yaxis().set_visible(False)

        axes[1, i].imshow(img2[:, i, :].reshape(side, side), cmap=cm.Greys_r, vmin=min2, vmax=max2,
                          interpolation='none')
        # axes[1, i].set_title('vae z=%d' % i)
        axes[1, i].get_xaxis().set_visible(False)
        axes[1, i].get_yaxis().set_visible(False)

        axes[2, i].imshow(img3[:, i, :].reshape(side, side), cmap=cm.Greys_r, vmin=min3, vmax=max3,
                          interpolation='none')
        # axes[2, i].set_title('pca z=%d' % i)
        axes[2, i].get_xaxis().set_visible(False)
        axes[2, i].get_yaxis().set_visible(False)
    plt.savefig(outdir + "/coronal_reconstruction.png")

    for i in range(side):
        axes[0, i].imshow(img1[:, :, i].reshape(side, side), cmap=cm.Greys_r, vmin=min1, vmax=max1,
                          interpolation='none')
        axes[0, i].set_title('x = %d' % i)
        axes[0, i].get_xaxis().set_visible(False)
        axes[0, i].get_yaxis().set_visible(False)

        axes[1, i].imshow(img2[:, :, i].reshape(side, side), cmap=cm.Greys_r, vmin=min2, vmax=max2,
                          interpolation='none')
        # axes[1, i].set_title('vae z=%d' % i)
        axes[1, i].get_xaxis().set_visible(False)
        axes[1, i].get_yaxis().set_visible(False)

        axes[2, i].imshow(img3[:, :, i].reshape(side, side), cmap=cm.Greys_r, vmin=min3, vmax=max3,
                          interpolation='none')
        # axes[2, i].set_title('pca z=%d' % i)
        axes[2, i].get_xaxis().set_visible(False)
        axes[2, i].get_yaxis().set_visible(False)
    # plt.savefig(outdir + "/sagital_reconstruction.png")
    plt.show()


def display_slices(case, min=0, max=1, size=9):
    # case: image data (num_data, size, size, size)
    data = np.asarray(case)
    if data.ndim < 4:
        data = data[np.newaxis, :]
        data = data.reshape([1, size, size, size])
    num_data, size, y, x = data.shape
    if num_data == 1:
        case = data.reshape(size, y, x)
        # sagittal
        fig, axes = plt.subplots(ncols=size, nrows=num_data, figsize=(size - 3, num_data), dpi=150)
        for i in range(size):
            axes[i].imshow(case[:, :, i].reshape(size, size), cmap=cm.Greys_r, vmin=min, vmax=max, interpolation='none')
            axes[i].set_title('x = %d' % i)
            axes[i].get_xaxis().set_visible(False)
            axes[i].get_yaxis().set_visible(False)
        plt.show()
        # coronal
        fig, axes = plt.subplots(ncols=size, nrows=num_data, figsize=(size - 3, num_data), dpi=150)
        for i in range(size):
            axes[i].imshow(case[:, i, :].reshape(size, size), cmap=cm.Greys_r, vmin=min, vmax=max, interpolation='none')
            axes[i].set_title('y = %d' % i)
            axes[i].get_xaxis().set_visible(False)
            axes[i].get_yaxis().set_visible(False)
        plt.show()
        # axial
        fig, axes = plt.subplots(ncols=size, nrows=num_data, figsize=(size - 3, num_data), dpi=150)
        for i in range(size):
            axes[i].imshow(case[i, :, :].reshape(size, size), cmap=cm.Greys_r, vmin=min, vmax=max, interpolation='none')
            axes[i].set_title('z = %d' % i)
            axes[i].get_xaxis().set_visible(False)
            axes[i].get_yaxis().set_visible(False)
        plt.show()
    else:
        # sagittal
        fig, axes = plt.subplots(ncols=size, nrows=num_data, figsize=(size - 2, num_data), dpi=150)
        for i in range(size):
            for j in range(num_data):
                axes[j, i].imshow(case[j, :, :, i].reshape(size, size), cmap=cm.Greys_r, vmin=min, vmax=max,
                                  interpolation='none')
                axes[j, i].set_title('x = %d' % i)
                axes[j, i].get_xaxis().set_visible(False)
                axes[j, i].get_yaxis().set_visible(False)
        plt.show()
        # coronal
        fig, axes = plt.subplots(ncols=size, nrows=num_data, figsize=(size - 2, num_data), dpi=150)
        for i in range(size):
            for j in range(num_data):
                axes[j, i].imshow(case[j, :, i, :].reshape(size, size), cmap=cm.Greys_r, vmin=min, vmax=max,
                                  interpolation='none')
                axes[j, i].set_title('y = %d' % i)
                axes[j, i].get_xaxis().set_visible(False)
                axes[j, i].get_yaxis().set_visible(False)
        plt.show()
        # axial
        fig, axes = plt.subplots(ncols=size, nrows=num_data, figsize=(size - 2, num_data), dpi=150)
        for i in range(size):
            for j in range(num_data):
                axes[j, i].imshow(case[j, i, :, :].reshape(size, size), cmap=cm.Greys_r, vmin=min, vmax=max,
                                  interpolation='none')
                axes[j, i].set_title('z = %d' % i)
                axes[j, i].get_xaxis().set_visible(False)
                axes[j, i].get_yaxis().set_visible(False)
        plt.show()
    plt.close()


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


def display_center_slices(case, size, num_data, min=0, max=1):
    # case: image data, num_data: number of data, size: length of a side
    # min = np.min(case)
    # max = np.max(case)
    patch_center = size // 2
    x = num_data
    # axial
    fig, axes = plt.subplots(ncols=x, nrows=x, figsize=(x, x))
    for j in range(x):
        for i in range(x):
            axes[j, i].imshow(case[i + x * j, patch_center, :, :].reshape(size, size), cmap=cm.Greys_r, vmin=min,
                              vmax=max, interpolation='none')
            axes[j, i].get_xaxis().set_visible(False)
            axes[j, i].get_yaxis().set_visible(False)
    # plt.savefig(outdir + "/.png")
    plt.show()


def display_any_slices(case, size, col, row, min=0, max=1, s=4, outdir=None, file=None):
    # case: image data, num_data: number of data, size: length of a side
    # min = np.min(case)
    # max = np.max(case)
    # axial
    fig, axes = plt.subplots(ncols=col, nrows=row, figsize=(col, row))
    for j in range(row):
        for i in range(col):
            print(i + col * j)
            axes[j, i].imshow(case[i + col * j, s, :, :].reshape(size, size), cmap=cm.Greys_r, vmin=min,
                              vmax=max, interpolation='none')
            axes[j, i].get_xaxis().set_visible(False)
            axes[j, i].get_yaxis().set_visible(False)
    if outdir:
        os.makedirs(os.path.dirname(outdir), exist_ok=True)
        plt.savefig(outdir + "/a_{}.png".format(file))
    else:
        plt.show()
    plt.close()

    # coronal
    fig, axes = plt.subplots(ncols=col, nrows=row, figsize=(col, row))
    for j in range(row):
        for i in range(col):
            axes[j, i].imshow(case[i + col * j, :, s, :].reshape(size, size), cmap=cm.Greys_r, vmin=min,
                              vmax=max, interpolation='none')
            axes[j, i].get_xaxis().set_visible(False)
            axes[j, i].get_yaxis().set_visible(False)
    if outdir:
        os.makedirs(os.path.dirname(outdir), exist_ok=True)
        plt.savefig(outdir + "/c_{}.png".format(file))
    else:
        plt.show()
    plt.close()

    # sagittal
    fig, axes = plt.subplots(ncols=col, nrows=row, figsize=(col, row))
    for j in range(row):
        for i in range(col):
            axes[j, i].imshow(case[i + col * j, :, :, s].reshape(size, size), cmap=cm.Greys_r, vmin=min,
                              vmax=max, interpolation='none')
            axes[j, i].get_xaxis().set_visible(False)
            axes[j, i].get_yaxis().set_visible(False)
    if outdir:
        os.makedirs(os.path.dirname(outdir), exist_ok=True)
        plt.savefig(outdir + "/s_{}.png".format(file))
    else:
        plt.show()
    plt.close()


# calculate L1
def L1norm(im1, im2):
    im1 = np.asarray(im1)
    im2 = np.asarray(im2)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    return np.double(np.mean(abs(im1 - im2)))


# calculate MSE
def MSE(im1, im2):
    im1 = np.asarray(im1)
    im2 = np.asarray(im2)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    return np.double(np.mean((im1 - im2) ** 2))


def get_dataset(input, patch_side, num_of_test):
    print('load data')
    list = io.load_list(input)
    data_set = np.empty((num_of_test, patch_side, patch_side, patch_side))
    for i in trange(num_of_test):
        data_set[i, :] = np.reshape(io.read_mhd_and_raw(list[i]), [patch_side, patch_side, patch_side])

    return data_set


def get_data_from_list(list, patch_side=9):
    print('load data')
    list = io.load_list(list)
    data_set = np.empty((len(list), patch_side, patch_side, patch_side))
    for i, name in enumerate(list):
        data_set[i, :] = np.reshape(io.read_mhd_and_raw(name), [patch_side, patch_side, patch_side])

    return data_set


def min_max(x, axis=None):
    x_min = x.min(axis=axis, keepdims=True)
    x_max = x.max(axis=axis, keepdims=True)
    return (x - x_min) / (x_max - x_min)


def rank_norm(x, h=13, l=5, axis=1):
    x_sort = np.sort(x, axis=axis)
    x_high = x_sort[:, -h].reshape([-1, 1])
    x_low = x_sort[:, l - 1].reshape([-1, 1])
    norm = (x - x_low) / (x_high - x_low)
    return norm.clip(0., 1.)


def per_norm(x, l=5, h=95):
    x_low = np.percentile(x, l)
    x_high = np.percentile(x, h)
    norm = (x - x_low) / (x_high - x_low)
    return norm.clip(0., 1.)


def PH_diag(img, patch_side):
    cc = gd.CubicalComplex(dimensions=(patch_side, patch_side, patch_side),
                           top_dimensional_cells=1 - img.flatten())
    diag = cc.persistence()
    plt.figure(figsize=(3, 3))
    # diag_clean = diag_tidy(diag, 1e-3)
    gd.plot_persistence_barcode(diag, max_intervals=0, inf_delta=100)
    print(diag)
    plt.xlim(0, 1)
    plt.ylim(-1, len(diag))
    plt.xticks(ticks=np.linspace(0, 1, 6), labels=np.round(np.linspace(1, 0, 6), 2))
    plt.yticks([])
    plt.show()


def getPB(dgminfo, dim):
    dgms, issublevel = dgminfo
    start, end = get_start_end(dgms[dim], issublevel)
    lengths = end - start
    death = start[lengths != 0]
    birth = end[lengths != 0]
    bar = torch.stack([1 - birth, 1 - death], dim=1).tolist()
    if len(bar) != 0:
        diag = []
        for i in range(len(bar)):
            diag.append([dim, bar[i]])
        return diag
    else:
        return None


def drawPB(data):
    z, y, x = data.shape
    cpx = init_tri_complex_3d(z, y, x)
    layer = LevelSetLayer(cpx, maxdim=2, sublevel=False)
    dgminfo = layer(torch.from_numpy(data).float())
    diag = []
    diag2 = getPB(dgminfo, 2)
    diag1 = getPB(dgminfo, 1)
    diag0 = getPB(dgminfo, 0)
    if diag2 != None: diag += diag2
    if diag1 != None: diag += diag1
    if diag0 != None: diag += diag0

    diag = diag_tidy(diag, 1e-3)
    print(diag)

    plt.figure(figsize=(3, 3))
    gd.plot_persistence_barcode(diag, max_intervals=0, inf_delta=100)
    plt.xlim(0, 1)
    plt.ylim(-1, len(diag))
    plt.xticks(ticks=np.linspace(0, 1, 6), labels=np.round(np.linspace(1, 0, 6), 2))
    plt.yticks([])
    plt.show()


def init_tri_complex_3d(width, height, depth):
    """
    initialize 3d complex in dumbest possible way
    """
    # initialize complex to use for persistence calculations
    axis_x = np.arange(0, width)
    axis_y = np.arange(0, height)
    axis_z = np.arange(0, depth)
    grid_axes = np.array(np.meshgrid(axis_x, axis_y, axis_z))
    grid_axes = np.transpose(grid_axes, (1, 2, 3, 0))

    # creation of a complex for calculations
    tri = Delaunay(grid_axes.reshape([-1, 3]))
    return unique_simplices(tri.simplices, 3)


def diag_tidy(diag, eps=1e-1):
    new_diag = []
    for _, x in diag:
        if np.abs(x[0] - x[1]) > eps:
            new_diag.append((_, x))
    return new_diag


def thresh(img, th=0.5):
    return (img > th) * 1.


def compute_Betti_bumbers(img, th=0.5):
    z, y, x = img.shape
    img_v = img.flatten()
    cc = gd.CubicalComplex(dimensions=(z, y, x),
                           top_dimensional_cells=1 - img_v)
    cc.persistence(min_persistence=th)
    return cc.persistent_betti_numbers(0, th)


def add_frame(target):
    v = np.ones_like(target[0]).reshape(1, 9, 9)
    img = np.vstack([v, target, v])
    h = np.ones_like(img[:, 0, :]).reshape(11, 1, 9)
    img = np.hstack([h, img, h])
    d = np.ones_like(img[:, :, 0]).reshape(11, 11, 1)
    img = np.dstack([d, img, d])
    return img
