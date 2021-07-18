from skimage.io import imread
from skimage import data
import numpy as np
import matplotlib.pyplot as plt
import cv2


def main():
    pass


def fast_glcm(img, vmin=0, vmax=255, nbit=8, kernel_size=5):
    mi, ma = vmin, vmax
    ks = kernel_size
    h, w = img.shape[0], img.shape[1]

    # digitize
    bins = np.linspace(mi, ma + 1, nbit + 1)
    gl1 = np.digitize(img, bins) - 1
    gl2 = np.append(gl1[:, 1:], gl1[:, -1:], axis=1)

    # make glcm
    glcm = np.zeros((nbit, nbit, h, w), dtype=np.uint8)
    for i in range(nbit):
        for j in range(nbit):
            mask = ((gl1 == i) & (gl2 == j))
            glcm[i, j, mask] = 1

    kernel = np.ones((ks, ks), dtype=np.uint8)
    for i in range(nbit):
        for j in range(nbit):
            glcm[i, j] = cv2.filter2D(glcm[i, j], -1, kernel)

    glcm = glcm.astype(np.float32)
    return glcm


def fast_glcm_mean(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm mean
    '''
    h, w = img.shape[0], img.shape[1]
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    mean = np.zeros((h, w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i, j] * i / (nbit) ** 2

    return mean


def fast_glcm_std(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm std
    '''
    h, w = img.shape[0], img.shape[1]
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    mean = np.zeros((h, w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i, j] * i / (nbit) ** 2

    std2 = np.zeros((h, w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            std2 += (glcm[i, j] * i - mean) ** 2

    std = np.sqrt(std2)
    return std


def fast_glcm_contrast(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm contrast
    '''
    h, w = img.shape[0], img.shape[1]
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    cont = np.zeros((h, w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            cont += glcm[i, j] * (i - j) ** 2

    return cont


def fast_glcm_dissimilarity(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm dissimilarity
    '''
    h, w = img.shape[0], img.shape[1]
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    diss = np.zeros((h, w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            diss += glcm[i, j] * np.abs(i - j)

    return diss


def fast_glcm_homogeneity(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm homogeneity
    '''
    h, w = img.shape[0], img.shape[1]
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    homo = np.zeros((h, w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            homo += glcm[i, j] / (1. + (i - j) ** 2)

    return homo


def fast_glcm_ASM(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm asm, energy
    '''
    h, w = img.shape[0], img.shape[1]
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    asm = np.zeros((h, w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            asm += glcm[i, j] ** 2

    ene = np.sqrt(asm)
    return asm, ene


def fast_glcm_max(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm max
    '''
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    max_ = np.max(glcm, axis=(0, 1))
    return max_


def fast_glcm_entropy(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm entropy
    '''
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    pnorm = glcm / np.sum(glcm, axis=(0, 1)) + 1. / ks ** 2
    ent = np.sum(-pnorm * np.log(pnorm), axis=(0, 1))
    return ent


def plot_glcm(img, blk):
    h, w = img.shape[0], img.shape[1]

    ks = 3
    # The "ks" (Kernel Size) is a smoothing parameter.
    mean = fast_glcm_mean(img, ks=ks)
    std = fast_glcm_std(img, ks=ks)
    cont = fast_glcm_contrast(img, ks=ks)
    diss = fast_glcm_dissimilarity(img, ks=ks)
    homo = fast_glcm_homogeneity(img, ks=ks)
    asm, ene = fast_glcm_ASM(img, ks=ks)
    ma = fast_glcm_max(img, ks=ks)
    ent = fast_glcm_entropy(img, ks=ks)

    fig, axes = plt.subplots(2, 5, sharex='all', sharey='all', figsize=(10, 4.5))
    fs = 15

    axes[0, 0].tick_params(labelbottom=False, labelleft=False)
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('original', fontsize=fs)

    axes[0, 1].tick_params(labelbottom=False, labelleft=False)
    axes[0, 1].imshow(mean)
    axes[0, 1].set_title('mean', fontsize=fs)

    axes[0, 2].tick_params(labelbottom=False, labelleft=False)
    axes[0, 2].imshow(std)
    axes[0, 2].set_title('std', fontsize=fs)

    axes[0, 3].tick_params(labelbottom=False, labelleft=False)
    axes[0, 3].imshow(cont)
    axes[0, 3].set_title('contrast', fontsize=fs)

    axes[0, 4].tick_params(labelbottom=False, labelleft=False)
    axes[0, 4].imshow(diss)
    axes[0, 4].set_title('dissimilarity', fontsize=fs)

    axes[1, 0].tick_params(labelbottom=False, labelleft=False)
    axes[1, 0].imshow(homo)
    axes[1, 0].set_title('homogeneity', fontsize=fs)

    axes[1, 1].tick_params(labelbottom=False, labelleft=False)
    axes[1, 1].imshow(asm)
    axes[1, 1].set_title('ASM', fontsize=fs)

    axes[1, 2].tick_params(labelbottom=False, labelleft=False)
    axes[1, 2].imshow(ene)
    axes[1, 2].set_title('energy', fontsize=fs)

    axes[1, 3].tick_params(labelbottom=False, labelleft=False)
    axes[1, 3].imshow(ma)
    axes[1, 3].set_title('max', fontsize=fs)

    axes[1, 4].tick_params(labelbottom=False, labelleft=False)
    axes[1, 4].imshow(ent)
    axes[1, 4].set_title('entropy', fontsize=fs)

    fig.tight_layout(pad=0.5)
    plt.show(block=blk)


root = 'D:/00.University/data/data sets/BD/train/'
imageDir = root + 'subset_images/'
img1 = imread(imageDir + 'hurricane-florence_00000094_pre_disaster.png')
img1 = img1[:, :, 0]
img2 = imread(imageDir + 'hurricane-florence_00000094_post_disaster.png')
img2 = img2[:, :, 0]


# plot_glcm(np.abs(img2.astype(float) - img1.astype(float)), blk=True)
plot_glcm(img1, blk=False)
plot_glcm(img2, blk=True)

