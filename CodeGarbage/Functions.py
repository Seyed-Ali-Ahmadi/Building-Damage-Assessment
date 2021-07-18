import re
import os
import cv2
import json
import numpy as np
from skimage import feature
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.draw import polygon
from keras.layers import MaxPool2D
from keras.models import Sequential


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


def read_jsondir(dir):
    jsonFiles = os.listdir(dir)
    for ind, item in enumerate(jsonFiles):
        jsonFiles[ind] = re.split('_post_disaster|_pre_disaster', item)[0]
    return set(jsonFiles)


def read_image_json(imageDir, jsonDir, uniqueName):
    pre = uniqueName + '_pre_disaster'
    post = uniqueName + '_post_disaster'

    # Read the json file.
    preJson = json.load(open(jsonDir + pre + '.json', ))
    postJson = json.load(open(jsonDir + post + '.json', ))

    # Read the corresponding image of the same JSON file.
    # preImage = (rgb2gray(imread(imageDir + pre + '.png')) * 255).astype(int)
    # postImage = (rgb2gray(imread(imageDir + post + '.png')) * 255).astype(int)

    # This is for OpenCV operations
    preImage = cv2.cvtColor(cv2.imread(imageDir + pre + '.png'), cv2.COLOR_BGR2GRAY)
    postImage = cv2.cvtColor(cv2.imread(imageDir + post + '.png'), cv2.COLOR_BGR2GRAY)
    return preJson, preImage, postJson, postImage


def get_lbp(image, radius=2, method='default'):
    # LBP
    r = radius
    n_points = 8 * radius
    m = method  # ‘default’, ‘ror’, ‘uniform’, ‘var’
    return feature.local_binary_pattern(image, n_points, r, method=m)


def get_glcms(image, ks=3):
    # The "ks" (Kernel Size) is a smoothing parameter.
    mean = fast_glcm_mean(image, ks=ks)
    std = fast_glcm_std(image, ks=ks)
    cont = fast_glcm_contrast(image, ks=ks)
    diss = fast_glcm_dissimilarity(image, ks=ks)
    # homo = fast_glcm_homogeneity(image, ks=ks)
    # asm, ene = fast_glcm_ASM(image, ks=ks)
    # ma = fast_glcm_max(image, ks=ks)
    ent = fast_glcm_entropy(image, ks=ks)
    # [mean, std, cont, diss, homo, asm, ene, ma, ent]
    return np.dstack((mean, std, cont, diss, ent))


def make_pad(pre, post, pad=10, case=1):
    pre = cv2.copyMakeBorder(pre, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    post = cv2.copyMakeBorder(post, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    return pre, post


def get_patch(image, rows, cols, pad=10):
    image_crop = image[int(np.floor(min(rows))) - pad:int(np.ceil(max(rows))) + pad,
                       int(np.floor(min(cols))) - pad:int(np.ceil(max(cols))) + pad]
    return image_crop


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def standardize(image, show=False):
    image = image.astype(float)

    # print('Number of image dimensions:   ', np.ndim(image))

    if np.ndim(image) == 2:
        # print('Image range:   ', [np.amin(image), np.amax(image)])
        standard = (image - np.amin(image)) / (np.amax(image) - np.amin(image))
        # print('Image range (standard):   ', [np.amin(standard), np.amax(standard)])

        if show:
            plt.figure()
            plt.subplot(121), plt.imshow(image, cmap='gray')
            plt.title('[' + str(np.amin(image)) + ',' + str(np.amax(image)) + ']')
            plt.subplot(122), plt.imshow(standard, cmap='gray')
            plt.title('[' + str(np.amin(standard)) + ',' + str(np.amax(standard)) + ']')
            plt.show()

    elif np.ndim(image) == 3:
        standard = np.empty_like(image)
        for i in range(image.shape[2]):
            # print('Image range:   ', [np.amin(image[:, :, i]), np.amax(image[:, :, i])])
            standard[:, :, i] = (image[:, :, i] - np.amin(image[:, :, i])) / \
                                (np.amax(image[:, :, i]) - np.amin(image[:, :, i]))
            # print('Image range (standard):   ', [np.amin(standard[:, :, i]), np.amax(standard[:, :, i])])

            if show:
                plt.figure()
                plt.subplot(121), plt.imshow(image[:, :, i], cmap='gray')
                plt.title('[' + str(np.amin(image[:, :, i])) + ',' + str(np.amax(image[:, :, i])) + ']')
                plt.subplot(122), plt.imshow(standard[:, :, i], cmap='gray')
                plt.title('[' + str(np.amin(standard[:, :, i])) + ',' + str(np.amax(standard[:, :, i])) + ']')
                plt.show()
    else:
        print('Invalid array shape.')
        standard = None

    return standard


def pad_features(features, pad):
    padded = np.empty((features.shape[0] + pad * 2,
                       features.shape[1] + pad * 2,
                       features.shape[2]))
    for i in range(features.shape[2]):
        padded[:, :, i] = cv2.copyMakeBorder(features[:, :, i], pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    return padded


def create_mask(labels, wclassification=True):
    mask = np.zeros((1024, 1024))
    classes = ['destroyed', 'minor-damage', 'major-damage', 'no-damage', 'un-classified']
    numOfBuildings = 0
    for building in labels:
        vertices = building['wkt'].partition('POLYGON ((')[2].partition('))')[0].split(', ')
        rows = []
        cols = []
        for vertex in vertices:
            cols.append(float(vertex.split(' ')[0]))
            rows.append(float(vertex.split(' ')[1]))
        rr, cc = polygon(rows, cols, (1024, 1024))
        pixels = list(zip(rr, cc))
        Class = classes.index(building['properties']['subtype']) + 1
        for rr, cc in pixels:
            if wclassification:
                mask[rr, cc] = Class
            else:
                mask[rr, cc] = numOfBuildings
        numOfBuildings += 1
    return mask


def max_kernel(patch):
    arr = patch.copy()
    arr = arr.reshape(1, 24, 24, 1)
    out = np.squeeze(Sequential([MaxPool2D(pool_size=3, strides=3)]).predict(arr))
    arr = out.reshape(1, 8, 8, 1)
    out = np.squeeze(Sequential([MaxPool2D(pool_size=2, strides=2)]).predict(arr))
    return out.flatten()

