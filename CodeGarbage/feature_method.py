from skimage.color import rgb2gray
from skimage.io import imread
import matplotlib.pyplot as plt
from Functions import unsharp_mask, get_lbp, get_glcms
from datetime import datetime as dt
from skimage.util import img_as_float, img_as_ubyte
from sklearn.preprocessing import minmax_scale
import numpy as np
from DCVA import dcva
from skimage import exposure
import cv2
import pickle
import MorphologyProfiles as mp
from sklearn.decomposition import PCA, KernelPCA


def gen_features(preChangeImage, postChangeImage, numfeatures=28):
    if numfeatures == 28:
        start_total = dt.now()
        # --------------------------------------------------------------------------------------------------------
        preSharp = unsharp_mask(cv2.medianBlur(preChangeImage, 3), kernel_size=(3, 3), sigma=1.5)
        postSharp = unsharp_mask(cv2.medianBlur(postChangeImage, 3), kernel_size=(3, 3), sigma=1.5)
        # --------------------------------------------------------------------------------------------------------
        preLBP = get_lbp(rgb2gray(preSharp))
        postLBP = get_lbp(rgb2gray(postSharp))
        # --------------------------------------------------------------------------------------------------------
        preGLCM = get_glcms(img_as_ubyte(rgb2gray(preSharp)))
        postGLCM = get_glcms(img_as_ubyte(rgb2gray(postSharp)))
        # --------------------------------------------------------------------------------------------------------
        deepCVA = dcva(preChangeImage, postChangeImage, layers=[2], feature=True)
        # --------------------------------------------------------------------------------------------------------
        difference = img_as_float(postChangeImage) - img_as_float(preChangeImage)
        # --------------------------------------------------------------------------------------------------------
        preChangeImage = minmax_scale(preChangeImage.reshape((1024 * 1024, 3))).reshape((1024, 1024, 3))
        postChangeImage = minmax_scale(postChangeImage.reshape((1024 * 1024, 3))).reshape((1024, 1024, 3))
        preLBP = minmax_scale(preLBP.reshape((1024 * 1024, 1))).reshape((1024, 1024))
        postLBP = minmax_scale(postLBP.reshape((1024 * 1024, 1))).reshape((1024, 1024))
        preGLCM = minmax_scale(preGLCM.reshape((1024 * 1024, 5))).reshape((1024, 1024, 5))
        postGLCM = minmax_scale(postGLCM.reshape((1024 * 1024, 5))).reshape((1024, 1024, 5))
        deepCVA = minmax_scale(deepCVA.reshape((1024 * 1024, 1))).reshape((1024, 1024))
        difference = minmax_scale(difference.reshape((1024 * 1024, 3))).reshape((1024, 1024, 3))
        # --------------------------------------------------------------------------------------------------------
        p2, p98 = np.percentile(preGLCM[:, :, 2], (2, 98))
        preGLCM[:, :, 2] = exposure.rescale_intensity(preGLCM[:, :, 2], in_range=(p2, p98))
        p2, p98 = np.percentile(postGLCM[:, :, 2], (2, 98))
        preGLCM[:, :, 2] = exposure.rescale_intensity(postGLCM[:, :, 2], in_range=(p2, p98))
        # --------------------------------------------------------------------------------------------------------
        pca_pre = PCA(n_components=1).fit_transform(
            np.dstack((preChangeImage, preGLCM)).reshape((1024 * 1024, 8))).reshape((1024, 1024))
        pca_pre = minmax_scale(pca_pre.reshape((1024 * 1024, 1))).reshape((1024, 1024))

        pca_post = PCA(n_components=1).fit_transform(
            np.dstack((postChangeImage, postGLCM)).reshape((1024 * 1024, 8))).reshape((1024, 1024))
        pca_post = minmax_scale(pca_post.reshape((1024 * 1024, 1))).reshape((1024, 1024))

        pca_diff = PCA(n_components=1).fit_transform(np.dstack((deepCVA, difference)).reshape((1024 * 1024, 4))).reshape(
            (1024, 1024))
        pca_diff = minmax_scale(pca_diff.reshape((1024 * 1024, 1))).reshape((1024, 1024))
        # --------------------------------------------------------------------------------------------------------
        emp = mp.build_emp(base_image=np.dstack((pca_pre, pca_post, pca_diff)), num_openings_closings=(2 * 2) + 1)
        pca_emp = PCA(n_components=3).fit_transform(emp.reshape((1024 * 1024, 33))).reshape((1024, 1024, 3))
        pca_emp = minmax_scale(pca_emp.reshape((1024 * 1024, 3))).reshape((1024, 1024, 3))
        # --------------------------------------------------------------------------------------------------------
        featureArray = np.dstack((preChangeImage, postChangeImage, preLBP, postLBP, preGLCM, postGLCM,
                                  pca_pre, pca_post, pca_diff, pca_emp, deepCVA, difference))
        # --------------------------------------------------------------------------------------------------------
        end_total = dt.now()
        print(end_total - start_total)

    elif numfeatures == 18:
        start_total = dt.now()
        # --------------------------------------------------------------------------------------------------------
        preSharp = unsharp_mask(cv2.medianBlur(preChangeImage, 3), kernel_size=(3, 3), sigma=1.5)
        postSharp = unsharp_mask(cv2.medianBlur(postChangeImage, 3), kernel_size=(3, 3), sigma=1.5)
        # --------------------------------------------------------------------------------------------------------
        preLBP = get_lbp(rgb2gray(preSharp))
        postLBP = get_lbp(rgb2gray(postSharp))
        # --------------------------------------------------------------------------------------------------------
        preGLCM = get_glcms(img_as_ubyte(rgb2gray(preSharp)))
        postGLCM = get_glcms(img_as_ubyte(rgb2gray(postSharp)))
        # --------------------------------------------------------------------------------------------------------
        preSharp = minmax_scale(preSharp.reshape((1024 * 1024, 3))).reshape((1024, 1024, 3))
        postSharp = minmax_scale(postSharp.reshape((1024 * 1024, 3))).reshape((1024, 1024, 3))
        preLBP = minmax_scale(preLBP.reshape((1024 * 1024, 1))).reshape((1024, 1024))
        postLBP = minmax_scale(postLBP.reshape((1024 * 1024, 1))).reshape((1024, 1024))
        preGLCM = minmax_scale(preGLCM.reshape((1024 * 1024, 5))).reshape((1024, 1024, 5))
        postGLCM = minmax_scale(postGLCM.reshape((1024 * 1024, 5))).reshape((1024, 1024, 5))
        # --------------------------------------------------------------------------------------------------------
        featureArray = np.dstack((preSharp, postSharp, preLBP, postLBP, preGLCM, postGLCM))
        # --------------------------------------------------------------------------------------------------------
        end_total = dt.now()
        print(end_total - start_total)

    elif numfeatures == 22:
        start_total = dt.now()
        # --------------------------------------------------------------------------------------------------------
        preSharp = unsharp_mask(cv2.medianBlur(preChangeImage, 3), kernel_size=(3, 3), sigma=1.5)
        postSharp = unsharp_mask(cv2.medianBlur(postChangeImage, 3), kernel_size=(3, 3), sigma=1.5)
        # --------------------------------------------------------------------------------------------------------
        preLBP = get_lbp(rgb2gray(preSharp))
        postLBP = get_lbp(rgb2gray(postSharp))
        # --------------------------------------------------------------------------------------------------------
        preGLCM = get_glcms(img_as_ubyte(rgb2gray(preSharp)))
        postGLCM = get_glcms(img_as_ubyte(rgb2gray(postSharp)))
        # --------------------------------------------------------------------------------------------------------
        deepCVA = dcva(preChangeImage, postChangeImage, layers=[2], feature=True)
        # --------------------------------------------------------------------------------------------------------
        difference = img_as_float(postChangeImage) - img_as_float(preChangeImage)
        # --------------------------------------------------------------------------------------------------------
        preSharp = minmax_scale(preSharp.reshape((1024 * 1024, 3))).reshape((1024, 1024, 3))
        postSharp = minmax_scale(postSharp.reshape((1024 * 1024, 3))).reshape((1024, 1024, 3))
        preLBP = minmax_scale(preLBP.reshape((1024 * 1024, 1))).reshape((1024, 1024))
        postLBP = minmax_scale(postLBP.reshape((1024 * 1024, 1))).reshape((1024, 1024))
        preGLCM = minmax_scale(preGLCM.reshape((1024 * 1024, 5))).reshape((1024, 1024, 5))
        postGLCM = minmax_scale(postGLCM.reshape((1024 * 1024, 5))).reshape((1024, 1024, 5))
        deepCVA = minmax_scale(deepCVA.reshape((1024 * 1024, 1))).reshape((1024, 1024))
        difference = minmax_scale(difference.reshape((1024 * 1024, 3))).reshape((1024, 1024, 3))
        # --------------------------------------------------------------------------------------------------------
        featureArray = np.dstack((preSharp, postSharp, preLBP, postLBP, preGLCM, postGLCM, deepCVA, difference))
        # --------------------------------------------------------------------------------------------------------
        end_total = dt.now()
        print(end_total - start_total)

    elif numfeatures == 6:
        start_total = dt.now()
        # --------------------------------------------------------------------------------------------------------
        preChangeImage = minmax_scale(preChangeImage.reshape((1024 * 1024, 3))).reshape((1024, 1024, 3))
        postChangeImage = minmax_scale(postChangeImage.reshape((1024 * 1024, 3))).reshape((1024, 1024, 3))
        # --------------------------------------------------------------------------------------------------------
        featureArray = np.dstack((preChangeImage, postChangeImage))
        # --------------------------------------------------------------------------------------------------------
        end_total = dt.now()
        print(end_total - start_total)

    return featureArray


