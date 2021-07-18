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
import pandas as pd


disaster_type = 'wind'
showFeatures = 0
n_features = 6

# Only extract features from No-Damage and Destroyed classes.
# classes = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']
classes = ['no-damage', 'destroyed']

database1 = pickle.load(open("databaseDictionary_wind_1.pkl", "rb"))
database2 = pickle.load(open("databaseDictionary_wind_2.pkl", "rb"))

FeatureVector = np.zeros((0, n_features + 1), dtype=np.float)

start_total = dt.now()

if n_features == 28:
    for c in classes:
        print('Checking files for ' + c.upper() + ' in ' + disaster_type.upper() + ' ...')
        files1 = database1[disaster_type][c]
        files2 = database2[disaster_type][c]
        files = pd.concat((files1, files2))
        print('\t\t There are\t{}\timages for feature extraction.'.format(files.shape[0]))
        file_num = 1
        for idx, row in files.iterrows():
            print('>' * file_num + '{0} / {1} processed.'.format(file_num, files.shape[0]))
            file_num += 1

            file = row['img_name']
            postFile = file.replace('labels', 'images') + '.png'
            preFile = postFile.replace('post', 'pre')

            # --------------------------------------------------------------------------------------------------------
            print('\n\n1- Reading Data >>>' + ' Image ' + file.split('/')[-1])
            startTime = dt.now()
            preChangeImage = imread(preFile)
            postChangeImage = imread(postFile)
            endTime = dt.now()
            print(endTime - startTime)
            # --------------------------------------------------------------------------------------------------------

            # --------------------------------------------------------------------------------------------------------
            print('2- Unsharp-Masking ...')
            startTime = dt.now()
            preSharp = unsharp_mask(cv2.medianBlur(preChangeImage, 3), kernel_size=(3, 3), sigma=1.5)
            postSharp = unsharp_mask(cv2.medianBlur(postChangeImage, 3), kernel_size=(3, 3), sigma=1.5)
            endTime = dt.now()
            print(endTime - startTime)
            # --------------------------------------------------------------------------------------------------------

            # --------------------------------------------------------------------------------------------------------
            print('3- Creating LBP ...')
            startTime = dt.now()
            preLBP = get_lbp(rgb2gray(preSharp))
            postLBP = get_lbp(rgb2gray(postSharp))
            endTime = dt.now()
            print(endTime - startTime)
            # --------------------------------------------------------------------------------------------------------

            # --------------------------------------------------------------------------------------------------------
            print('4- Creating GLCM ...')
            startTime = dt.now()
            preGLCM = get_glcms(img_as_ubyte(rgb2gray(preSharp)))
            postGLCM = get_glcms(img_as_ubyte(rgb2gray(postSharp)))
            endTime = dt.now()
            print(endTime - startTime)
            # --------------------------------------------------------------------------------------------------------

            # --------------------------------------------------------------------------------------------------------
            print('5- Creating DCVA ...')
            startTime = dt.now()
            deepCVA = dcva(preChangeImage, postChangeImage, layers=[2], feature=True)
            endTime = dt.now()
            print(endTime - startTime)
            # --------------------------------------------------------------------------------------------------------

            # --------------------------------------------------------------------------------------------------------
            print('6- Creating Difference Image ...')
            startTime = dt.now()
            difference = img_as_float(postChangeImage) - img_as_float(preChangeImage)
            endTime = dt.now()
            print(endTime - startTime)
            # --------------------------------------------------------------------------------------------------------

            # --------------------------------------------------------------------------------------------------------
            print('7- Data type conversion ...')
            preChangeImage = minmax_scale(preChangeImage.reshape((1024*1024, 3))).reshape((1024, 1024, 3))
            postChangeImage = minmax_scale(postChangeImage.reshape((1024*1024, 3))).reshape((1024, 1024, 3))
            preLBP = minmax_scale(preLBP.reshape((1024*1024, 1))).reshape((1024, 1024))
            postLBP = minmax_scale(postLBP.reshape((1024*1024, 1))).reshape((1024, 1024))
            preGLCM = minmax_scale(preGLCM.reshape((1024*1024, 5))).reshape((1024, 1024, 5))
            postGLCM = minmax_scale(postGLCM.reshape((1024*1024, 5))).reshape((1024, 1024, 5))
            deepCVA = minmax_scale(deepCVA.reshape((1024*1024, 1))).reshape((1024, 1024))
            difference = minmax_scale(difference.reshape((1024*1024, 3))).reshape((1024, 1024, 3))
            # --------------------------------------------------------------------------------------------------------

            # --------------------------------------------------------------------------------------------------------
            p2, p98 = np.percentile(preGLCM[:, :, 2], (2, 98))
            preGLCM[:, :, 2] = exposure.rescale_intensity(preGLCM[:, :, 2], in_range=(p2, p98))
            p2, p98 = np.percentile(postGLCM[:, :, 2], (2, 98))
            preGLCM[:, :, 2] = exposure.rescale_intensity(postGLCM[:, :, 2], in_range=(p2, p98))
            # --------------------------------------------------------------------------------------------------------

            # --------------------------------------------------------------------------------------------------------
            print('8- Creating PCAs ...')
            startTime = dt.now()

            pca_pre = PCA(n_components=1).fit_transform(
                np.dstack((preChangeImage, preGLCM)).reshape((1024 * 1024, 8))).reshape((1024, 1024))
            pca_pre = minmax_scale(pca_pre.reshape((1024*1024, 1))).reshape((1024, 1024))

            pca_post = PCA(n_components=1).fit_transform(
                np.dstack((postChangeImage, postGLCM)).reshape((1024 * 1024, 8))).reshape((1024, 1024))
            pca_post = minmax_scale(pca_post.reshape((1024*1024, 1))).reshape((1024, 1024))

            pca_diff = PCA(n_components=1).fit_transform(np.dstack((deepCVA, difference)).reshape((1024 * 1024, 4))).reshape(
                (1024, 1024))
            pca_diff = minmax_scale(pca_diff.reshape((1024*1024, 1))).reshape((1024, 1024))

            endTime = dt.now()
            print(endTime - startTime)
            # --------------------------------------------------------------------------------------------------------

            # --------------------------------------------------------------------------------------------------------
            print('9- Creating Extended Morphological Profiles ...')  # Extended MP: if PCA is used instead of the image
            startTime = dt.now()
            emp = mp.build_emp(base_image=np.dstack((pca_pre, pca_post, pca_diff)), num_openings_closings=(2*2)+1)
            endTime = dt.now()
            print(endTime - startTime)
            pca_emp = PCA(n_components=3).fit_transform(emp.reshape((1024 * 1024, 33))).reshape((1024, 1024, 3))
            pca_emp = minmax_scale(pca_emp.reshape((1024 * 1024, 3))).reshape((1024, 1024, 3))
            # --------------------------------------------------------------------------------------------------------

            # --------------------------------------------------------------------------------------------------------
            if showFeatures:
                plt.figure()
                plt.subplot(221), plt.imshow(pca_pre), plt.xticks([]), plt.yticks([])
                plt.subplot(222), plt.imshow(pca_post), plt.xticks([]), plt.yticks([])
                plt.subplot(223), plt.imshow(pca_diff), plt.xticks([]), plt.yticks([])
                plt.subplot(224), plt.imshow(pca_emp), plt.xticks([]), plt.yticks([])
                plt.subplots_adjust(left=0, right=0.01, top=0.01, bottom=0, wspace=0)
                plt.tight_layout(pad=0, w_pad=0, h_pad=0)

                # Disply all features except EMPs
                fig, ax = plt.subplots(3, 6, sharex=True, sharey=True)

                ax[0, 0].imshow(preChangeImage), plt.xticks([]), plt.yticks([])
                ax[0, 1].imshow(postChangeImage), plt.xticks([]), plt.yticks([])
                ax[0, 2].imshow(preSharp), plt.xticks([]), plt.yticks([])
                ax[0, 3].imshow(postSharp), plt.xticks([]), plt.yticks([])
                ax[0, 4].imshow(preLBP), plt.xticks([]), plt.yticks([])
                ax[0, 5].imshow(postLBP), plt.xticks([]), plt.yticks([])

                ax[1, 0].imshow(preGLCM[:, :, 0]), plt.xticks([]), plt.yticks([])
                ax[1, 1].imshow(preGLCM[:, :, 1]), plt.xticks([]), plt.yticks([])
                ax[1, 2].imshow(preGLCM[:, :, 2]), plt.xticks([]), plt.yticks([])
                ax[1, 3].imshow(preGLCM[:, :, 3]), plt.xticks([]), plt.yticks([])
                ax[1, 4].imshow(preGLCM[:, :, 4]), plt.xticks([]), plt.yticks([])
                ax[1, 5].imshow(deepCVA), plt.xticks([]), plt.yticks([])

                ax[2, 0].imshow(postGLCM[:, :, 0]), plt.xticks([]), plt.yticks([])
                ax[2, 1].imshow(postGLCM[:, :, 1]), plt.xticks([]), plt.yticks([])
                ax[2, 2].imshow(postGLCM[:, :, 2]), plt.xticks([]), plt.yticks([])
                ax[2, 3].imshow(postGLCM[:, :, 3]), plt.xticks([]), plt.yticks([])
                ax[2, 4].imshow(postGLCM[:, :, 4]), plt.xticks([]), plt.yticks([])
                ax[2, 5].imshow(difference), plt.xticks([]), plt.yticks([])

                plt.subplots_adjust(left=0, right=0.01, top=0.01, bottom=0, wspace=0)
                plt.tight_layout(pad=0, w_pad=0, h_pad=0)
                plt.show()

            featureArray = np.dstack((preChangeImage, postChangeImage, preLBP, postLBP, preGLCM, postGLCM,
                                      pca_pre, pca_post, pca_diff, pca_emp, deepCVA, difference))

            print('>>> Feature images are generated ...')
            n_samples = len(row['samplePixels'])
            features = np.empty((n_samples, n_features + 1))
            count = 0

            startTime = dt.now()
            for rr, cc in row['samplePixels']:
                features[count, :n_features] = featureArray[rr, cc, :]
                count += 1
            features[:, -1] = classes.index(c) + 1
            endTime = dt.now()
            print(endTime - startTime)

            FeatureVector = np.vstack((FeatureVector, features))
            print(FeatureVector.shape)

elif n_features == 95:
    for c in classes:
        print('Checking files for ' + c.upper() + ' in ' + disaster_type.upper() + ' ...')
        files1 = database1[disaster_type][c]
        files2 = database2[disaster_type][c]
        files = pd.concat((files1, files2))
        print('\t\t There are\t{}\timages for feature extraction.'.format(files.shape[0]))
        file_num = 1
        for idx, row in files.iterrows():
            print('>' * file_num + '{0} / {1} processed.'.format(file_num, files.shape[0]))
            file_num += 1

            file = row['img_name']
            postFile = file.replace('labels', 'images') + '.png'
            preFile = postFile.replace('post', 'pre')
            # --------------------------------------------------------------------------------------------------------
            print('\n\n1- Reading Data >>>' + ' Image ' + file.split('/')[-1])
            startTime = dt.now()
            preChangeImage = imread(preFile)
            postChangeImage = imread(postFile)
            endTime = dt.now()
            print(endTime - startTime)
            # --------------------------------------------------------------------------------------------------------
            print('2- Unsharp-Masking ...')
            startTime = dt.now()
            preSharp = unsharp_mask(cv2.medianBlur(preChangeImage, 3), kernel_size=(3, 3), sigma=1.5)
            postSharp = unsharp_mask(cv2.medianBlur(postChangeImage, 3), kernel_size=(3, 3), sigma=1.5)
            endTime = dt.now()
            print(endTime - startTime)
            # --------------------------------------------------------------------------------------------------------
            print('3- Creating LBP ...')
            startTime = dt.now()
            preLBP = get_lbp(rgb2gray(preSharp))
            postLBP = get_lbp(rgb2gray(postSharp))
            endTime = dt.now()
            print(endTime - startTime)
            # --------------------------------------------------------------------------------------------------------
            print('4- Creating GLCM ...')
            startTime = dt.now()
            preGLCM = get_glcms(img_as_ubyte(rgb2gray(preSharp)))
            postGLCM = get_glcms(img_as_ubyte(rgb2gray(postSharp)))
            endTime = dt.now()
            print(endTime - startTime)
            # --------------------------------------------------------------------------------------------------------
            print('5- Creating DCVA ...')
            startTime = dt.now()
            deepCVA = dcva(preChangeImage, postChangeImage, layers=[2], feature=True)
            endTime = dt.now()
            print(endTime - startTime)
            # --------------------------------------------------------------------------------------------------------
            print('6- Creating Difference Image ...')
            startTime = dt.now()
            difference = img_as_float(postChangeImage) - img_as_float(preChangeImage)
            endTime = dt.now()
            print(endTime - startTime)
            # --------------------------------------------------------------------------------------------------------
            print('7- Data type conversion ...')
            preSharp = minmax_scale(preSharp.reshape((1024*1024, 3))).reshape((1024, 1024, 3))
            postSharp = minmax_scale(postSharp.reshape((1024*1024, 3))).reshape((1024, 1024, 3))
            preLBP = minmax_scale(preLBP.reshape((1024*1024, 1))).reshape((1024, 1024))
            postLBP = minmax_scale(postLBP.reshape((1024*1024, 1))).reshape((1024, 1024))
            preGLCM = minmax_scale(preGLCM.reshape((1024*1024, 5))).reshape((1024, 1024, 5))
            postGLCM = minmax_scale(postGLCM.reshape((1024*1024, 5))).reshape((1024, 1024, 5))
            deepCVA = minmax_scale(deepCVA.reshape((1024*1024, 1))).reshape((1024, 1024))
            difference = minmax_scale(difference.reshape((1024*1024, 3))).reshape((1024, 1024, 3))
            # --------------------------------------------------------------------------------------------------------
            p2, p98 = np.percentile(preGLCM[:, :, 2], (2, 98))
            preGLCM[:, :, 2] = exposure.rescale_intensity(preGLCM[:, :, 2], in_range=(p2, p98))
            p2, p98 = np.percentile(postGLCM[:, :, 2], (2, 98))
            preGLCM[:, :, 2] = exposure.rescale_intensity(postGLCM[:, :, 2], in_range=(p2, p98))
            # --------------------------------------------------------------------------------------------------------
            print('8- Creating PCAs ...')
            startTime = dt.now()
            pca_pre = PCA(n_components=2).fit_transform(
                np.dstack((preSharp, preGLCM)).reshape((1024 * 1024, 8))).reshape((1024, 1024))
            pca_pre = minmax_scale(pca_pre.reshape((1024*1024, 2))).reshape((1024, 1024, 2))
            pca_post = PCA(n_components=2).fit_transform(
                np.dstack((postSharp, postGLCM)).reshape((1024 * 1024, 8))).reshape((1024, 1024))
            pca_post = minmax_scale(pca_post.reshape((1024*1024, 2))).reshape((1024, 1024, 2))
            pca_diff = PCA(n_components=2).fit_transform(np.dstack((deepCVA, difference)).reshape((1024 * 1024, 4))).reshape(
                (1024, 1024, 2))
            pca_diff = minmax_scale(pca_diff.reshape((1024*1024, 2))).reshape((1024, 1024, 2))
            endTime = dt.now()
            print(endTime - startTime)
            # --------------------------------------------------------------------------------------------------------
            print('9- Creating Extended Morphological Profiles ...')  # Extended MP: if PCA is used instead of the image
            startTime = dt.now()
            emp = mp.build_emp(base_image=np.dstack((pca_pre, pca_post, pca_diff)), num_openings_closings=(2*2)+1)
            endTime = dt.now()
            print(endTime - startTime)
            pca_emp = minmax_scale(pca_emp.reshape((1024 * 1024, 66))).reshape((1024, 1024, 66))
            # --------------------------------------------------------------------------------------------------------
            featureArray = np.dstack((preSharp, postSharp, preLBP, postLBP, preGLCM, postGLCM,
                                      pca_pre, pca_post, pca_diff, pca_emp, deepCVA, difference))

            print('>>> Feature images are generated ...')
            n_samples = len(row['samplePixels'])
            features = np.empty((n_samples, 95))
            count = 0

            startTime = dt.now()
            for rr, cc in row['samplePixels']:
                features[count, :95] = featureArray[rr, cc, :]
                count += 1
            features[:, -1] = classes.index(c) + 1
            endTime = dt.now()
            print(endTime - startTime)

            FeatureVector = np.vstack((FeatureVector, features))
            print(FeatureVector.shape)

elif n_features == 18:
    for c in classes:
        print('Checking files for ' + c.upper() + ' in ' + disaster_type.upper() + ' ...')
        files1 = database1[disaster_type][c]
        files2 = database2[disaster_type][c]
        files = pd.concat((files1, files2))
        print('\t\t There are\t{}\timages for feature extraction.'.format(files.shape[0]))
        file_num = 1
        for idx, row in files.iterrows():
            print('>' * file_num + '{0} / {1} processed.'.format(file_num, files.shape[0]))
            file_num += 1

            file = row['img_name']
            postFile = file.replace('labels', 'images') + '.png'
            preFile = postFile.replace('post', 'pre')
            # --------------------------------------------------------------------------------------------------------
            print('\n\n1- Reading Data >>>' + ' Image ' + file.split('/')[-1])
            startTime = dt.now()
            preChangeImage = imread(preFile)
            postChangeImage = imread(postFile)
            endTime = dt.now()
            print(endTime - startTime)
            # --------------------------------------------------------------------------------------------------------
            print('2- Unsharp-Masking ...')
            startTime = dt.now()
            preSharp = unsharp_mask(cv2.medianBlur(preChangeImage, 3), kernel_size=(3, 3), sigma=1.5)
            postSharp = unsharp_mask(cv2.medianBlur(postChangeImage, 3), kernel_size=(3, 3), sigma=1.5)
            endTime = dt.now()
            print(endTime - startTime)
            # --------------------------------------------------------------------------------------------------------
            print('3- Creating LBP ...')
            startTime = dt.now()
            preLBP = get_lbp(rgb2gray(preSharp))
            postLBP = get_lbp(rgb2gray(postSharp))
            endTime = dt.now()
            print(endTime - startTime)
            # --------------------------------------------------------------------------------------------------------
            print('4- Creating GLCM ...')
            startTime = dt.now()
            preGLCM = get_glcms(img_as_ubyte(rgb2gray(preSharp)))
            postGLCM = get_glcms(img_as_ubyte(rgb2gray(postSharp)))
            endTime = dt.now()
            print(endTime - startTime)
            # --------------------------------------------------------------------------------------------------------
            print('7- Data type conversion ...')
            preSharp = minmax_scale(preSharp.reshape((1024*1024, 3))).reshape((1024, 1024, 3))
            postSharp = minmax_scale(postSharp.reshape((1024*1024, 3))).reshape((1024, 1024, 3))
            preLBP = minmax_scale(preLBP.reshape((1024*1024, 1))).reshape((1024, 1024))
            postLBP = minmax_scale(postLBP.reshape((1024*1024, 1))).reshape((1024, 1024))
            preGLCM = minmax_scale(preGLCM.reshape((1024*1024, 5))).reshape((1024, 1024, 5))
            postGLCM = minmax_scale(postGLCM.reshape((1024*1024, 5))).reshape((1024, 1024, 5))
            # --------------------------------------------------------------------------------------------------------
            featureArray = np.dstack((preSharp, postSharp, preLBP, postLBP, preGLCM, postGLCM))
            print('>>> Feature images are generated ...')
            n_samples = len(row['samplePixels'])
            features = np.empty((n_samples, n_features + 1))
            count = 0
            startTime = dt.now()
            for rr, cc in row['samplePixels']:
                features[count, :n_features] = featureArray[rr, cc, :]
                count += 1
            features[:, -1] = classes.index(c) + 1
            endTime = dt.now()
            print(endTime - startTime)
            FeatureVector = np.vstack((FeatureVector, features))
            print(FeatureVector.shape)

elif n_features == 22:
    for c in classes:
        print('Checking files for ' + c.upper() + ' in ' + disaster_type.upper() + ' ...')
        files1 = database1[disaster_type][c]
        files2 = database2[disaster_type][c]
        files = pd.concat((files1, files2))
        print('\t\t There are\t{}\timages for feature extraction.'.format(files.shape[0]))
        file_num = 1
        for idx, row in files.iterrows():
            print('>'*file_num + '{0} / {1} processed.'.format(file_num, files.shape[0]))
            file_num += 1

            file = row['img_name']
            postFile = file.replace('labels', 'images') + '.png'
            preFile = postFile.replace('post', 'pre')
            # --------------------------------------------------------------------------------------------------------
            print('\n\n1- Reading Data >>>' + ' Image ' + file.split('/')[-1])
            startTime = dt.now()
            preChangeImage = imread(preFile)
            postChangeImage = imread(postFile)
            endTime = dt.now()
            print(endTime - startTime)
            # --------------------------------------------------------------------------------------------------------
            print('2- Unsharp-Masking ...')
            startTime = dt.now()
            preSharp = unsharp_mask(cv2.medianBlur(preChangeImage, 3), kernel_size=(3, 3), sigma=1.5)
            postSharp = unsharp_mask(cv2.medianBlur(postChangeImage, 3), kernel_size=(3, 3), sigma=1.5)
            endTime = dt.now()
            print(endTime - startTime)
            # --------------------------------------------------------------------------------------------------------
            print('3- Creating LBP ...')
            startTime = dt.now()
            preLBP = get_lbp(rgb2gray(preSharp))
            postLBP = get_lbp(rgb2gray(postSharp))
            endTime = dt.now()
            print(endTime - startTime)
            # --------------------------------------------------------------------------------------------------------
            print('4- Creating GLCM ...')
            startTime = dt.now()
            preGLCM = get_glcms(img_as_ubyte(rgb2gray(preSharp)))
            postGLCM = get_glcms(img_as_ubyte(rgb2gray(postSharp)))
            endTime = dt.now()
            print(endTime - startTime)
            # --------------------------------------------------------------------------------------------------------
            print('5- Creating DCVA ...')
            startTime = dt.now()
            deepCVA = dcva(preChangeImage, postChangeImage, layers=[2], feature=True)
            endTime = dt.now()
            print(endTime - startTime)
            # --------------------------------------------------------------------------------------------------------
            print('6- Creating Difference Image ...')
            startTime = dt.now()
            difference = img_as_float(postChangeImage) - img_as_float(preChangeImage)
            endTime = dt.now()
            print(endTime - startTime)
            # --------------------------------------------------------------------------------------------------------
            print('7- Data type conversion ...')
            preSharp = minmax_scale(preSharp.reshape((1024*1024, 3))).reshape((1024, 1024, 3))
            postSharp = minmax_scale(postSharp.reshape((1024*1024, 3))).reshape((1024, 1024, 3))
            preLBP = minmax_scale(preLBP.reshape((1024*1024, 1))).reshape((1024, 1024))
            postLBP = minmax_scale(postLBP.reshape((1024*1024, 1))).reshape((1024, 1024))
            preGLCM = minmax_scale(preGLCM.reshape((1024*1024, 5))).reshape((1024, 1024, 5))
            postGLCM = minmax_scale(postGLCM.reshape((1024*1024, 5))).reshape((1024, 1024, 5))
            deepCVA = minmax_scale(deepCVA.reshape((1024*1024, 1))).reshape((1024, 1024))
            difference = minmax_scale(difference.reshape((1024*1024, 3))).reshape((1024, 1024, 3))
            # --------------------------------------------------------------------------------------------------------
            p2, p98 = np.percentile(preGLCM[:, :, 2], (2, 98))
            preGLCM[:, :, 2] = exposure.rescale_intensity(preGLCM[:, :, 2], in_range=(p2, p98))
            p2, p98 = np.percentile(postGLCM[:, :, 2], (2, 98))
            preGLCM[:, :, 2] = exposure.rescale_intensity(postGLCM[:, :, 2], in_range=(p2, p98))
            # --------------------------------------------------------------------------------------------------------
            featureArray = np.dstack((preSharp, postSharp, preLBP, postLBP,
                                      preGLCM, postGLCM, deepCVA, difference))
            print('>>> Feature images are generated ...')
            n_samples = len(row['samplePixels'])
            features = np.empty((n_samples, n_features + 1))
            count = 0
            startTime = dt.now()
            for rr, cc in row['samplePixels']:
                features[count, :n_features] = featureArray[rr, cc, :]
                count += 1
            features[:, -1] = classes.index(c) + 1
            endTime = dt.now()
            print(endTime - startTime)
            FeatureVector = np.vstack((FeatureVector, features))
            print(FeatureVector.shape)

elif n_features == 6:
    for c in classes:
        print('Checking files for ' + c.upper() + ' in ' + disaster_type.upper() + ' ...')
        files1 = database1[disaster_type][c]
        files2 = database2[disaster_type][c]
        files = pd.concat((files1, files2))
        print('\t\t There are\t{}\timages for feature extraction.'.format(files.shape[0]))
        file_num = 1
        for idx, row in files.iterrows():
            print('>'*file_num + '{0} / {1} processed.'.format(file_num, files.shape[0]))
            file_num += 1

            file = row['img_name']
            postFile = file.replace('labels', 'images') + '.png'
            preFile = postFile.replace('post', 'pre')
            # --------------------------------------------------------------------------------------------------------
            print('\n\n1- Reading Data >>>' + ' Image ' + file.split('/')[-1])
            startTime = dt.now()
            preChangeImage = imread(preFile)
            postChangeImage = imread(postFile)
            endTime = dt.now()
            print(endTime - startTime)
            # --------------------------------------------------------------------------------------------------------
            print('7- Data type conversion ...')
            preChangeImage = minmax_scale(preChangeImage.reshape((1024*1024, 3))).reshape((1024, 1024, 3))
            postChangeImage = minmax_scale(postChangeImage.reshape((1024*1024, 3))).reshape((1024, 1024, 3))
            # --------------------------------------------------------------------------------------------------------
            featureArray = np.dstack((preChangeImage, postChangeImage))
            print('>>> Feature images are generated ...')
            n_samples = len(row['samplePixels'])
            features = np.empty((n_samples, n_features + 1))
            count = 0
            startTime = dt.now()
            for rr, cc in row['samplePixels']:
                features[count, :n_features] = featureArray[rr, cc, :]
                count += 1
            features[:, -1] = classes.index(c) + 1
            endTime = dt.now()
            print(endTime - startTime)
            FeatureVector = np.vstack((FeatureVector, features))
            print(FeatureVector.shape)

end_total = dt.now()
print('Total time spent ...')
print(end_total - start_total)
np.save('Features_' + disaster_type.title() + '_' + str(len(classes)) + 'class_' + str(n_features) + 'features.npy', FeatureVector)
# ======================================================================================================================

