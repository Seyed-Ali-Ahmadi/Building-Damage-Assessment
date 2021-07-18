from Functions import unsharp_mask, standardize, pad_features
from Functions import read_jsondir, read_image_json
from Functions import get_lbp, get_glcms
from skimage.draw import polygon
from skimage.morphology import closing as close
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import pickle
import joblib

# print more columns in console
np.set_printoptions(linewidth=1000)

root = 'D:/00.University/data/data sets/BD/train/'
imageDir = root + '1/'
labelDir = root + '2/'
# imageDir = root + 'subset_images/'
# labelDir = root + 'subset_labels/'

# remove _pre _post postfixes and obtain unique images
uniqueNames = sorted(list(read_jsondir(labelDir)), key=str.lower)
# Indices are unique.
print(uniqueNames)

class_names = ['destroyed', 'major-damage', 'minor-damage', 'no-damage', 'un-classified']
class_labels = [4, 3, 2, 1, -1]

TestVector = np.empty((0, 17))
print('0- Loading Classifier ...')
t0 = time.time()
clf = pickle.load(open(root + 'classifier_RF_estm20_depth3_job3_warm.p', 'rb'))
print('         ' + str(round(time.time() - t0, 2)))

predictionMap = np.zeros((1044, 1044), dtype=float)

for imgID, item in enumerate(uniqueNames):
    classificationMap = np.zeros((1044, 1044), dtype=int)
    plt.figure(figsize=(11, 7))
    plt.tight_layout()

    print('1- Reading Data ...')
    print('     Image ' + uniqueNames[imgID])
    t0 = time.time()
    # Images are gray-scale uint-8
    _, preImage, postJson, postImage = read_image_json(imageDir, labelDir, item)
    # Perform unsharp-masking
    smoothPre = unsharp_mask(cv2.medianBlur(preImage, 3), kernel_size=(3, 3), sigma=1.5)
    smoothPost = unsharp_mask(cv2.medianBlur(postImage, 3), kernel_size=(3, 3), sigma=1.5)
    print('         ' + str(round(time.time() - t0, 2)))

    # ------------------------- Extract LBP features ---------------------------
    print('2- Creating LBP ...')
    t0 = time.time()
    # Pre LBP Image
    preLBP = get_lbp(smoothPre)
    # Post LBP Image
    postLBP = get_lbp(smoothPost)
    # diffLBP = postLBP - preLBP
    # I think that Diff(LBP) is not useful. I can try it later.
    print('         ' + str(round(time.time() - t0, 2)))

    # ------------------------- Extract GLCM features ---------------------------
    print('3- Creating GLCM ...')
    t0 = time.time()
    preGLCM = get_glcms(smoothPre)
    postGLCM = get_glcms(smoothPost)
    print('         ' + str(round(time.time() - t0, 2)))

    # ------------------------------ Stack features -------------------------------
    print('4- Standardization ...')
    t0 = time.time()
    # Make data standard
    smoothPre_stndrd = standardize(smoothPre, show=False)
    smoothPost_stndrd = standardize(smoothPost, show=False)

    preLBP_stndrd = standardize(preLBP, show=False)
    postLBP_stndrd = standardize(postLBP, show=False)

    preGLCM_stndrd = standardize(preGLCM, show=False)
    postGLCM_stndrd = standardize(postGLCM, show=False)

    # Stacking
    features_pre = np.dstack((smoothPre_stndrd, preLBP_stndrd, preGLCM_stndrd))
    features_post = np.dstack((smoothPost_stndrd, postLBP_stndrd, postGLCM_stndrd))
    features = np.dstack((features_pre, features_post))
    features_padded = pad_features(features, pad=10)
    print('         ' + str(round(time.time() - t0, 2)))

    # ------------------------------ Create Mask -------------------------------
    print('>>> Predicting ...')
    t0 = time.time()
    pad = 10
    # Loop through buildings in each image.
    for bldID, building in enumerate(postJson['features']['xy']):
        damageType = building['properties']['subtype']
        # Extract x-y coordinate of each vertex from decoding the json pattern.
        vertices = building['wkt'].partition('POLYGON ((')[2].partition('))')[0].split(', ')
        n_vertices = len(vertices)

        rows = []
        cols = []
        for vertex in vertices:
            cols.append(float(vertex.split(' ')[0]) + pad)
            rows.append(float(vertex.split(' ')[1]) + pad)

        rr, cc = polygon(rows, cols, (1024 + 2 * pad, 1024 + 2 * pad))

        # Buidling-by-Building classification
        toClassify = np.zeros((len(rr), 19), dtype=float)
        toClassify[:, 0] = imgID
        toClassify[:, 1] = bldID
        toClassify[:, 2] = rr
        toClassify[:, 3] = cc

        for pixel in range(len(rr)):
            toClassify[pixel, 4:-1] = features_padded[int(toClassify[pixel, 2]), int(toClassify[pixel, 3]), :]

        Classified = clf.predict(toClassify[:, 4:-1])
        for pixel in range(len(rr)):
            classificationMap[rr[pixel], cc[pixel]] = int(Classified[pixel])

        # Pixel-by-Pixel classification
        # for rc in range(len(rr)):
        #     vec = [imgID, bldID]
        #     for f in range(features_padded.shape[2]):
        #         vec.append(features_padded[rr[rc], cc[rc], f])
        #     vec.append(class_labels[class_names.index(damageType)])
        #     vec = np.array(vec, ndmin=2)
        #     print(vec[0, 2:-1])
        #     classificationMap[rr[rc], cc[rc]] = clf.predict(vec[0, 2:-1].reshape(1, -1))

    classificationMap = close(classificationMap, selem=np.ones((3, 3)))
    np.save(imageDir + uniqueNames[imgID] + '.npy', classificationMap)
    print('         ' + str(round(time.time() - t0, 2)))
    plt.imshow(classificationMap.astype(int), cmap='gray')
    plt.title(uniqueNames[imgID])
    plt.savefig(imageDir + uniqueNames[imgID] + '_RF_1' + '.png')
    plt.show(block=False)
    plt.pause(2)
    plt.close('all')
    print('<<< Classification ... Finished!')






