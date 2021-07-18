from Functions import unsharp_mask, standardize, pad_features
from Functions import read_jsondir, read_image_json
from Functions import get_lbp, get_glcms
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
import pickle
import time
import cv2

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

FeatureVector = np.empty((0, 17))

for imgID, item in enumerate(uniqueNames):
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
    print('>>> Creating Features ...')
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

        H = (int(np.ceil(max(rows))) + pad) - (int(np.floor(min(rows))) - pad)
        W = (int(np.ceil(max(cols))) + pad) - (int(np.floor(min(cols))) - pad)
        n_samples = int(np.sqrt(H * W))
        # Uniformly distributed
        sampleR = np.random.randint(int(np.floor(min(rows))), int(np.floor(max(rows))), n_samples)
        sampleC = np.random.randint(int(np.floor(min(cols))), int(np.floor(max(cols))), n_samples)

        for i in range(n_samples):
            vec = [imgID, bldID]
            for f in range(features_padded.shape[2]):
                vec.append(features_padded[sampleR[i], sampleC[i], f])
            # features_padded[sampleR[i], sampleC[i], 0] = 1
            vec.append(class_labels[class_names.index(damageType)])
            vec = np.array(vec, ndmin=2)
            FeatureVector = np.vstack((FeatureVector, vec))
        # plt.figure()
        # plt.title(str(imgID) + ', ' + str(bldID) + ', ' + damageType + '\n' +
        #           str(H) + ', ' + str(W) + ', ' + str(n_samples))
        # plt.imshow(features_padded[int(np.floor(min(rows))) - pad:int(np.ceil(max(rows))) + pad,
        #                            int(np.floor(min(cols))) - pad:int(np.ceil(max(cols))) + pad, 7])
        # plt.show()
    print('         ' + str(round(time.time() - t0, 2)))
    print('<<< Creating Features ... Finished!')
    print('By now:  ', FeatureVector.shape)

print('Finally:   ', FeatureVector.shape)

pickle.dump(FeatureVector, open(root + 'Features.p', 'wb'))
plt.figure(), plt.hist(FeatureVector[:, -1].ravel()), plt.show()

# print('5- Fitting classifier ...')
# t0 = time.time()
# clf = SVC(probability=True)
# clf.fit(FeatureVector[:, 2:-1], FeatureVector[:, -1])
# print('         ' + str(round(time.time() - t0, 2)))
#
# print('---- Writing File.')
# pickle.dump(clf, open(root + 'classifier.p', 'wb'))
