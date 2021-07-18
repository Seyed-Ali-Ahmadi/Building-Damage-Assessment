from skimage.color import rgb2gray
from skimage.io import imread
import matplotlib.pyplot as plt
from Functions import unsharp_mask, get_lbp, get_glcms, max_kernel
from datetime import datetime as dt
from skimage.util import img_as_float, img_as_ubyte
from sklearn.preprocessing import minmax_scale
from skimage.feature import greycomatrix, greycoprops
from skimage.transform import resize
from skimage.feature import hog
import numpy as np
from skimage import exposure
import cv2
import pickle
import pandas as pd
import warnings
import logging

from tensorflow.keras.applications import NASNetMobile, InceptionResNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

disaster_type = 'wind'
showFeatures = 0
# Maxpool(Gray(RGB(Diff))): 16 + Hist(LBP(Gray(Diff(RGB)))): 20 +
# Homo(Diff(RGB)): 15 + Hist(Diff(RGB)): 30 + Hist(HOG(RGB)): 15  ==> 96
w = 32
pad = 10
angles = [0, np.pi / 4, np.pi / 2]
distances = [1, 2, 3, 4, 5]

classes = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']

database1 = pickle.load(open("databaseDictionary_wind_2_patch.pkl", "rb"))
numBuildings = 0
for c in classes:
    for bboxes in database1[disaster_type][c]['samplePatches']:
        numBuildings += len(bboxes)

print('There are totally {0} building patches for the {1} disaster.'.format(numBuildings, disaster_type.upper()))

model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(75, 75, 3))
# model = NASNetMobile(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
patches_PRE = []
patches_POS = []

building_added = 0
count = 0
for c in classes:
    print('Checking files for ' + c.upper() + ' in ' + disaster_type.upper() + ' ...')
    files = database1[disaster_type][c]
    print('\t\t There are\t{}\timages for feature extraction.'.format(files.shape[0]))
    file_num = 1
    for idx, row in files.iterrows():
        print('>' * file_num + '{0} / {1} processed.'.format(file_num, files.shape[0]))
        file_num += 1

        file = row['img_name']
        postFile = file.replace('labels', 'images') + '.png'
        preFile = postFile.replace('post', 'pre')
        preimg = imread(row['img_name'].replace('labels', 'images').replace('post', 'pre') + '.png')
        postimg = imread(row['img_name'].replace('labels', 'images') + '.png')

        for region in row['samplePatches']:
            minr, minc, maxr, maxc = region

            try:
                patch_pre = resize(preimg[minr - pad:maxr + pad, minc - pad:maxc + pad, :], (w, w, 3))
                patch_pos = resize(postimg[minr - pad:maxr + pad, minc - pad:maxc + pad, :], (w, w, 3))
            except ValueError:
                patch_pre = resize(preimg[minr:maxr, minc:maxc, :], (w, w, 3))
                patch_pos = resize(postimg[minr:maxr, minc:maxc, :], (w, w, 3))

            # patches_PRE.append(resize(patch_pre, (224, 224, 3)))
            # patches_POS.append(resize(patch_pos, (224, 224, 3)))
            patches_PRE.append(resize(patch_pre, (75, 75, 3)))
            patches_POS.append(resize(patch_pos, (75, 75, 3)))

            count += 1
            print(count)
            if count == 20:
                break

        if count == 20:
            break
    if count == 20:
        break

imagesPRE = np.expand_dims(patches_PRE, axis=0)
# imagesPRE = imagesPRE.reshape((imagesPRE.shape[1], 224, 224, 3))
imagesPRE = imagesPRE.reshape((imagesPRE.shape[1], 75, 75, 3))
imagesPOS = np.expand_dims(patches_POS, axis=0)
# imagesPOS = imagesPOS.reshape((imagesPOS.shape[1], 224, 224, 3))
imagesPOS = imagesPOS.reshape((imagesPOS.shape[1], 75, 75, 3))

imagesPRE_preprocessed = imagesPRE.copy()
imagesPOS_preprocessed = imagesPOS.copy()
imagesPRE_preprocessed = preprocess_input(imagesPRE_preprocessed)
imagesPOS_preprocessed = preprocess_input(imagesPOS_preprocessed)

features = model.predict(imagesPRE_preprocessed)
featuresFlat_pre = GlobalAveragePooling2D()(features).numpy()
features = model.predict(imagesPOS_preprocessed)
featuresFlat_pos = GlobalAveragePooling2D()(features).numpy()

featuresFlat_pre = featuresFlat_pre - np.mean(featuresFlat_pre, axis=0)
featuresFlat_pos = featuresFlat_pos - np.mean(featuresFlat_pos, axis=0)
print(featuresFlat_pos.shape)

plt.figure()
plt.subplot(211)
plt.plot(featuresFlat_pre.T, '.')
plt.subplot(212)
plt.plot(featuresFlat_pos.T, '.')
plt.show()

for i in range(imagesPOS_preprocessed.shape[0]):
    plt.figure()
    plt.subplot(221), plt.imshow(imagesPRE[i, :, :, :])
    plt.subplot(222), plt.imshow(imagesPOS[i, :, :, :])
    plt.subplot(223), plt.plot(featuresFlat_pre[i, :], 'k')
    plt.subplot(224), plt.plot(featuresFlat_pos[i, :], 'b')
    plt.show()

