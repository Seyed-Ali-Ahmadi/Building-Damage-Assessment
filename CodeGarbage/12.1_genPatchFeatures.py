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

warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

disaster_type = 'wind'
showFeatures = 0
# Maxpool(Gray(RGB(Diff))): 16 + Hist(LBP(Gray(Diff(RGB)))): 20 +
# Homo(Diff(RGB)): 15 + Hist(Diff(RGB)): 30 + Hist(HOG(RGB)): 15  ==> 96
n_features = 1536 * 2     # 1056 * 2, 96
w = 24
pad = 5
angles = [0, np.pi / 4, np.pi / 2]
distances = [1, 2, 3, 4, 5]

classes = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']

database1 = pickle.load(open("databaseDictionary_wind_2_patch.pkl", "rb"))
numBuildings = 0
for c in classes:
    for bboxes in database1[disaster_type][c]['samplePatches']:
        numBuildings += len(bboxes)

print('There are totally {0} building patches for the {1} disaster.'.format(numBuildings, disaster_type.upper()))

# Basic shape of feature vector
FeatureVector = np.empty((numBuildings, n_features + 1), dtype=np.float)

start_total = dt.now()

if n_features == 96:
    building_added = 0
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

            LBP_dif = get_lbp(rgb2gray(postimg - preimg), method='uniform')

            for region in row['samplePatches']:
                minr, minc, maxr, maxc = region
                try:
                    patch_pre = resize(preimg[minr - pad:maxr + pad, minc - pad:maxc + pad, :], (w, w, 3))
                    patch_pos = resize(postimg[minr - pad:maxr + pad, minc - pad:maxc + pad, :], (w, w, 3))
                    patch_lbp_dif = resize(LBP_dif[minr - pad:maxr + pad, minc - pad:maxc + pad], (w, w))
                except ValueError:
                    patch_pre = resize(preimg[minr:maxr, minc:maxc, :], (w, w, 3))
                    patch_pos = resize(postimg[minr:maxr, minc:maxc, :], (w, w, 3))
                    patch_lbp_dif = resize(LBP_dif[minr:maxr, minc:maxc], (w, w))

                diff = patch_pos - patch_pre
                hog_dif_feat = hog(diff, orientations=8, pixels_per_cell=(6, 6),
                                   cells_per_block=(1, 1), feature_vector=True, multichannel=True)
                GLCM_dif = greycomatrix(img_as_ubyte(rgb2gray(diff)),
                                        distances=distances, angles=angles, levels=256, symmetric=True)

                maxpool_flat = max_kernel(rgb2gray(diff))
                hist_lbp_dif, _ = np.histogram(patch_lbp_dif.ravel(), bins=20)
                glcm_dif = greycoprops(GLCM_dif, 'homogeneity').ravel()
                hist_hog, _ = np.histogram(hog_dif_feat, bins=15)
                hist_diff_r, _ = np.histogram(diff[:, :, 0], bins=10)
                hist_diff_g, _ = np.histogram(diff[:, :, 1], bins=10)
                hist_diff_b, _ = np.histogram(diff[:, :, 2], bins=10)

                FeatureVector[building_added, :-1] = np.concatenate([maxpool_flat, hist_lbp_dif, glcm_dif, hist_hog,
                                                                     hist_diff_r, hist_diff_g, hist_diff_b], axis=0)
                FeatureVector[building_added, -1] = classes.index(c) + 1
                building_added += 1

elif n_features == 1536 * 2:    # 1056
    from tensorflow.keras.applications import NASNetMobile, InceptionResNetV2
    from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
    import numpy as np
    from keras.models import Model
    from tensorflow.keras.layers import GlobalAveragePooling2D

    model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(75, 75, 3))
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    w = 32
    pad = 10
    building_added = 0
    previous = 0
    for c in classes:
        print('Checking files for ' + c.upper() + ' in ' + disaster_type.upper() + ' ...')
        files = database1[disaster_type][c]
        print('\t\t There are\t{}\timages for feature extraction.'.format(files.shape[0]))
        file_num = 1

        patches_PRE = []
        patches_POS = []

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

                patches_PRE.append(resize(patch_pre, (75, 75, 3)))
                patches_POS.append(resize(patch_pos, (75, 75, 3)))

                FeatureVector[building_added, -1] = classes.index(c) + 1
                building_added += 1

        imagesPRE = np.expand_dims(patches_PRE, axis=0)
        imagesPRE = imagesPRE.reshape((imagesPRE.shape[1], 75, 75, 3))
        imagesPOS = np.expand_dims(patches_POS, axis=0)
        imagesPOS = imagesPOS.reshape((imagesPOS.shape[1], 75, 75, 3))

        imagesPRE_preprocessed = imagesPRE.copy()
        imagesPOS_preprocessed = imagesPOS.copy()
        imagesPRE_preprocessed = preprocess_input(imagesPRE_preprocessed)
        imagesPOS_preprocessed = preprocess_input(imagesPOS_preprocessed)

        features = model.predict(imagesPRE_preprocessed)
        featuresFlat_pre = GlobalAveragePooling2D()(features).numpy()
        features = model.predict(imagesPOS_preprocessed)
        featuresFlat_pos = GlobalAveragePooling2D()(features).numpy()

        FeatureVector[previous:building_added, :1536] = featuresFlat_pre
        FeatureVector[previous:building_added, 1536:-1] = featuresFlat_pos
        print(previous, building_added)
        previous = building_added

    np.save('InceptionResNetV2_DeepFeaturesMean.npy', np.mean(FeatureVector[:, :-1], axis=0))
    FeatureVector[:, :-1] = FeatureVector[:, :-1] - np.mean(FeatureVector[:, :-1], axis=0)


end_total = dt.now()
print('Total time spent ...')
print(end_total - start_total)
np.save('Features_' + disaster_type.title() + '_' + str(len(classes)) + 'class_' + str(n_features) +
        'Patch_featuresInceptionResNetV2.npy', FeatureVector)



# # ---------
# import warnings
# warnings.simplefilter(action="ignore", category=FutureWarning)
#
# from keras.models import Model
# from keras.layers import Input
#
# model_name = 'vgg16'
# weights = 'imagenet'
# include_top = False
#
# if model_name == "vgg16":
#     from keras.applications.vgg16 import VGG16, preprocess_input
#     base_model = VGG16(weights=weights)
#     model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
#     image_size = (224, 224)
# elif model_name == "vgg19":
#     from keras.applications.vgg19 import VGG19, preprocess_input
#     base_model = VGG19(weights=weights)
#     model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
#     image_size = (224, 224)
# elif model_name == "resnet50":
#     from keras.applications.resnet50 import ResNet50, preprocess_input
#     base_model = ResNet50(weights=weights)
#     model = Model(input=base_model.input, output=base_model.get_layer('flatten').output)
#     image_size = (224, 224)
# elif model_name == "inceptionv3":
#     from keras.applications.inception_v3 import InceptionV3, preprocess_input
#     base_model = InceptionV3(include_top=include_top, weights=weights, input_tensor=Input(shape=(299, 299, 3)))
#     model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
#     image_size = (299, 299)
# elif model_name == "inceptionresnetv2":
#     from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
#     base_model = InceptionResNetV2(include_top=include_top, weights=weights, input_tensor=Input(shape=(299, 299, 3)))
#     model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
#     image_size = (299, 299)
# elif model_name == "mobilenet":
#     from keras.applications.mobilenet import MobileNet, preprocess_input
#     base_model = MobileNet(include_top=include_top, weights=weights, input_tensor=Input(shape=(224, 224, 3)),
#                            input_shape=(224, 224, 3))
#     model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
#     image_size = (224, 224)
# elif model_name == "xception":
#     from keras.applications.xception import Xception, preprocess_input
#     base_model = Xception(weights=weights)
#     model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
#     image_size = (299, 299)

