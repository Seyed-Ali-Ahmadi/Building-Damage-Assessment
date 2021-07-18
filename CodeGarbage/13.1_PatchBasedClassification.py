from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.svm import LinearSVC as lsvm
from sklearn.linear_model import LogisticRegression as lr
import datetime as dt
from sklearn.utils import shuffle
import random
import pickle
from skimage.color import rgb2gray
from skimage.io import imread
import matplotlib.pyplot as plt
from Functions import get_lbp, max_kernel, create_mask
from skimage.util import img_as_ubyte
from sklearn.preprocessing import minmax_scale
from skimage.feature import greycomatrix, greycoprops
from skimage.transform import resize
from skimage.measure import regionprops
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

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# n_features = 1536 * 2     # 1536 * 2, 96
n_features = None
rotation_forest = False
xgboost = False
lgbm = True

if n_features == 96:
    w = 24
    pad = 5
    angles = [0, np.pi / 4, np.pi / 2]
    distances = [1, 2, 3, 4, 5]

    # ----------------------------------------------------------------------------------------------
    features96_wind = np.load('Features_Wind_4class_96Patch_features2.npy')
    features96_wind = shuffle(features96_wind)
    features96_wind[:, :-1] = minmax_scale(features96_wind[:, :-1])
    print(features96_wind.shape, np.max(features96_wind, axis=0), np.min(features96_wind, axis=0))

    RF = rfc(n_jobs=-1, warm_start=True, n_estimators=200, max_depth=10)  # , random_state=2
    start_train = dt.datetime.now()
    RF.fit(features96_wind[:, :-1], features96_wind[:, -1])
    end_train = dt.datetime.now()
    print(end_train - start_train)
    pickle.dump(RF, open('./training_0/RF_200_10_random_4class_96features2_patch.pkl', 'wb'))

    # ----------------------------------------------------------------------------------------------
    disaster_type = 'wind'
    classes = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']

    # Read the database and filter unnecessary files
    labelsDB = pd.read_pickle('All_Data_Props.pkl')
    labelsDB = labelsDB[labelsDB['Group'] != 'Test']
    labelsDB_post = labelsDB[labelsDB['Pre_Post'] == 'post']
    labelsDB_post_gt0 = labelsDB_post[labelsDB_post['buildings#'] > 0]
    labelsDB_post_gt0_disaster = labelsDB_post_gt0[labelsDB_post_gt0['disaster_type'] == disaster_type].reset_index()
    print(labelsDB_post_gt0_disaster.shape, ' size of database of post disaster images which have buildings.')
    print(labelsDB_post_gt0_disaster)

    # ----------------------------------------------------------------------------------------------
    df = labelsDB_post_gt0_disaster[['destroyed#', 'minor-damage#', 'major-damage#', 'no-damage#']]
    df = pd.concat((labelsDB_post_gt0_disaster, df.prod(axis=1)), axis=1).sort_values(by=0, ascending=False).drop(
        columns=0).head(60)

    image_number = 0
    start_prediction = dt.datetime.now()
    for idx, row in df.iterrows():
        print(image_number)
        image_number += 1
        file = row['img_name']
        Labels = pd.read_json(open(file + '.json', ))['features']['xy']
        build_loc = create_mask(Labels, wclassification=False).astype(int)

        postimg = imread(file.replace('labels', 'images') + '.png')
        preimg = imread(file.replace('labels', 'images').replace('post', 'pre') + '.png')

        LBP_dif = get_lbp(rgb2gray(postimg - preimg), method='uniform')

        classified_image = np.zeros((1024, 1024))
        n_patches = np.zeros((np.amax(build_loc), 96))
        building_num = 0
        for region in regionprops(build_loc):
            minr, minc, maxr, maxc = region.bbox
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

            n_patches[building_num, :] = np.concatenate([maxpool_flat, hist_lbp_dif, glcm_dif, hist_hog,
                                                         hist_diff_r, hist_diff_g, hist_diff_b], axis=0)
            building_num += 1

        n_patches = minmax_scale(n_patches)
        n_classes = RF.predict(n_patches)
        count = 0
        for region in regionprops(build_loc):
            for rr, cc in region.coords:
                classified_image[rr, cc] = n_classes[count]
            count += 1

        np.save('./training_0/classified_images_RF_200_10_random_4class_96Patch_features8060/' + file.split('/')[-1] + '.npy',
                classified_image)

    end_prediction = dt.datetime.now()
    print(end_prediction - start_prediction)

if n_features == 1536*2:
    from tensorflow.keras.applications import NASNetMobile, InceptionResNetV2
    from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
    from keras.models import Model
    from tensorflow.keras.layers import GlobalAveragePooling2D
    import random

    randomfeatures = random.sample(range(1536 * 2), 1024)

    model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(75, 75, 3))
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    featuresMean = np.load('InceptionResNetV2_DeepFeaturesMean.npy')

    w = 32
    pad = 10
    features3072_wind = np.load('Features_Wind_4class_3072Patch_featuresInceptionResNetV2.npy')
    features3072_wind = shuffle(features3072_wind)
    print(features3072_wind.shape)
    # features1024 = features3072_wind[:, randomfeatures]

    RF = rfc(n_jobs=-1, warm_start=True, n_estimators=200)  # , random_state=2, max_depth=10
    start_train = dt.datetime.now()
    RF.fit(features3072_wind[:, :-1], features3072_wind[:, -1])
    # RF.fit(features1024, features3072_wind[:, -1])
    end_train = dt.datetime.now()
    print(end_train - start_train)
    pickle.dump(RF, open('./training_0/RF_200_10_random_4class_maxDepth_featuresInceptionResNetV2_patch.pkl', 'wb'))

    # SVM = lsvm(tol=1e-5, dual=False, max_iter=2000)
    # start_train = dt.datetime.now()
    # SVM.fit(features3072_wind[:, :-1], features3072_wind[:, -1])
    # end_train = dt.datetime.now()
    # print(end_train - start_train)
    # pickle.dump(SVM, open('./training_0/SVM_2000_featuresInceptionResNetV2_patch.pkl', 'wb'))

    # ----------------------------------------------------------------------------------------------
    disaster_type = 'wind'
    classes = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']

    # Read the database and filter unnecessary files
    labelsDB = pd.read_pickle('All_Data_Props.pkl')
    labelsDB = labelsDB[labelsDB['Group'] != 'Test']
    labelsDB_post = labelsDB[labelsDB['Pre_Post'] == 'post']
    labelsDB_post_gt0 = labelsDB_post[labelsDB_post['buildings#'] > 0]
    labelsDB_post_gt0_disaster = labelsDB_post_gt0[labelsDB_post_gt0['disaster_type'] == disaster_type].reset_index()
    print(labelsDB_post_gt0_disaster.shape, ' size of database of post disaster images which have buildings.')
    print(labelsDB_post_gt0_disaster)

    # ----------------------------------------------------------------------------------------------
    df = labelsDB_post_gt0_disaster[['destroyed#', 'minor-damage#', 'major-damage#', 'no-damage#']]
    df = pd.concat((labelsDB_post_gt0_disaster, df.prod(axis=1)), axis=1).sort_values(by=0, ascending=False).drop(
        columns=0).head(60)

    image_number = 0
    no = 0
    start_prediction = dt.datetime.now()
    for idx, row in df.iterrows():
        print(image_number)
        image_number += 1
        file = row['img_name']
        Labels = pd.read_json(open(file + '.json', ))['features']['xy']
        build_loc = create_mask(Labels, wclassification=False).astype(int)

        postimg = imread(file.replace('labels', 'images') + '.png')
        preimg = imread(file.replace('labels', 'images').replace('post', 'pre') + '.png')

        classified_image = np.zeros((1024, 1024))
        n_patches = np.zeros((np.amax(build_loc), 3072))
        building_num = 0
        patches_PRE = []
        patches_POS = []
        for region in regionprops(build_loc):
            minr, minc, maxr, maxc = region.bbox
            try:
                patch_pre = resize(preimg[minr - pad:maxr + pad, minc - pad:maxc + pad, :], (w, w, 3))
                patch_pos = resize(postimg[minr - pad:maxr + pad, minc - pad:maxc + pad, :], (w, w, 3))
            except ValueError:
                patch_pre = resize(preimg[minr:maxr, minc:maxc, :], (w, w, 3))
                patch_pos = resize(postimg[minr:maxr, minc:maxc, :], (w, w, 3))

            patches_PRE.append(resize(patch_pre, (75, 75, 3)))
            patches_POS.append(resize(patch_pos, (75, 75, 3)))

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

        try:
            n_patches[:, :1536] = featuresFlat_pre
            n_patches[:, 1536:] = featuresFlat_pos
        except ValueError:
            n_patches[:-1, :1536] = featuresFlat_pre
            n_patches[:-1, 1536:] = featuresFlat_pos

        n_patches = n_patches - featuresMean
        building_num += 1

        kind = 1
        if kind == 1:
            n_classes = RF.predict(n_patches[:, randomfeatures])
            # n_classes = SVM.predict(n_patches)
        elif kind == 2:
            if no == 0:
                from keras.models import model_from_json
                json_file = open('SavedModel.json', 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                loaded_model = model_from_json(loaded_model_json)
                loaded_model.load_weights("SavedModel.h5")
                print("Loaded model from disk")
                no = 1
            n_classes = loaded_model.predict(n_patches)
            n_classes = np.argmax(n_classes, axis=1) + 1

        count = 0
        for region in regionprops(build_loc):
            for rr, cc in region.coords:
                classified_image[rr, cc] = n_classes[count]
            count += 1

        np.save('./training_0/classified_images_SVM_2000_4class_featuresInceptionResNetV2_patch/' + file.split('/')[
            -1] + '.npy', classified_image)

    end_prediction = dt.datetime.now()
    print(end_prediction - start_prediction)


if rotation_forest:
    from tensorflow.keras.applications import NASNetMobile, InceptionResNetV2
    from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
    from keras.models import Model
    from tensorflow.keras.layers import GlobalAveragePooling2D
    import random
    from RotationForest import RotationForest

    model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(75, 75, 3))
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    featuresMean = np.load('InceptionResNetV2_DeepFeaturesMean.npy')

    w = 32
    pad = 10
    features3072_wind = np.load('Features_Wind_4class_3072Patch_featuresInceptionResNetV2.npy')
    features3072_wind = shuffle(features3072_wind)
    print(features3072_wind.shape)

    # randomfeatures = random.sample(range(1536 * 2), 50)
    # features50 = features3072_wind[:, randomfeatures]

    rf = RotationForest(n_trees=100, n_features=48, bootstrap=True)
    start_train = dt.datetime.now()
    rf.fit(features3072_wind[:, :-1], features3072_wind[:, -1])
    # rf.fit(features50, features3072_wind[:, -1])
    end_train = dt.datetime.now()
    print(end_train - start_train)
    pickle.dump(rf, open('./training_0/RotF_100_48_4class_featuresInceptionResNetV2_patch.pkl', 'wb'))

    # ----------------------------------------------------------------------------------------------
    disaster_type = 'wind'
    classes = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']

    # Read the database and filter unnecessary files
    labelsDB = pd.read_pickle('All_Data_Props.pkl')
    labelsDB = labelsDB[labelsDB['Group'] != 'Test']
    labelsDB_post = labelsDB[labelsDB['Pre_Post'] == 'post']
    labelsDB_post_gt0 = labelsDB_post[labelsDB_post['buildings#'] > 0]
    labelsDB_post_gt0_disaster = labelsDB_post_gt0[labelsDB_post_gt0['disaster_type'] == disaster_type].reset_index()
    print(labelsDB_post_gt0_disaster.shape, ' size of database of post disaster images which have buildings.')
    print(labelsDB_post_gt0_disaster)

    # ----------------------------------------------------------------------------------------------
    df = labelsDB_post_gt0_disaster[['destroyed#', 'minor-damage#', 'major-damage#', 'no-damage#']]
    df = pd.concat((labelsDB_post_gt0_disaster, df.prod(axis=1)), axis=1).sort_values(by=0, ascending=False).drop(
        columns=0).head(60)

    image_number = 0
    no = 0
    start_prediction = dt.datetime.now()
    for idx, row in df.iterrows():
        print(image_number)
        image_number += 1
        file = row['img_name']
        Labels = pd.read_json(open(file + '.json', ))['features']['xy']
        build_loc = create_mask(Labels, wclassification=False).astype(int)

        postimg = imread(file.replace('labels', 'images') + '.png')
        preimg = imread(file.replace('labels', 'images').replace('post', 'pre') + '.png')

        classified_image = np.zeros((1024, 1024))
        n_patches = np.zeros((np.amax(build_loc), 3072))
        building_num = 0
        patches_PRE = []
        patches_POS = []
        for region in regionprops(build_loc):
            minr, minc, maxr, maxc = region.bbox
            try:
                patch_pre = resize(preimg[minr - pad:maxr + pad, minc - pad:maxc + pad, :], (w, w, 3))
                patch_pos = resize(postimg[minr - pad:maxr + pad, minc - pad:maxc + pad, :], (w, w, 3))
            except ValueError:
                patch_pre = resize(preimg[minr:maxr, minc:maxc, :], (w, w, 3))
                patch_pos = resize(postimg[minr:maxr, minc:maxc, :], (w, w, 3))

            patches_PRE.append(resize(patch_pre, (75, 75, 3)))
            patches_POS.append(resize(patch_pos, (75, 75, 3)))

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

        try:
            n_patches[:, :1536] = featuresFlat_pre
            n_patches[:, 1536:] = featuresFlat_pos
        except ValueError:
            n_patches[:-1, :1536] = featuresFlat_pre
            n_patches[:-1, 1536:] = featuresFlat_pos

        n_patches = n_patches - featuresMean
        building_num += 1

        n_classes = rf.predict(n_patches)
        count = 0
        for region in regionprops(build_loc):
            for rr, cc in region.coords:
                classified_image[rr, cc] = n_classes[count]
            count += 1

        np.save('./training_0/classified_images_RotF_100_3_4class_featuresInceptionResNetV2_patch/' + file.split('/')[
            -1] + '.npy', classified_image)

    end_prediction = dt.datetime.now()
    print(end_prediction - start_prediction)


if xgboost:
    from tensorflow.keras.applications import NASNetMobile, InceptionResNetV2
    from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
    from keras.models import Model
    from tensorflow.keras.layers import GlobalAveragePooling2D
    import random
    from xgboost import XGBClassifier

    model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(75, 75, 3))
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    featuresMean = np.load('InceptionResNetV2_DeepFeaturesMean.npy')

    w = 32
    pad = 10
    features3072_wind = np.load('Features_Wind_4class_3072Patch_featuresInceptionResNetV2.npy')
    features3072_wind = shuffle(features3072_wind)
    print(features3072_wind.shape)

    xgb = XGBClassifier(nthread=2)
    start_train = dt.datetime.now()
    xgb.fit(features3072_wind[:, :-1], features3072_wind[:, -1])
    end_train = dt.datetime.now()
    print(end_train - start_train)
    pickle.dump(xgb, open('./training_0/XGB_4class_featuresInceptionResNetV2_patch.pkl', 'wb'))

    # ----------------------------------------------------------------------------------------------
    disaster_type = 'wind'
    classes = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']

    # Read the database and filter unnecessary files
    labelsDB = pd.read_pickle('All_Data_Props.pkl')
    labelsDB = labelsDB[labelsDB['Group'] != 'Test']
    labelsDB_post = labelsDB[labelsDB['Pre_Post'] == 'post']
    labelsDB_post_gt0 = labelsDB_post[labelsDB_post['buildings#'] > 0]
    labelsDB_post_gt0_disaster = labelsDB_post_gt0[labelsDB_post_gt0['disaster_type'] == disaster_type].reset_index()
    print(labelsDB_post_gt0_disaster.shape, ' size of database of post disaster images which have buildings.')
    print(labelsDB_post_gt0_disaster)

    # ----------------------------------------------------------------------------------------------
    df = labelsDB_post_gt0_disaster[['destroyed#', 'minor-damage#', 'major-damage#', 'no-damage#']]
    df = pd.concat((labelsDB_post_gt0_disaster, df.prod(axis=1)), axis=1).sort_values(by=0, ascending=False).drop(
        columns=0).head(60)

    image_number = 0
    no = 0
    start_prediction = dt.datetime.now()
    for idx, row in df.iterrows():
        print(image_number)
        image_number += 1
        file = row['img_name']
        Labels = pd.read_json(open(file + '.json', ))['features']['xy']
        build_loc = create_mask(Labels, wclassification=False).astype(int)

        postimg = imread(file.replace('labels', 'images') + '.png')
        preimg = imread(file.replace('labels', 'images').replace('post', 'pre') + '.png')

        classified_image = np.zeros((1024, 1024))
        n_patches = np.zeros((np.amax(build_loc), 3072))
        building_num = 0
        patches_PRE = []
        patches_POS = []
        for region in regionprops(build_loc):
            minr, minc, maxr, maxc = region.bbox
            try:
                patch_pre = resize(preimg[minr - pad:maxr + pad, minc - pad:maxc + pad, :], (w, w, 3))
                patch_pos = resize(postimg[minr - pad:maxr + pad, minc - pad:maxc + pad, :], (w, w, 3))
            except ValueError:
                patch_pre = resize(preimg[minr:maxr, minc:maxc, :], (w, w, 3))
                patch_pos = resize(postimg[minr:maxr, minc:maxc, :], (w, w, 3))

            patches_PRE.append(resize(patch_pre, (75, 75, 3)))
            patches_POS.append(resize(patch_pos, (75, 75, 3)))

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

        try:
            n_patches[:, :1536] = featuresFlat_pre
            n_patches[:, 1536:] = featuresFlat_pos
        except ValueError:
            n_patches[:-1, :1536] = featuresFlat_pre
            n_patches[:-1, 1536:] = featuresFlat_pos

        n_patches = n_patches - featuresMean
        building_num += 1

        n_classes = xgb.predict(n_patches)
        count = 0
        for region in regionprops(build_loc):
            for rr, cc in region.coords:
                classified_image[rr, cc] = n_classes[count]
            count += 1

        np.save('./training_0/classified_images_XGB_4class_featuresInceptionResNetV2_patch/' + file.split('/')[
            -1] + '.npy', classified_image)

    end_prediction = dt.datetime.now()
    print(end_prediction - start_prediction)


if lgbm:
    from tensorflow.keras.applications import NASNetMobile, InceptionResNetV2
    from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
    from keras.models import Model
    from tensorflow.keras.layers import GlobalAveragePooling2D
    import random
    import lightgbm as lgbm

    model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(75, 75, 3))
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    featuresMean = np.load('InceptionResNetV2_DeepFeaturesMean.npy')

    w = 32
    pad = 10
    features3072_wind = np.load('Features_Wind_4class_3072Patch_featuresInceptionResNetV2.npy')
    features3072_wind = shuffle(features3072_wind)
    print(features3072_wind.shape)

    lgbmCLS = lgbm.LGBMClassifier(n_estimators=200)
    start_train = dt.datetime.now()
    lgbmCLS.fit(features3072_wind[:, :-1], features3072_wind[:, -1])
    end_train = dt.datetime.now()
    print(end_train - start_train)
    pickle.dump(lgbmCLS, open('./training_0/LGBM_4class_featuresInceptionResNetV2_patch.pkl', 'wb'))

    # ----------------------------------------------------------------------------------------------
    disaster_type = 'wind'
    classes = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']

    # Read the database and filter unnecessary files
    labelsDB = pd.read_pickle('All_Data_Props.pkl')
    labelsDB = labelsDB[labelsDB['Group'] != 'Test']
    labelsDB_post = labelsDB[labelsDB['Pre_Post'] == 'post']
    labelsDB_post_gt0 = labelsDB_post[labelsDB_post['buildings#'] > 0]
    labelsDB_post_gt0_disaster = labelsDB_post_gt0[labelsDB_post_gt0['disaster_type'] == disaster_type].reset_index()
    print(labelsDB_post_gt0_disaster.shape, ' size of database of post disaster images which have buildings.')
    print(labelsDB_post_gt0_disaster)

    # ----------------------------------------------------------------------------------------------
    df = labelsDB_post_gt0_disaster[['destroyed#', 'minor-damage#', 'major-damage#', 'no-damage#']]
    df = pd.concat((labelsDB_post_gt0_disaster, df.prod(axis=1)), axis=1).sort_values(by=0, ascending=False).drop(
        columns=0).head(60)

    image_number = 0
    no = 0
    start_prediction = dt.datetime.now()
    for idx, row in df.iterrows():
        print(image_number)
        image_number += 1
        file = row['img_name']
        Labels = pd.read_json(open(file + '.json', ))['features']['xy']
        build_loc = create_mask(Labels, wclassification=False).astype(int)

        postimg = imread(file.replace('labels', 'images') + '.png')
        preimg = imread(file.replace('labels', 'images').replace('post', 'pre') + '.png')

        classified_image = np.zeros((1024, 1024))
        n_patches = np.zeros((np.amax(build_loc), 3072))
        building_num = 0
        patches_PRE = []
        patches_POS = []
        for region in regionprops(build_loc):
            minr, minc, maxr, maxc = region.bbox
            try:
                patch_pre = resize(preimg[minr - pad:maxr + pad, minc - pad:maxc + pad, :], (w, w, 3))
                patch_pos = resize(postimg[minr - pad:maxr + pad, minc - pad:maxc + pad, :], (w, w, 3))
            except ValueError:
                patch_pre = resize(preimg[minr:maxr, minc:maxc, :], (w, w, 3))
                patch_pos = resize(postimg[minr:maxr, minc:maxc, :], (w, w, 3))

            patches_PRE.append(resize(patch_pre, (75, 75, 3)))
            patches_POS.append(resize(patch_pos, (75, 75, 3)))

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

        try:
            n_patches[:, :1536] = featuresFlat_pre
            n_patches[:, 1536:] = featuresFlat_pos
        except ValueError:
            n_patches[:-1, :1536] = featuresFlat_pre
            n_patches[:-1, 1536:] = featuresFlat_pos

        n_patches = n_patches - featuresMean
        building_num += 1

        n_classes = lgbmCLS.predict(n_patches)
        count = 0
        for region in regionprops(build_loc):
            for rr, cc in region.coords:
                classified_image[rr, cc] = n_classes[count]
            count += 1

        np.save('./training_0/classified_images_LGBM_4class_featuresInceptionResNetV2_patch/' + file.split('/')[
            -1] + '.npy', classified_image)

    end_prediction = dt.datetime.now()
    print(end_prediction - start_prediction)

