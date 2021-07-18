import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.svm import LinearSVC as lsvm
from sklearn.linear_model import LogisticRegression as lr
import pandas as pd
from skimage.draw import polygon
import datetime as dt
from sklearn.utils import shuffle
import random
from skimage.io import imread
import pickle

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# features28_wind = np.load('Features_Wind.npy')
# print(features28_wind.shape)

# ----------------------------------------------------------------------------------------------
# # Train a RF classifier with some default values
# RF = rfc(n_jobs=-1, warm_start=True, n_estimators=400, max_depth=20, random_state=1)
# start_train = dt.datetime.now()
# RF.fit(features28_wind[:, :-1], features28_wind[:, -1])
# end_train = dt.datetime.now()
# print(end_train - start_train)
# pickle.dump(RF, open('./training_0/RF_400_20.pkl', 'wb'))

# ----------------------------------------------------------------------------------------------
# # Train a SVM classifier with some default values
# SVM = lsvm(random_state=1, tol=1e-5, dual=False, max_iter=2000)
# start_train = dt.datetime.now()
# SVM.fit(features28_wind[:, :-1], features28_wind[:, -1])
# end_train = dt.datetime.now()
# print(end_train - start_train)
# pickle.dump(SVM, open('./training_0/SVM_2000.pkl', 'wb'))

# ----------------------------------------------------------------------------------------------
# # Train a LR classifier with some default values
# LR = lr(random_state=1, tol=1e-4, max_iter=2000, warm_start=True, n_jobs=-1)
# start_train = dt.datetime.now()
# LR.fit(features28_wind[:, :-1], features28_wind[:, -1])
# end_train = dt.datetime.now()
# print(end_train - start_train)
# pickle.dump(LR, open('./training_0/LR_1000.pkl', 'wb'))

# ----------------------------------------------------------------------------------------------
n_features = 6
features6_wind = np.load('Features_Wind_2class_6features.npy')
features6_wind = shuffle(features6_wind)
print(features6_wind.shape)
RF = rfc(n_jobs=-1, warm_start=True, n_estimators=200, max_depth=10)    # , random_state=2
start_train = dt.datetime.now()
RF.fit(features6_wind[:, :-1], features6_wind[:, -1])
end_train = dt.datetime.now()
print(end_train - start_train)
pickle.dump(RF, open('./training_0/RF_200_10_random_2class_6features.pkl', 'wb'))

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
# Classify buildings over these images (The first 20 images with most buildings at all classes)
from feature_method import gen_features as gf

df = labelsDB_post_gt0_disaster[['destroyed#', 'minor-damage#', 'major-damage#', 'no-damage#']]
df = pd.concat((labelsDB_post_gt0_disaster, df.prod(axis=1)), axis=1).sort_values(by=0, ascending=False).drop(columns=0).head(20)

start_prediction = dt.datetime.now()
for idx, row in df.iterrows():
    file = row['img_name']
    post_image = imread(file.replace('labels', 'images') + '.png')
    pre_image = imread(file.replace('labels', 'images').replace('post', 'pre') + '.png')
    featureArray = gf(pre_image, post_image, numfeatures=n_features)

    labels = pd.read_json(open(file + '.json', ))['features']['xy']
    Pixels = []
    for building in labels:
        vertices = building['wkt'].partition('POLYGON ((')[2].partition('))')[0].split(', ')
        rows = []
        cols = []
        for vertex in vertices:
            cols.append(float(vertex.split(' ')[0]))
            rows.append(float(vertex.split(' ')[1]))
        rr, cc = polygon(rows, cols, (1024, 1024))
        Pixels.extend(list(zip(rr, cc)))

    count = 0
    n_samples = np.empty((len(Pixels), n_features), dtype=np.float)
    for rr, cc in Pixels:
        n_samples[count, :] = featureArray[rr, cc, :]
        count += 1

    n_classes = RF.predict(n_samples)
    # n_classes = SVM.predict(n_samples)
    # n_classes = LR.predict(n_samples)
    classified_image = np.zeros((1024, 1024))
    count = 0
    for rr, cc in Pixels:
        classified_image[rr, cc] = n_classes[count]
        count += 1

    np.save('./training_0/classified_images_RF_200_10_random_2class_6features/' + file.split('/')[-1] + '.npy', classified_image)

end_prediction = dt.datetime.now()
print(end_prediction - start_prediction)
