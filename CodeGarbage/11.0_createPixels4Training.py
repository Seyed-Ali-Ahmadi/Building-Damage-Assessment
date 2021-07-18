"""
In this code, I want to select some pixels as samples for the
training procedure. I don't want to read any images as inputs
and just do this task by using file names (json files) and the
database which I created previously for "data properties"
extraction.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.draw import polygon
import datetime as dt
from sklearn.utils import shuffle
import random
from skimage.io import imread

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# ----------------------------------------------------------------------------------------------
# Read the database
labelsDB = pd.read_pickle('All_Data_Props.pkl')
print(labelsDB.shape, ' size of all database.')
# Use only training images and not the test ones.
labelsDB = labelsDB[labelsDB['Group'] != 'Test']

# Separate pre/post files from each other.
# Only the post images data will be used because the masks
# are the same and the classes are inside the post json files.
labelsDB_post = labelsDB[labelsDB['Pre_Post'] == 'post']
print(labelsDB_post.shape, ' size of database for post disaster images.')
labelsDB_post_gt0 = labelsDB_post[labelsDB_post['buildings#'] > 0]
print(labelsDB_post_gt0.shape, ' size of database of post disaster images which have buildings.')

# Get unique disaster types.
disTypes = pd.unique(labelsDB_post['disaster_type'])

# Building damage classes
classes = ['destroyed', 'minor-damage', 'major-damage', 'no-damage']
classNumbers = [1, 2, 3, 4]

# How many pixels are required for the training.
# Now, totally 200k pixels (50k each class)
numOfSamplesPerClass = 5 * 1e4

# Percent of extracting samples from each building.
# Now, 40% of each building's pixels are extracted as samples.
buildingPercent = 0.35

# Have an empty database to store required information for creating samples.
database = {}
for disType in disTypes:
    database.update({disType: {}})
database.update({'mixed': {}})

# Do you want to have samples for each disaster separately? (= 1)
# Or you want to train a model using all disasters' samples? (= 0)
separate_disasters = 1

# Shuffle the images in the database
shuffled = shuffle(labelsDB_post_gt0)   # , random_state=2
if separate_disasters:
    for disType in disTypes:
        print('\n\tProcessing   ----------->  disaster type ' + disType)
        perClass = shuffled[shuffled['disaster_type'] == disType]
        sampleCount = [0, 0, 0, 0]
        id = [0, 0, 0, 0]
        for c in classes:
            print('\t\t\tProcessing   ----------->  ' + c + ' class')
            df4class = perClass[perClass[c + '#'] > 0]
            try:
                while sampleCount[classes.index(c)] < numOfSamplesPerClass:
                    sampleCount[classes.index(c)] += \
                        int(buildingPercent * df4class[c + '#pix'].iloc[id[classes.index(c)]])
                    id[classes.index(c)] += 1
            except IndexError:
                pass
            database[disType].update({c: df4class.iloc[:id[classes.index(c)]]})
            # print(sampleCount)
            # print(id)
            # print(database[disType][c])
        print(sampleCount, id)
else:
    sampleCount = [0, 0, 0, 0]
    id = [0, 0, 0, 0]
    for c in classes:
        print('\t\t\tProcessing      ----------->      ' + c + ' class:')
        df4class = shuffled[shuffled[c + '#'] > 0]
        while sampleCount[classes.index(c)] < numOfSamplesPerClass:
            sampleCount[classes.index(c)] += \
                int(buildingPercent * df4class[c + '#pix'].iloc[id[classes.index(c)]])
            id[classes.index(c)] += 1
        database['mixed'].update({c: df4class.iloc[:id[classes.index(c)]]})
        # print(sampleCount)
        # print(id)
        # print(database['mixed'][c])
    print('\n\n', sampleCount, id)
# ----------------------------------------------------------------------------------------------
# Extract pixel locations in order to be used later as samples for training.
disaster_type = 'wind'  # 'volcano', 'flooding', 'wind', 'earthquake', 'tsunami', 'fire', 'mixed
for i, (Class, files) in enumerate(database[disaster_type].items()):
    database[disaster_type][Class] = database[disaster_type][Class].copy()
    database[disaster_type][Class]['samplePixels'] = [[] for x in range(len(files))]
    for file in range(len(files)):
        json = pd.read_json(open(files['img_name'].iloc[file] + '.json', ))['features']['xy']
        for building in json:
            if building['properties']['subtype'] == Class:
                vertices = building['wkt'].partition('POLYGON ((')[2].partition('))')[0].split(', ')
                rows = []
                cols = []
                for vertex in vertices:
                    cols.append(float(vertex.split(' ')[0]))
                    rows.append(float(vertex.split(' ')[1]))
                rr, cc = polygon(rows, cols, (1024, 1024))
                pixels = list(zip(rr, cc))
                # Add extracted pixels to the database
                database[disaster_type][Class]['samplePixels'].iloc[file].extend(
                    random.sample(pixels, int(len(pixels) * buildingPercent)))

import pickle
pickle.dump(database, open("databaseDictionary_wind_1.pkl", "wb"))
# ----------------------------------------------------------------------------------------------
# Check the correctness of extracted building samples by drawing them on the images.
damage_class = 'destroyed'  # 'destroyed', 'minor-damage', 'major-damage', 'no-damage'
print('Checking files for ' + damage_class.upper() + ' in ' + disaster_type.upper() + ' ...')
files = database[disaster_type][damage_class]
for idx, row in files.iterrows():
    file = row['img_name']
    postFile = file.replace('labels', 'images') + '.png'
    preFile = postFile.replace('post', 'pre')

    postImg = imread(postFile)
    preImg = imread(preFile)

    samplePixels = row['samplePixels']
    for rc in samplePixels:
        postImg[rc[0], rc[1], :] = [255, 255, 0]
        preImg[rc[0], rc[1], :] = [255, 255, 0]

    plt.figure()
    plt.suptitle(file.split('/')[-1])
    plt.subplot(121), plt.imshow(preImg), plt.xticks([]), plt.yticks([]), plt.title('pre image')
    plt.subplot(122), plt.imshow(postImg), plt.xticks([]), plt.yticks([]), plt.title('post image')
    plt.tight_layout()
    plt.show()
# ----------------------------------------------------------------------------------------------
