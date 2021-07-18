import os
import pickle
import random
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.draw import polygon
from skimage.measure import regionprops
from Functions import create_mask
from sklearn.utils import shuffle
from matplotlib.patches import Rectangle


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

# How many building patches are required for training?
numOfPatchesPerClass = 2000

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
        patchCount = [0, 0, 0, 0]
        id = [0, 0, 0, 0]
        for c in classes:
            print('\t\t\tProcessing   ----------->  ' + c + ' class')
            df4class = perClass[perClass[c + '#'] > 0]
            try:
                while patchCount[classes.index(c)] < numOfPatchesPerClass:
                    patchCount[classes.index(c)] += int(df4class[c + '#'].iloc[id[classes.index(c)]])
                    id[classes.index(c)] += 1
            except IndexError:
                pass
            database[disType].update({c: df4class.iloc[:id[classes.index(c)]]})
        print(patchCount, id)
else:
    patchCount = [0, 0, 0, 0]
    id = [0, 0, 0, 0]
    for c in classes:
        print('\t\t\tProcessing      ----------->      ' + c + ' class:')
        df4class = shuffled[shuffled[c + '#'] > 0]
        while patchCount[classes.index(c)] < numOfPatchesPerClass:
            patchCount[classes.index(c)] += int(df4class[c + '#'].iloc[id[classes.index(c)]])
            id[classes.index(c)] += 1
        database['mixed'].update({c: df4class.iloc[:id[classes.index(c)]]})
    print(patchCount, id)
# ----------------------------------------------------------------------------------------------
# Extract building-patch bounding boxes in order to be used later as patches for training.
disaster_type = 'wind'  # 'volcano', 'flooding', 'wind', 'earthquake', 'tsunami', 'fire', 'mixed
for i, (Class, files) in enumerate(database[disaster_type].items()):
    database[disaster_type][Class] = database[disaster_type][Class].copy()
    database[disaster_type][Class]['samplePatches'] = [[] for x in range(len(files))]
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
                bounding_box = min(rr), min(cc), max(rr), max(cc)
                # Add extracted pixels to the database
                database[disaster_type][Class]['samplePatches'].iloc[file].append(bounding_box)

pickle.dump(database, open("databaseDictionary_wind_2_patch.pkl", "wb"))
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

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    plt.suptitle(file.split('/')[-1])
    ax1.imshow(preImg), plt.xticks([]), plt.yticks([]), ax1.set_title('pre image')
    ax2.imshow(postImg), plt.xticks([]), plt.yticks([]), ax2.set_title('post image')

    samplePatches = row['samplePatches']
    for bbox in samplePatches:
        rect1 = Rectangle((bbox[1], bbox[2]), bbox[3] - bbox[1], bbox[0] - bbox[2], edgecolor='b', facecolor='none')
        rect2 = Rectangle((bbox[1], bbox[2]), bbox[3] - bbox[1], bbox[0] - bbox[2], edgecolor='b', facecolor='none')
        ax1.add_patch(rect1)
        ax2.add_patch(rect2)

    plt.tight_layout()
    plt.show()







