"""
Generate image patches for deep learning networks.
"""

import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
import re

root = 'D:/00.University/data/data sets/BD/train/'
imageDir = root + 'images/'
labelDir = root + 'labels/'
selectedFiles = root + 'Floods & Earthquakes.txt'
width = 1024
height = 1024
with open(selectedFiles, 'r') as f:
    selectedItems = f.read()
    f.close()
selectedItems = set(re.split('_post_disaster\n|_pre_disaster\n', selectedItems))
selectedItems.remove('')

patches = np.empty((64, 64, 0), dtype=float)
count = 0
num_imgs = 0
address = []

pad = 10
for item in selectedItems:

    if count > 10000:
        import pickle
        file = open(root + 'patches.pkl', 'wb')
        pickle.dump([address, patches], file)
        file.close()
        break

    # Compute difference image.
    preImg = imread(imageDir + item + '_pre_disaster.png').astype(float) / 255
    postImg = imread(imageDir + item + '_post_disaster.png').astype(float) / 255
    diff = (postImg - preImg + 1) / 2
    diff = rgb2gray(cv2.copyMakeBorder(diff, pad, pad, pad, pad, cv2.BORDER_REPLICATE))
    num_imgs += 1
    print(num_imgs)
    # plt.figure(), plt.imshow(diff, cmap='gray'), plt.show()

    f = open(labelDir + item + '_post_disaster.json',)
    data = json.load(f)
    # Create a new building mask.
    building_mask = np.zeros((height + 2 * pad, width + 2 * pad))
    bounding_box = np.zeros((height + 2 * pad, width + 2 * pad))

    # Loop through buildings in each image.
    for bldID, building in enumerate(data['features']['xy']):
        address.append([item, bldID, building['properties']['subtype']])
        # Extract x-y coordinate of each vertex from decoding the json pattern.
        vertices = building['wkt'].partition('POLYGON ((')[2].partition('))')[0].split(', ')
        n_vertices = len(vertices)
        rows = []
        cols = []
        for vertex in vertices:
            cols.append(np.abs(float(vertex.split(' ')[0])) + pad)
            rows.append(np.abs(float(vertex.split(' ')[1])) + pad)
        # Obtain the image of each building and its sorrounding.
        building_crop = diff[int(np.floor(min(rows))) - pad:
                             int(np.ceil(max(rows))) + pad,
                             int(np.floor(min(cols))) - pad:
                             int(np.ceil(max(cols))) + pad]
        building_crop_resized = resize(building_crop, (64, 64), anti_aliasing=True, preserve_range=True)
        patches = np.dstack((patches, building_crop_resized))
        count += 1
        print(count)
        # plt.figure(), plt.imshow(building_crop_resized, cmap='gray'), plt.show()
