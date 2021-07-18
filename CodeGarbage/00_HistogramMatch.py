import re
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
from skimage.draw import polygon
from skimage.io import imread

root = 'D:/00.University/data/data sets/BD/train/'
imagesDir = root + 'images/'
labelsDir = root + 'labels/'
selectedFiles = root + 'Floods & Earthquakes.txt'
imageWidth = 1024
imageHeight = 1024


with open(selectedFiles, 'r') as f:
    selectedItems = f.read()
    f.close()

selectedItems = set(re.split('_post_disaster\n|_pre_disaster\n', selectedItems))
selectedItems.remove('')

# for item in selectedItems:
#     preImg = imread(imagesDir + item + '_pre_disaster' + '.png')
#     postImg = imread(imagesDir + item + '_post_disaster' + '.png')
#     plt.figure()
#     plt.subplot(131), plt.imshow(preImg)
#     plt.subplot(132), plt.imshow(postImg)
#     matched = match_histograms(postImg / 255, preImg / 255, multichannel=True)
#     plt.subplot(133), plt.imshow(matched)
#     plt.suptitle(item)
#     plt.show()


for item in selectedItems:
    building_mask_pre = np.zeros((imageHeight, imageWidth))
    building_mask_post = np.zeros((imageHeight, imageWidth))

    preLbl = json.load(open(labelsDir + item + '_pre_disaster' + '.json'),)
    postLbl = json.load(open(labelsDir + item + '_post_disaster' + '.json'),)
    preImg = imread(imagesDir + item + '_pre_disaster' + '.png')
    postImg = imread(imagesDir + item + '_post_disaster' + '.png')

    for building in preLbl['features']['xy']:
        vertices = building['wkt']
        vertices = vertices.partition('POLYGON ((')[2].partition('))')[0].split(', ')
        n_vertices = len(vertices)
        rows = []
        cols = []
        for vertex in vertices:
            cols.append(float(vertex.split(' ')[0]))
            rows.append(float(vertex.split(' ')[1]))
        rr, cc = polygon(rows, cols, (1024, 1024))
        building_mask_pre[rr, cc] = 1

    for building in postLbl['features']['xy']:
        damageClass = building['properties']['subtype']
        vertices = building['wkt']
        vertices = vertices.partition('POLYGON ((')[2].partition('))')[0].split(', ')
        n_vertices = len(vertices)
        rows = []
        cols = []
        for vertex in vertices:
            cols.append(float(vertex.split(' ')[0]))
            rows.append(float(vertex.split(' ')[1]))
        rr, cc = polygon(rows, cols, (1024, 1024))

        if damageClass == 'no-damage':
            building_mask_post[rr, cc] = 1
        if damageClass == 'minor-damage':
            building_mask_post[rr, cc] = 2
        if damageClass == 'major-damage':
            building_mask_post[rr, cc] = 3
        if damageClass == 'destroyed':
            building_mask_post[rr, cc] = 4

    matched = match_histograms(postImg / 255, preImg / 255, multichannel=True)
    diff = preImg.astype(float) - postImg.astype(float)
    plt.figure(figsize=(11, 6))
    plt.subplot(331), plt.imshow(preImg), plt.title('Pre')
    plt.subplot(332), plt.imshow(postImg), plt.title('Post')
    plt.subplot(333), plt.imshow(matched), plt.title('Matched')
    plt.subplot(334), plt.imshow(building_mask_pre, vmin=0, vmax=4)
    plt.subplot(335), plt.imshow(building_mask_post, vmin=0, vmax=4)
    plt.subplot(336), plt.imshow(diff, vmin=np.amin(diff), vmax=np.amax(diff)), plt.title('Diff (Pre - Post)')
    plt.subplot(337), plt.hist(preImg.flatten(), bins=256)
    plt.subplot(338), plt.hist(postImg.flatten(), bins=256)
    plt.subplot(339), plt.hist(diff.flatten(), bins=256)
    plt.suptitle(item)
    plt.show()
