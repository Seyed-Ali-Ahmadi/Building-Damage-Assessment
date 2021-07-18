import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray


root = 'D:/00.University/data/data sets/BD/train/'
imageDir = root + 'subset_images/'
labelDir = root + 'subset_labels/'
jsonFiles = os.listdir(labelDir)

# Add some amount of padding to be able to handle the problem of the borders.
pad = 10

for jsonFile in jsonFiles:
    f = open(labelDir + jsonFile, )
    data = json.load(f)

    # Create a new building mask.
    building_mask = np.zeros((1024 + 2 * pad, 1024 + 2 * pad))
    bounding_box = np.zeros((1024 + 2 * pad, 1024 + 2 * pad))

    # Read the corresponding image of the same JSON file.
    image = imread(imageDir + jsonFile[:-4] + 'png')
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    # Loop through buildings in each image.
    for building in data['features']['xy']:

        # Extract x-y coordinate of each vertex from decoding the json pattern.
        vertices = building['wkt'].partition('POLYGON ((')[2].partition('))')[0].split(', ')
        n_vertices = len(vertices)

        rows = []
        cols = []
        for vertex in vertices:
            cols.append(float(vertex.split(' ')[0]) + pad)
            rows.append(float(vertex.split(' ')[1]) + pad)

        # Fill the location of each building.
        # Use a greater image due to padding.
        rr, cc = polygon(rows, cols, (1024 + 2 * pad, 1024 + 2 * pad))
        building_mask[rr, cc] = 1
        # Fill the location of its bounding box for further use.
        br, bc = polygon([min(rows), min(rows), max(rows), max(rows), min(rows)],
                         [min(cols), max(cols), max(cols), min(cols), min(cols)],
                         (1024 + 2 * pad, 1024 + 2 * pad))
        bounding_box[br, bc] = 1

        mask_box = building_mask + bounding_box
        # mask_box = building_mask
        # Obtain the image of each building and its sorrounding.
        building_crop = image[int(np.floor(min(rows))) - pad:
                              int(np.ceil(max(rows))) + pad,
                              int(np.floor(min(cols))) - pad:
                              int(np.ceil(max(cols))) + pad, :]
        mask_crop = mask_box[int(np.floor(min(rows))) - pad:
                             int(np.ceil(max(rows))) + pad,
                             int(np.floor(min(cols))) - pad:
                             int(np.ceil(max(cols))) + pad]

        mask_crop_resized = resize(mask_crop, (50, 50), anti_aliasing=True, preserve_range=True)
        building_crop_resized = resize(building_crop, (50, 50), anti_aliasing=True, preserve_range=True)

        fig, axes = plt.subplots(2, 2, sharey='all', sharex='all', figsize=(10, 6))
        axes[0, 0].imshow(mask_crop_resized)
        axes[0, 1].imshow(building_crop_resized.astype(int))
        axes[1, 0].imshow(mask_crop_resized > 1.001)
        axes[1, 1].imshow(np.multiply(mask_crop_resized > 1.001, rgb2gray(building_crop_resized)), cmap='gray')
        plt.show()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex='all', sharey='all', figsize=(10, 6))
    ax1.imshow(mask_box, cmap='jet')
    ax2.imshow(image)
    ax3.imshow(bounding_box)
    plt.show()
