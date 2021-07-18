import os
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon
from skimage.measure import label


root = 'D:/00.University/data/data sets/BD/train/'
imageDir = root + '1/'
labelDir = root + '2/'

class_names = ['destroyed', 'major-damage', 'minor-damage', 'no-damage', 'un-classified']
# class_labels = [4, 3, 2, 1, -1]
class_labels = [4, 3, 2, 1, 1]
pad = 10

files = os.listdir(imageDir)

num_buildings = 0
num_false = 0
for item in os.listdir(imageDir):
    if item[-3:] == 'npy':
        classificationMap = np.load(imageDir + item)
        postJson = json.load(open(labelDir + item[:-4] + '_post_disaster.json', ))

        z1 = np.zeros_like(classificationMap, dtype=np.float)
        z2 = np.zeros_like(classificationMap, dtype=np.float)
        z3 = np.zeros_like(classificationMap, dtype=np.float)

        for bldID, building in enumerate(postJson['features']['xy']):
            num_buildings += 1
            damageType = building['properties']['subtype']
            # Extract x-y coordinate of each vertex from decoding the json pattern.
            vertices = building['wkt'].partition('POLYGON ((')[2].partition('))')[0].split(', ')
            n_vertices = len(vertices)

            rows = []
            cols = []
            for vertex in vertices:
                cols.append(float(vertex.split(' ')[0]) + pad)
                rows.append(float(vertex.split(' ')[1]) + pad)

            rr, cc = polygon(rows, cols, (1024 + 2 * pad, 1024 + 2 * pad))

            bldVals = []
            for pixel in range(len(rr)):
                bldVals.append(classificationMap[rr[pixel], cc[pixel]])

            percent = round(1 - sum(np.array(bldVals) == 1) / len(bldVals), 2)
            for pixel in range(len(rr)):
                z2[rr[pixel], cc[pixel]] = class_labels[class_names.index(damageType)]

                # If building is destroyed, major/minor damaged, put it in destroyed class.
                if class_labels[class_names.index(damageType)] > 1:
                    z3[rr[pixel], cc[pixel]] = 2
                else:
                    z3[rr[pixel], cc[pixel]] = 1

                # If more than 40% of pixels of a building are labeled as destroyed,
                # put the building in destroyed class.
                if percent > 0.4:
                    z1[rr[pixel], cc[pixel]] = 2
                else:
                    z1[rr[pixel], cc[pixel]] = 1

        num_false += len(np.unique(label(np.abs(z3 - z1)))) - 1

        plt.figure(figsize=(9, 5))
        plt.subplot(221), plt.imshow(classificationMap, cmap='jet'), plt.xticks([]), plt.yticks([])
        plt.subplot(222), plt.imshow(z1, cmap='jet'), plt.xticks([]), plt.yticks([])
        plt.subplot(223), plt.imshow(z2, cmap='jet'), plt.xticks([]), plt.yticks([])
        plt.subplot(224), plt.imshow(z3 - z1, cmap='jet'), plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.show()

print(num_buildings)
print(num_false)
print((1965 - 246) / 1965)



