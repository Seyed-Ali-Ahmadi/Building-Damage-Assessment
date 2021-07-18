# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from skimage.io import imread
# from skimage.draw import polygon
# import pandas as pd
# from sklearn.metrics import confusion_matrix, classification_report
# from skimage.util import img_as_float
#
# disaster_type = 'wind'
# labelsDB = pd.read_pickle('All_Data_Props.pkl')
# labelsDB = labelsDB[labelsDB['Group'] != 'Test']
# labelsDB_post = labelsDB[labelsDB['Pre_Post'] == 'post']
# labelsDB_post_gt0 = labelsDB_post[labelsDB_post['buildings#'] > 0]
# labelsDB_post_gt0_disaster = labelsDB_post_gt0[labelsDB_post_gt0['disaster_type'] == disaster_type].reset_index()
# print(labelsDB_post_gt0_disaster.shape, ' size of database of post disaster images which have buildings.')
# print(labelsDB_post_gt0_disaster)
#
# df = labelsDB_post_gt0_disaster[['destroyed#', 'minor-damage#', 'major-damage#', 'no-damage#']]
# df = pd.concat((labelsDB_post_gt0_disaster, df.prod(axis=1)), axis=1).sort_values(by=0, ascending=False).drop(
#     columns=0).head(20)
#
# classes = ['no-damage', 'minor-damage', 'major-damage', 'destroyed', 'un-classified']
#
# y_true = np.empty((0, 1))
# y_pred = np.empty((0, 1))
#
# for i in range(len(df['img_name'])):
#     name = (df['img_name'].iloc[i].split('/')[-1] + '.npy')
#     classified = np.load('./training_0/classified_images_SVM_2000/' + name)
#
#     post = imread(df['img_name'].iloc[i].replace('labels', 'images') + '.png')
#     pre = imread(df['img_name'].iloc[i].replace('labels', 'images').replace('post', 'pre') + '.png')
#     labels = pd.read_json(open(df['img_name'].iloc[i] + '.json', ))['features']['xy']
#     mask = np.zeros((1024, 1024))
#     classified[classified == 3] = 2
#     for building in labels:
#         Class = classes.index(building['properties']['subtype']) + 1
#         vertices = building['wkt'].partition('POLYGON ((')[2].partition('))')[0].split(', ')
#         rows = []
#         cols = []
#         for vertex in vertices:
#             cols.append(float(vertex.split(' ')[0]))
#             rows.append(float(vertex.split(' ')[1]))
#         rr, cc = polygon(rows, cols, (1024, 1024))
#         pixels = list(zip(rr, cc))
#         for rr, cc in pixels:
#             mask[rr, cc] = Class
#
#
#     plt.figure()
#     plt.subplot(221), plt.imshow(classified), plt.xticks([]), plt.yticks([])
#     plt.subplot(222), plt.imshow(mask, vmin=0, vmax=4), plt.xticks([]), plt.yticks([])
#     mask[mask > 0] = 1
#     mask = np.dstack((mask, mask, mask))
#     plt.subplot(223), plt.imshow(np.multiply(img_as_float(pre), mask)), plt.xticks([]), plt.yticks([])
#     plt.subplot(224), plt.imshow(np.multiply(img_as_float(post), mask)), plt.xticks([]), plt.yticks([])
#     plt.subplots_adjust(left=0, right=0.01, top=0.01, bottom=0, wspace=0)
#     plt.tight_layout(pad=0, w_pad=0, h_pad=0)
#     plt.show()
#
#     y_pred = np.vstack((y_pred, np.atleast_2d(classified[classified > 0]).T))
#     y_true = np.vstack((y_true, np.atleast_2d(mask[mask > 0]).T))
#     # cm = confusion_matrix(y_true, y_pred)
#     # plt.figure()
#     # plt.imshow(cm, cmap='Blues')
#     # plt.xticks(ticks=[0, 1, 2, 3], labels=['ND', 'MiD', 'MaD', 'D'])
#     # plt.yticks(ticks=[0, 1, 2, 3], labels=['ND', 'MiD', 'MaD', 'D'])
#     # plt.show()
#
# cm = confusion_matrix(y_true, y_pred)
# print(cm)
# print(classification_report(y_true, y_pred))
# plt.figure()
# plt.imshow(cm[:-1, :-1], cmap='Blues')
# plt.ylabel('True label')
# plt.xlabel('Classified label')
# plt.xticks(ticks=[0, 1, 2, 3], labels=['ND', 'MiD', 'MaD', 'D'])
# plt.yticks(ticks=[0, 1, 2, 3], labels=['ND', 'MiD', 'MaD', 'D'])
# plt.show()

# ----------------------------------------------------------------------------------- check with post-processing applied
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from skimage.io import imread
from matplotlib.colors import ListedColormap
cmap = ListedColormap(["pink", "forestgreen", "gold", "darkorange", "red", "black"])


def create_mask(labels, wclassification=True):
    mask = np.zeros((1024, 1024))
    classes = ['no-damage', 'minor-damage', 'major-damage', 'destroyed', 'un-classified']
    numOfBuildings = 0
    for building in labels:
        vertices = building['wkt'].partition('POLYGON ((')[2].partition('))')[0].split(', ')
        rows = []
        cols = []
        for vertex in vertices:
            cols.append(float(vertex.split(' ')[0]))
            rows.append(float(vertex.split(' ')[1]))
        rr, cc = polygon(rows, cols, (1024, 1024))
        pixels = list(zip(rr, cc))
        Class = classes.index(building['properties']['subtype']) + 1
        for rr, cc in pixels:
            if wclassification:
                mask[rr, cc] = Class
            else:
                mask[rr, cc] = numOfBuildings
        numOfBuildings += 1
    return mask


disaster_type = 'wind'
labelsDB = pd.read_pickle('All_Data_Props.pkl')
labelsDB = labelsDB[labelsDB['Group'] != 'Test']
labelsDB_post = labelsDB[labelsDB['Pre_Post'] == 'post']
labelsDB_post_gt0 = labelsDB_post[labelsDB_post['buildings#'] > 0]
labelsDB_post_gt0_disaster = labelsDB_post_gt0[labelsDB_post_gt0['disaster_type'] == disaster_type].reset_index()
print(labelsDB_post_gt0_disaster.shape, ' size of database of post disaster images which have buildings.')

df = labelsDB_post_gt0_disaster[['destroyed#', 'minor-damage#', 'major-damage#', 'no-damage#']]
df = pd.concat((labelsDB_post_gt0_disaster, df.prod(axis=1)), axis=1).sort_values(by=0, ascending=False).drop(
    columns=0).head(20)  # 60
classes = ['no-damage', 'minor-damage', 'major-damage', 'destroyed', 'un-classified']

areaPercentage = {'no-damage': [], 'minor-damage': [],
                  'major-damage': [], 'destroyed': []}

y_true = []
y_pred = []
for i in range(len(df['img_name'])):
    name = (df['img_name'].iloc[i].split('/')[-1] + '.npy')
    # classified = np.load('./training_0/classified_images_1st_place/' + name)
    classified = np.load('./training_0/classified_images_LGBM_4class_featuresInceptionResNetV2_patch/' + name)
    Labels = pd.read_json(open(df['img_name'].iloc[i] + '.json', ))['features']['xy']

    preimg = imread(df['img_name'].iloc[i].replace('labels', 'images').replace('post', 'pre') + '.png')
    postimg = imread(df['img_name'].iloc[i].replace('labels', 'images') + '.png')

    build_loc = create_mask(Labels, wclassification=False)
    # build_loc[build_loc > 0] = 1
    # build_loc = np.dstack((build_loc, build_loc, build_loc))
    build_cls = create_mask(Labels, wclassification=True)

    # classified_ = np.zeros((1024, 1024))
    for idx in np.unique(build_loc)[1:]:
        #     building_pixels = sum(sum(build_loc == idx))
        #     damaged_pixels = sum(sum(np.multiply((build_loc == idx), classified) == 2))
        #     building_class = np.unique(np.multiply((build_loc == idx), build_cls))[1]
        #     if building_class != 5:
        #         areaPercentage[classes[int(building_class)-1]].append(damaged_pixels / building_pixels)
        #
        #     per = damaged_pixels / building_pixels
        #     if per < 0.3:
        #         classified_[build_loc == idx] = 1
        #     elif (per > 0.3) and (per < 0.48):
        #         classified_[build_loc == idx] = 2
        #     elif (per > 0.48) and (per < 0.66):
        #         classified_[build_loc == idx] = 3
        #     elif (per > 0.66) and (per < 1.00):
        #         classified_[build_loc == idx] = 4
        #
        # y_pred.extend(np.unique(classified[build_loc == idx]))  # Building-wise accuracy assessment
        y_pred.extend(np.array([np.median(classified[build_loc == idx])]))  # Building-wise accuracy assessment
        y_true.extend(np.unique(build_cls[build_loc == idx]))

    # y_pred.extend(classified_[np.multiply(build_cls != 0, build_cls != 5)])
    # y_pred.extend(classified[np.multiply(build_cls != 0, build_cls != 5)])  # Pixel-wise accuracy assessment
    # y_true.extend(build_cls[np.multiply(build_cls != 0, build_cls != 5)])

    # plt.figure()
    # plt.subplot(131), plt.imshow(classified, cmap='jet'), plt.xticks([]), plt.yticks([])
    # plt.subplot(132), plt.imshow(build_cls, cmap=cmap, vmax=5, vmin=0), plt.xticks([]), plt.yticks([])
    # plt.subplot(133), plt.imshow(classified_, cmap=cmap, vmax=5, vmin=0), plt.xticks([]), plt.yticks([])
    # plt.subplots_adjust(left=0, right=0.01, top=0.01, bottom=0, wspace=0)
    # plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    # plt.show()

    # plt.figure()
    # plt.suptitle(df['img_name'].iloc[i])
    # plt.subplot(221), plt.imshow(classified, cmap=cmap, vmax=5, vmin=0), plt.xticks([]), plt.yticks([])
    # plt.subplot(222), plt.imshow(build_cls, cmap=cmap, vmax=5, vmin=0), plt.xticks([]), plt.yticks([])
    # plt.subplot(223), plt.imshow(np.multiply(preimg[:, :, 0], build_loc).astype(int)), plt.xticks([]), plt.yticks([])
    # plt.subplot(224), plt.imshow(np.multiply(postimg[:, :, 0], build_loc).astype(int)), plt.xticks([]), plt.yticks([])
    # plt.subplots_adjust(left=0, right=0.01, top=0.01, bottom=0, wspace=0)
    # plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    # plt.show()

y_pred = np.array(y_pred).astype(int)
y_true = np.array(y_true).astype(int)

y_true = y_true[y_pred != 0]
y_pred = y_pred[y_pred != 0]

y_pred = y_pred[y_true != 5]
y_true = y_true[y_true != 5]

# y_pred[y_pred == 2] = 1
# y_true[y_true == 2] = 1
# y_pred[y_pred == 3] = 4
# y_true[y_true == 3] = 4

cm = confusion_matrix(y_true.astype(int), y_pred.astype(int))
print(cm)
print(classification_report(y_true.astype(int), y_pred.astype(int)))
plt.figure()
plt.imshow(cm, cmap='Blues')
plt.ylabel('True label')
plt.xlabel('Classified label')
plt.xticks(ticks=[0, 1, 2, 3], labels=['ND', 'MiD', 'MaD', 'D'])
plt.yticks(ticks=[0, 1, 2, 3], labels=['ND', 'MiD', 'MaD', 'D'])
plt.show()

plt.figure()
plt.plot(np.random.uniform(low=0.9, high=1.1, size=(len(areaPercentage['no-damage']),)),
         areaPercentage['no-damage'], mfc='navy', marker='o', alpha=0.08)
plt.plot(np.random.uniform(low=1.9, high=2.1, size=(len(areaPercentage['minor-damage']),)),
         areaPercentage['minor-damage'], mfc='darkcyan', marker='*', alpha=0.08)
plt.plot(np.random.uniform(low=2.9, high=3.1, size=(len(areaPercentage['major-damage']),)),
         areaPercentage['major-damage'], mfc='deeppink', marker='^', alpha=0.08)
plt.plot(np.random.uniform(low=3.9, high=4.1, size=(len(areaPercentage['destroyed']),)),
         areaPercentage['destroyed'], mfc='orangered', marker='p', alpha=0.08)

plt.boxplot([areaPercentage['no-damage'], areaPercentage['minor-damage'],
             areaPercentage['major-damage'], areaPercentage['destroyed']],
            labels=['No-Damage', 'Minor-Damage', 'Major-Damage', 'Destroyed'],
            showfliers=True, notch=True, widths=0.2, showmeans=True)

plt.hlines(0.00, 0.9, 1.1, colors='gold')
plt.hlines(0.30, 0.9, 1.1, colors='gold')
plt.hlines(0.30, 1.9, 2.1, colors='gold')
plt.hlines(0.48, 1.9, 2.1, colors='gold')
plt.hlines(0.48, 2.9, 3.1, colors='gold')
plt.hlines(0.66, 2.9, 3.1, colors='gold')
plt.hlines(0.66, 3.9, 4.1, colors='gold')
plt.hlines(1.00, 3.9, 4.1, colors='gold')

plt.ylabel('Percent of damaged pixels in each building.')
plt.title('Percentage distribution of damaged pixels per building,\npredicted by classifier per damage class.')
plt.grid(True, linestyle='--', color='grey', alpha=.25)
plt.tight_layout()
plt.show()

