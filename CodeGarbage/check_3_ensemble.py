root = 'D:/00.University/PhD Thesis Implementation/thesisEnv/training_0/'

folders = ['classified_images_LR_2000_4class_28features',
           'classified_images_RF_200_10_random_2class_6features',
           'classified_images_RF_200_10_random_2class_18features',
           'classified_images_RF_200_10_random_2class_22features',
           'classified_images_RF_200_10_random_4class_96Patch_features_4096',
           'classified_images_RF_200_10_random_4class_96Patch_features8060',
           'classified_images_RF_200_10_random_4class_3072Patch_Deepfeatures8060',
           'classified_images_RF_200_20_4class_28features',
           'classified_images_RF_400_20_4class_28features',
           'classified_images_Sequential_1',
           'classified_images_SVM_2000_4class_28features',
           'classified_RF_200_10_random_4class_maxDepth_featuresInceptionResNetV2_patch']

cls_images = ['hurricane-matthew_00000296_post_disaster.npy', 'hurricane-matthew_00000302_post_disaster.npy',
              'hurricane-michael_00000133_post_disaster.npy', 'hurricane-michael_00000247_post_disaster.npy',
              'hurricane-michael_00000299_post_disaster.npy', 'joplin-tornado_00000001_post_disaster.npy',
              'joplin-tornado_00000027_post_disaster.npy', 'joplin-tornado_00000031_post_disaster.npy',
              'joplin-tornado_00000083_post_disaster.npy', 'joplin-tornado_00000114_post_disaster.npy',
              'joplin-tornado_00000116_post_disaster.npy', 'joplin-tornado_00000118_post_disaster.npy',
              'joplin-tornado_00000120_post_disaster.npy', 'moore-tornado_00000035_post_disaster.npy',
              'moore-tornado_00000051_post_disaster.npy', 'moore-tornado_00000056_post_disaster.npy',
              'moore-tornado_00000104_post_disaster.npy', 'moore-tornado_00000110_post_disaster.npy',
              'moore-tornado_00000122_post_disaster.npy', 'moore-tornado_00000140_post_disaster.npy']

import os
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt

for file in cls_images:
    image = np.empty((1024, 1024, 1))
    for folder in folders:
        if file in os.listdir(root + folder):
            img = np.load(root + folder + '/' + file)
            image = np.dstack((image, img))

    image = image[:, :, 1:]
    classified = np.median(image, axis=2)
    np.save(root + 'ensemble_1' + '/' + file, classified)
    # plt.figure()
    # plt.imshow(classified)
    # plt.show()
