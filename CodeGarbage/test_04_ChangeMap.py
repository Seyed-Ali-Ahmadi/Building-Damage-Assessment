# from Functions import unsharp_mask, standardize, pad_features
# from Functions import read_jsondir, read_image_json
# from Functions import get_lbp, get_glcms
# import matplotlib.pyplot as plt
# from sklearn.svm import SVC
# import numpy as np
# import pickle
# import time
# import cv2
# from skimage.io import imread
#
#
# root = 'D:/00.University/data/data sets/BD/train/'
# imageDir = root + '1/'
# labelDir = root + '2/'
#
# # remove _pre _post postfixes and obtain unique images
# uniqueNames = sorted(list(read_jsondir(labelDir)), key=str.lower)
# # Indices are unique.
# print(uniqueNames)
#
# class_names = ['destroyed', 'major-damage', 'minor-damage', 'no-damage', 'un-classified']
# class_labels = [4, 3, 2, 1, -1]
#
# FeatureVector = np.empty((0, 17))
#
# for imgID, item in enumerate(uniqueNames):
#     print('1- Reading Data ...')
#     print('     Image ' + uniqueNames[imgID])
#     t0 = time.time()
#     # Images are gray-scale uint-8
#     _, preImage, postJson, postImage = read_image_json(imageDir, labelDir, item)
#     # Perform unsharp-masking
#     smoothPre = unsharp_mask(cv2.medianBlur(preImage, 3), kernel_size=(3, 3), sigma=1.5)
#     smoothPost = unsharp_mask(cv2.medianBlur(postImage, 3), kernel_size=(3, 3), sigma=1.5)
#     print('         ' + str(round(time.time() - t0, 2)))
#
#     preRGB = imread(imageDir + item + '_pre_disaster.png')
#     postRGB = imread(imageDir + item + '_post_disaster.png')
#
#     plt.figure()
#     plt.subplot(121), plt.imshow(postImage - preImage, cmap='gray')
#     plt.subplot(122), plt.imshow(postRGB - preRGB)
#     plt.show()
#
#
# -------------------------------------------------------------------------------
from DCVA import dcva
from skimage.io import imread
from sklearn.preprocessing import minmax_scale
import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.transform import resize

postImg = imread(r"D:\00.University\data\data sets\BD\tier3\images\joplin-tornado_00000027_post_disaster.png")
preImg = imread(r"D:\00.University\data\data sets\BD\tier3\images\joplin-tornado_00000027_pre_disaster.png")

from skimage.feature import hog

fd_pre, hog_pre = hog(resize(preImg[:90, 870:970, :], (100, 100, 3)), orientations=8, pixels_per_cell=(6, 6),
                      cells_per_block=(1, 1), visualize=True, multichannel=True, feature_vector=False)
fd_post, hog_post = hog(resize(postImg[:90, 870:970, :], (100, 100, 3)), orientations=8, pixels_per_cell=(6, 6),
                        cells_per_block=(1, 1), visualize=True, multichannel=True, feature_vector=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
ax1.axis('off')
hog_pre = exposure.rescale_intensity(hog_pre, in_range=(0, 10))
ax1.imshow(hog_pre, cmap='gray')
ax1.set_title('Input image')
ax2.axis('off')
hog_post = exposure.rescale_intensity(hog_post, in_range=(0, 10))
ax2.imshow(hog_post, cmap='gray')
ax2.set_title('Histogram of Oriented Gradients')
plt.show()

change = dcva(preImg, postImg, layers=[2], feature=True)

plt.figure()
plt.subplot(221), plt.imshow(preImg), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(postImg), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(np.abs(rgb2gray(postImg) - rgb2gray(preImg)), cmap='gray'), plt.xticks([]), plt.yticks([])
# plt.subplot(224), plt.imshow(1-change, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(np.multiply(np.abs(rgb2gray(postImg) - rgb2gray(preImg)), 1-change), cmap='gray'), plt.xticks([]), plt.yticks([])
plt.subplots_adjust(left=0, right=0.01, top=0.01, bottom=0, wspace=0)
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.show()

change = np.multiply(np.abs(rgb2gray(postImg) - rgb2gray(preImg)), 1 - change)
p2, p98 = np.percentile(change, (2, 98))
change = exposure.rescale_intensity(change, in_range=(p2, p98))

from skimage.morphology import closing

change = closing(change, selem=np.ones((5, 5)))
from scipy.ndimage import uniform_filter

change = uniform_filter(change, 15)

plt.figure()
plt.subplot(131), plt.imshow(preImg), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(postImg), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(change, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.subplots_adjust(left=0, right=0.01, top=0.01, bottom=0, wspace=0)
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.show()
