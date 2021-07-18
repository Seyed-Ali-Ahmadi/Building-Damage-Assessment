# import the necessary packages
from skimage import feature
from skimage.io import imread
import matplotlib.pyplot as plt


root = 'D:/00.University/data/data sets/BD/train/'
imageDir = root + 'subset_images/'
img1 = imread(imageDir + 'hurricane-michael_00000187_post_disaster.png')
image = img1[:, :, 0]

# settings for LBP
radius = 1
n_points = 8 * radius
lbp = feature.local_binary_pattern(image, n_points, radius, method="default")
# Check method="var". It's interesting!

plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.subplot(122), plt.imshow(lbp, cmap='gray')
plt.show()

