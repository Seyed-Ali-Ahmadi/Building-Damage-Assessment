from Functions import *
from skimage import feature
import matplotlib.pyplot as plt
from skimage.draw import polygon
import cv2

root = 'D:/00.University/data/data sets/BD/train/'
imageDir = root + 'subset_images/'
labelDir = root + 'subset_labels/'
jsonFiles = os.listdir(labelDir)

# remove _pre _post postfixes and obtain unique images
jsonFiles = read_jsondir(labelDir)

for item in jsonFiles:
    preJson, preImage, postJson, postImage = read_image_json(imageDir, labelDir, item)

    pad = 10
    building_mask = np.zeros((1024 + 2 * pad, 1024 + 2 * pad))
    bounding_box = np.zeros((1024 + 2 * pad, 1024 + 2 * pad))

    preImage, postImage = make_pad(preImage, postImage)
    # Loop through buildings in each image.
    for building in postJson['features']['xy']:
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

        # Obtain the image of each building and its sorrounding.
        preImageCrop = get_patch(preImage, rows, cols)
        postImageCrop = get_patch(postImage, rows, cols)
        # plt.figure(), plt.imshow(preImageCrop), plt.show()

        sift = cv2.SIFT_create(nfeatures=10)
        kp1 = sift.detect(preImageCrop, None)
        kp2 = sift.detect(postImageCrop, None)
        img_1 = cv2.drawKeypoints(preImageCrop, kp1, preImageCrop,
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img_2 = cv2.drawKeypoints(postImageCrop, kp2, postImageCrop,
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.figure()
        plt.subplot(121), plt.imshow(img_1)
        plt.subplot(122), plt.imshow(img_2)
        plt.show()



