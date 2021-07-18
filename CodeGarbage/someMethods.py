import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.draw import polygon
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.transform import resize
from skimage.util import img_as_ubyte
from skimage.measure import label, regionprops
from Functions import get_glcms
from skimage import exposure
import pandas as pd
from Functions import get_lbp
from skimage.feature import greycomatrix, greycoprops
from keras.layers import MaxPool2D
from keras.models import Sequential


def max_kernel(patch):
    arr = patch.copy()
    arr = arr.reshape(1, 24, 24, 1)
    out = np.squeeze(Sequential([MaxPool2D(pool_size=3, strides=3)]).predict(arr))
    arr = out.reshape(1, 8, 8, 1)
    out = np.squeeze(Sequential([MaxPool2D(pool_size=2, strides=2)]).predict(arr))
    plt.figure()
    plt.subplot(121), plt.imshow(patch)
    plt.subplot(122), plt.imshow(out)
    plt.show()


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
print(labelsDB_post_gt0_disaster)

df = labelsDB_post_gt0_disaster[['destroyed#', 'minor-damage#', 'major-damage#', 'no-damage#']]
df = pd.concat((labelsDB_post_gt0_disaster, df.prod(axis=1)), axis=1).sort_values(by=0, ascending=False).drop(
    columns=0).head(20)
classes = ['no-damage', 'minor-damage', 'major-damage', 'destroyed', 'un-classified']
angles = [0, np.pi / 4, np.pi / 2]
distances = [1, 2, 3, 4, 5]

for i in range(len(df['img_name'])):
    name = (df['img_name'].iloc[i].split('/')[-1] + '.npy')
    classified = np.load('./training_0/classified_images_RF_200_10_random_2class_6features/' + name)
    Labels = pd.read_json(open(df['img_name'].iloc[i] + '.json', ))['features']['xy']

    preimg = imread(df['img_name'].iloc[i].replace('labels', 'images').replace('post', 'pre') + '.png')
    postimg = imread(df['img_name'].iloc[i].replace('labels', 'images') + '.png')

    build_loc = create_mask(Labels, wclassification=False).astype(int)
    build_loc_bbox = np.zeros_like(build_loc)
    LBP_pre = get_lbp(rgb2gray(preimg), method='uniform')
    LBP_pos = get_lbp(rgb2gray(postimg), method='uniform')
    LBP_dif = get_lbp(rgb2gray(postimg - preimg), method='uniform')
    dif_LBP = LBP_pos - LBP_pre

    count = 1
    for region in regionprops(build_loc):
        minr, minc, maxr, maxc = region.bbox
        w = 24
        pad = 5
        try:
            patch_pre = resize(preimg[minr - pad:maxr + pad, minc - pad:maxc + pad, :], (w, w, 3))
            patch_pre_lbp = resize(LBP_pre[minr - pad:maxr + pad, minc - pad:maxc + pad], (w, w))
            patch_pos = resize(postimg[minr - pad:maxr + pad, minc - pad:maxc + pad, :], (w, w, 3))
            patch_pos_lbp = resize(LBP_pos[minr - pad:maxr + pad, minc - pad:maxc + pad], (w, w))
            patch_lbp_dif = resize(LBP_dif[minr - pad:maxr + pad, minc - pad:maxc + pad], (w, w))
            patch_dif_lbp = resize(dif_LBP[minr - pad:maxr + pad, minc - pad:maxc + pad], (w, w))

        except ValueError:
            patch_pre = resize(preimg[minr:maxr, minc:maxc, :], (w, w, 3))
            patch_pre_lbp = resize(LBP_pre[minr:maxr, minc:maxc], (w, w))
            patch_pos = resize(postimg[minr:maxr, minc:maxc, :], (w, w, 3))
            patch_pos_lbp = resize(LBP_pos[minr:maxr, minc:maxc], (w, w))
            patch_lbp_dif = resize(LBP_dif[minr:maxr, minc:maxc], (w, w))
            patch_dif_lbp = resize(dif_LBP[minr:maxr, minc:maxc], (w, w))

        pre_hog_feat, pre_hog_vis = hog(patch_pre, orientations=8, pixels_per_cell=(6, 6), cells_per_block=(1, 1),
                                        visualize=True, multichannel=True)
        pre_hog_vis = exposure.rescale_intensity(pre_hog_vis, in_range=(0, 10))
        post_hog_feat, post_hog_vis = hog(patch_pos, orientations=8, pixels_per_cell=(6, 6), cells_per_block=(1, 1),
                                          visualize=True, multichannel=True)
        post_hog_vis = exposure.rescale_intensity(post_hog_vis, in_range=(0, 10))

        hog_dif_feat, hog_dif_vis = hog(patch_pos - patch_pre, orientations=8, pixels_per_cell=(6, 6),
                                        cells_per_block=(1, 1), visualize=True, multichannel=True)
        hog_dif_vis = exposure.rescale_intensity(hog_dif_vis, in_range=(0, 10))
        dif_hog_vis = exposure.rescale_intensity(post_hog_vis - pre_hog_vis, in_range=(0, 10))
        dif_hog_feat = post_hog_feat - pre_hog_feat

        fig, ax = plt.subplots(2, 4, figsize=(12, 6))
        ax[0, 0].axis('off'), ax[0, 0].imshow(pre_hog_vis), ax[0, 0].set_title('HOG pre')
        ax[0, 1].hist(pre_hog_feat.flatten(), bins=15), ax[0, 1].set_title('HOG pre features')
        ax[0, 2].axis('off'), ax[0, 2].imshow(post_hog_vis), ax[0, 2].set_title('HOG post')
        ax[0, 3].hist(post_hog_feat.flatten(), bins=15), ax[0, 3].set_title('HOG post features')
        # -----------------------------------------------------
        ax[1, 0].axis('off'), ax[1, 0].imshow(hog_dif_vis), ax[1, 0].set_title('HOG post - pre')
        ax[1, 1].hist(hog_dif_feat.flatten(), bins=15), ax[1, 1].set_title('HOG post - pre features')
        ax[1, 2].axis('off'), ax[1, 2].imshow(dif_hog_vis), ax[1, 2].set_title('HOG post - HOG pre')
        ax[1, 3].hist(dif_hog_feat.flatten(), bins=15), ax[1, 3].set_title('HOG post - HOG pre features')
        plt.subplots_adjust(left=0, right=0.01, top=0.01, bottom=0, wspace=0)
        plt.tight_layout(pad=0, w_pad=0, h_pad=0)

        GLCM_dif = greycomatrix(img_as_ubyte(np.mean(patch_pos - patch_pre, axis=2)),
                                distances=distances, angles=angles, levels=256, symmetric=True)
        preGLCM = greycomatrix(img_as_ubyte(np.mean(patch_pre, axis=2)), distances=distances, angles=angles, levels=256,
                               symmetric=True)
        posGLCM = greycomatrix(img_as_ubyte(np.mean(patch_pos, axis=2)), distances=distances, angles=angles, levels=256,
                               symmetric=True)
        dif_GLCM = posGLCM - preGLCM

        fig, ax = plt.subplots(2, 5, figsize=(12, 6))
        ax[0, 0].axis('off'), ax[0, 0].imshow(patch_pre), ax[0, 0].set_title('Pre image')
        ax[0, 1].axis('off'), ax[0, 1].imshow(patch_lbp_dif, cmap='gray'), ax[0, 1].set_title('LBP of Difference')
        ax[0, 2].hist(patch_lbp_dif.ravel(), bins=20), ax[0, 2].set_title('Histogram of LBP of Difference')
        ax[0, 3].bar(range(15), greycoprops(GLCM_dif, 'homogeneity').ravel()), ax[0, 3].set_title('GLCM of Difference')
        diff = exposure.rescale_intensity(patch_pos - patch_pre, in_range=(0, 1))
        ax[0, 4].axis('off'), ax[0, 4].imshow(diff), ax[0, 4].set_title('Difference Image')
        # -----------------------------------------------------
        ax[1, 0].axis('off'), ax[1, 0].imshow(patch_pos), ax[1, 0].set_title('Post image')
        ax[1, 1].axis('off'), ax[1, 1].imshow(patch_dif_lbp, cmap='gray'), ax[1, 1].set_title('Difference of LBP')
        ax[1, 2].hist(patch_dif_lbp.ravel(), bins=20), ax[1, 2].set_title('Histogram of Difference of LBP')
        ax[1, 3].bar(range(15), greycoprops(dif_GLCM, 'homogeneity').ravel()), ax[1, 3].set_title('Difference of GLCM')
        ax[1, 4].hist(diff.reshape((w * w, 3)), bins=10, color=['brown', 'navy', 'darkcyan']), ax[1, 4].set_title('Histogram of Difference')
        plt.subplots_adjust(left=0, right=0.01, top=0.01, bottom=0, wspace=0)
        plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        plt.show()

        max_kernel(rgb2gray(patch_pos - patch_pre))

