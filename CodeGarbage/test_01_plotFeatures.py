from Functions import *
from skimage import feature
import matplotlib.pyplot as plt
from skimage.draw import polygon

root = 'D:/00.University/data/data sets/BD/train/'
imageDir = root + '1/'
labelDir = root + '2/'
jsonFiles = os.listdir(labelDir)

# remove _pre _post postfixes and obtain unique images
uniqueNames = read_jsondir(labelDir)
# --------------------------------------------------------------------------
for item in uniqueNames:
    _, preImage, postJson, postImage = read_image_json(imageDir, labelDir, item)
# --------------------------- Extract features -----------------------------
    preLBP = get_lbp(preImage.astype(float)/255)
    postLBP = get_lbp(postImage.astype(float)/255)
# --------------------------------------------------------------------------
#     preEdge = feature.canny(preImage.astype(float)/255, sigma=0.5).astype(int)
#     postEdge = feature.canny(postImage.astype(float)/255, sigma=0.5).astype(int)
# --------------------------------------------------------------------------
    preGLCM = get_glcms(preImage)
    postGLCM = get_glcms(postImage)
# --------------------------------------------------------------------------
    # Create a new building mask.
    pad = 10
    building_mask = np.zeros((1024 + 2 * pad, 1024 + 2 * pad))
    bounding_box = np.zeros((1024 + 2 * pad, 1024 + 2 * pad))

    preImage, postImage = make_pad(preImage, postImage)
    preLBP, postLBP = make_pad(preLBP, postLBP)
    # preLBP, postLBP = make_pad(preEdge, postEdge)
    preMean, postMean = make_pad(preGLCM[0], postGLCM[0])
    preSTD, postSTD = make_pad(preGLCM[1], postGLCM[1])
    preCont, postCont = make_pad(preGLCM[2], postGLCM[2])
    preDiss, postDiss = make_pad(preGLCM[3], postGLCM[3])
    preHomo, postHomo = make_pad(preGLCM[4], postGLCM[4])
    preAsm, postAsm = make_pad(preGLCM[5], postGLCM[5])
    preEne, postEne = make_pad(preGLCM[6], postGLCM[6])
    preMa, postMa = make_pad(preGLCM[7], postGLCM[7])
    preEnt, postEnt = make_pad(preGLCM[8], postGLCM[8])

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
        preLBPCrop = get_patch(preLBP, rows, cols)
        postLBPCrop = get_patch(postLBP, rows, cols)
        preMeanCrop = get_patch(preMean, rows, cols)
        postMeanCrop = get_patch(postMean, rows, cols)
        preSTDCrop = get_patch(preSTD, rows, cols)
        postSTDCrop = get_patch(postSTD, rows, cols)
        preContCrop = get_patch(preCont, rows, cols)
        postContCrop = get_patch(postCont, rows, cols)
        preDissCrop = get_patch(preDiss, rows, cols)
        postDissCrop = get_patch(postDiss, rows, cols)
        preHomoCrop = get_patch(preHomo, rows, cols)
        postHomoCrop = get_patch(postHomo, rows, cols)
        preAsmCrop = get_patch(preAsm, rows, cols)
        postAsmCrop = get_patch(postAsm, rows, cols)
        preEneCrop = get_patch(preEne, rows, cols)
        postEneCrop = get_patch(postEne, rows, cols)
        preMaCrop = get_patch(preMa, rows, cols)
        postMaCrop = get_patch(postMa, rows, cols)
        preEntCrop = get_patch(preEnt, rows, cols)
        postEntCrop = get_patch(postEnt, rows, cols)

        mask_crop = mask_box[int(np.floor(min(rows))) - pad:
                             int(np.ceil(max(rows))) + pad,
                             int(np.floor(min(cols))) - pad:
                             int(np.ceil(max(cols))) + pad]

        # mask_crop_resized = resize(mask_crop, (50, 50), anti_aliasing=True, preserve_range=True)
        # building_crop_resized = resize(image_crop, (50, 50), anti_aliasing=True, preserve_range=True)

# --------------------------------------------------------------------------
        fig, axes = plt.subplots(3, 5, sharex='all', sharey='all', figsize=(11.5, 7))
        axes[0, 0].imshow(mask_crop, cmap='gray'), axes[0, 0].set_title('mask')
        axes[0, 1].imshow(preImageCrop, cmap='gray'), axes[0, 1].set_title('pre image')
        axes[0, 2].imshow(np.abs(preImageCrop - postImageCrop), cmap='gray'), axes[0, 2].set_title('|pre - post|')
        axes[0, 3].imshow(preImageCrop.astype(float)/255 - postImageCrop.astype(float)/255,
                          cmap='gray'), axes[0, 3].set_title('pre - post (float)')
        axes[0, 4].imshow(postLBPCrop - preLBPCrop, cmap='gray'), axes[0, 4].set_title('pre LBP')
        axes[1, 0].imshow(np.multiply(mask_crop > 1, preImageCrop), cmap='gray'), axes[1, 0].set_title('masked pre')
        axes[1, 1].imshow(postMeanCrop - preMeanCrop, cmap='gray'), axes[1, 1].set_title('pre Mean')
        axes[1, 2].imshow(postSTDCrop - preSTDCrop, cmap='gray'), axes[1, 2].set_title('pre STD')
        axes[1, 3].imshow(postContCrop - preContCrop, cmap='gray'), axes[1, 3].set_title('pre Cont')
        axes[1, 4].imshow(postDissCrop - preDissCrop, cmap='gray'), axes[1, 4].set_title('pre Diss')
        axes[2, 0].imshow(postHomoCrop - preHomoCrop, cmap='gray'), axes[2, 0].set_title('pre Homo')
        axes[2, 1].imshow(postAsmCrop - preAsmCrop, cmap='gray'), axes[2, 1].set_title('pre Asm')
        axes[2, 2].imshow(postEneCrop - preEneCrop, cmap='gray'), axes[2, 2].set_title('pre Ene')
        axes[2, 3].imshow(postMaCrop - preMaCrop, cmap='gray'), axes[2, 3].set_title('pre Ma')
        axes[2, 4].imshow(postEntCrop - preEntCrop, cmap='gray'), axes[2, 4].set_title('pre Ent')
        plt.suptitle('Pre disaster image of buildings.')

        fig, axes = plt.subplots(3, 5, sharex='all', sharey='all', figsize=(11.5, 7))
        axes[0, 0].imshow(mask_crop, cmap='gray'), axes[0, 0].set_title('mask')
        axes[0, 1].imshow(postImageCrop, cmap='gray'), axes[0, 1].set_title('post image')
        axes[0, 2].imshow(np.abs(postImageCrop - preImageCrop), cmap='gray'), axes[0, 2].set_title('|post - pre|')
        axes[0, 3].imshow(postImageCrop.astype(float)/255 - preImageCrop.astype(float)/255,
                          cmap='gray'), axes[0, 3].set_title('post - pre (float)')
        axes[0, 4].imshow(postLBPCrop, cmap='gray'), axes[0, 4].set_title('post LBP')
        axes[1, 0].imshow(np.multiply(mask_crop > 1, postImageCrop), cmap='gray'), axes[1, 0].set_title('masked post')
        axes[1, 1].imshow(postMeanCrop, cmap='gray'), axes[1, 1].set_title('post Mean')
        axes[1, 2].imshow(postSTDCrop, cmap='gray'), axes[1, 2].set_title('post STD')
        axes[1, 3].imshow(postContCrop, cmap='gray'), axes[1, 3].set_title('post Cont')
        axes[1, 4].imshow(postDissCrop, cmap='gray'), axes[1, 4].set_title('post Diss')
        axes[2, 0].imshow(postHomoCrop, cmap='gray'), axes[2, 0].set_title('post Homo')
        axes[2, 1].imshow(postAsmCrop, cmap='gray'), axes[2, 1].set_title('post Asm')
        axes[2, 2].imshow(postEneCrop, cmap='gray'), axes[2, 2].set_title('post Ene')
        axes[2, 3].imshow(postMaCrop, cmap='gray'), axes[2, 3].set_title('post Ma')
        axes[2, 4].imshow(postEntCrop, cmap='gray'), axes[2, 4].set_title('post Ent')
        plt.suptitle(building['properties']['subtype'])

        plt.show()


