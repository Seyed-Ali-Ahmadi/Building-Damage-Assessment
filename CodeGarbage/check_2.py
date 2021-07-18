import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon
import pandas as pd
from skimage.io import imread
from sklearn.preprocessing import minmax_scale
from DCVA import dcva
from skimage.color import rgb2gray
from skimage.morphology import closing, dilation
from skimage import exposure
from scipy.ndimage import uniform_filter
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
print(labelsDB_post_gt0_disaster)

df = labelsDB_post_gt0_disaster[['destroyed#', 'minor-damage#', 'major-damage#', 'no-damage#']]
df = pd.concat((labelsDB_post_gt0_disaster, df.prod(axis=1)), axis=1).sort_values(by=0, ascending=False).drop(
    columns=0).head(20)
classes = ['no-damage', 'minor-damage', 'major-damage', 'destroyed', 'un-classified']
info = []
# for i in range(len(df['img_name'])):
#     name = (df['img_name'].iloc[i].split('/')[-1] + '.npy')
#     classified = np.load('./training_0/classified_images_RF_200_10_random_2class_22features/' + name)
#     Labels = pd.read_json(open(df['img_name'].iloc[i] + '.json', ))['features']['xy']
#
#     prefile = df['img_name'].iloc[i].replace('labels', 'images').replace('post', 'pre')
#     posfile = df['img_name'].iloc[i].replace('labels', 'images')
#     preimg = imread(prefile + '.png')
#     posimg = imread(posfile + '.png')
#
#     change = dcva(preimg, posimg, layers=[2], feature=True)
#     change = np.multiply(np.abs(rgb2gray(posimg) - rgb2gray(preimg)), 1 - change)
#     p2, p98 = np.percentile(change, (2, 98))
#     change = exposure.rescale_intensity(change, in_range=(p2, p98))
#     # change = closing(change, selem=np.ones((5, 5)))
#     change = uniform_filter(change, 5)
#     change = minmax_scale(change.reshape((1024*1024, 1))).reshape((1024, 1024))
#
#     build_loc = create_mask(Labels, wclassification=False)
#     build_cls = create_mask(Labels, wclassification=True)
#
#     for idx in np.unique(build_loc)[1:]:
#         building_pixels = sum(sum(build_loc == idx))
#         damaged_pixels = sum(sum(np.multiply((build_loc == idx), classified) == 2))
#         building_class = np.unique(np.multiply((build_loc == idx), build_cls))[1]
#
#         building = np.zeros((1024, 1024))
#         building[build_loc == idx] = 1
#         building = dilation(building, selem=np.ones((20, 20)))
#         weight = np.multiply(building, change)
#
#         surr = np.multiply(weight, np.invert(build_loc == idx))
#         buil = np.multiply(weight, build_loc == idx)
#         mu_surr = np.mean(surr[surr != 0])
#         mu_buil = np.mean(buil[buil != 0])
#
#         info.append([mu_buil, mu_surr, damaged_pixels / building_pixels, building_class])
#         # plt.imshow(np.multiply(weight, np.invert(build_loc == idx))), plt.show()
#
# info = np.array(info)
# np.save('building_surrounding_information2.npy', info)

Sym = ['s', 'o', '^', 'p', '*', 'x']
Col = ["pink", "forestgreen", "gold", "darkorange", "red", "white"]
info = np.load('building_surrounding_information2.npy')
symbols = []
colors = []
for i in range(info.shape[0]):
    colors.append(Col[int(info[i, -1])])
    symbols.append(Sym[int(info[i, -1])])

plt.figure()
plt.style.use('dark_background')
plt.scatter(info[:, 0], info[:, 1], s=info[:, 2]*60, c=colors)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.grid(True, linestyle='--', color='grey', alpha=.25)
plt.tight_layout()
plt.show()


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(info[:, 0], info[:, 1], info[:, 2], c=colors)
plt.show()


from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=3, perplexity=100).fit_transform(info[:, :3])
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2],
           s=info[:, 2] * 50, c=colors)
plt.subplots_adjust(left=0, right=0.01, top=0.01, bottom=0, wspace=0)
plt.tight_layout()
plt.show()
