"""

"""
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import exposure
import numpy as np

import pickle
database = pickle.load(open("databaseDictionary.pkl", "rb"))
disaster_type = 'mixed'
damage_class = 'major-damage'


model = Sequential()
model.add(Dense(12, input_dim=3, kernel_initializer='normal', activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='linear'))
model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])


path1 = 'D:/00.University/PhD Thesis Implementation/thesisEnv/dataset/train/images/hurricane-florence_00000001_pre_disaster.png'
path2 = 'D:/00.University/PhD Thesis Implementation/thesisEnv/dataset/train/images/hurricane-florence_00000001_post_disaster.png'
img1 = imread(path1) / 255
img2 = imread(path2) / 255

input = img2.reshape((1024*1024, 3))
output = img1.reshape((1024*1024, 3))

inp = input[np.arange(0, 1024*1024, 100), :]
oup = output[np.arange(0, 1024*1024, 100), :]

history = model.fit(inp, oup, epochs=100, batch_size=200,  verbose=0, validation_split=0.2)

print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


output2 = model.predict(input)
print(output2.shape)

# ----------------------------------------------------------------------------------------------
diff1 = img1 - img2
p2, p98 = np.percentile(diff1, (2, 98))
img_rescale1 = exposure.rescale_intensity(diff1, in_range=(p2, p98))
img_eq1 = exposure.equalize_hist(diff1)

output2 = output2.reshape((1024, 1024, 3))
diff = img1 - output2
p2, p98 = np.percentile(diff, (2, 98))
img_rescale = exposure.rescale_intensity(diff, in_range=(p2, p98))
img_eq = exposure.equalize_hist(diff)

img_adapteq1 = exposure.equalize_adapthist(diff1, clip_limit=0.03)
img_adapteq = exposure.equalize_adapthist(diff, clip_limit=0.03)


diff1 = (diff1 - np.amin(diff1)) / (np.amax(diff1) - np.amin(diff1))
diff = (diff - np.amin(diff)) / (np.amax(diff) - np.amin(diff))
img_rescale = (img_rescale - np.amin(img_rescale)) / (np.amax(img_rescale) - np.amin(img_rescale))
img_rescale1 = (img_rescale1 - np.amin(img_rescale1)) / (np.amax(img_rescale1) - np.amin(img_rescale1))


plt.figure(figsize=(16, 20))
plt.subplot(4, 3, 1), plt.imshow(img1), plt.title('image pre')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 2), plt.imshow(img2), plt.title('image post')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 3), plt.imshow(output2), plt.title('color matched')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 4), plt.imshow(diff1), plt.title('pre - post')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 5), plt.imshow(diff), plt.title('pre - matched')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 6), plt.imshow(img_rescale), plt.title('pre - matched : adj')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 7), plt.imshow(img_eq1), plt.title('pre - post : eq')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 8), plt.imshow(img_eq), plt.title('pre - matched : eq')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 9), plt.imshow(img_rescale1), plt.title('pre - post : adj')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 10), plt.imshow(img_adapteq1), plt.title('pre - post : adpt')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 11), plt.imshow(img_adapteq), plt.title('pre - matched : adpt')
plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()


# ----------------------------------------------------------------------------------------------
print('Checking files for ' + damage_class.upper() + ' in ' + disaster_type.upper() + ' ...')
files = database[disaster_type][damage_class]
for idx, row in files.iterrows():
    file = row['img_name']
    postFile = file.replace('labels', 'images') + '.png'
    preFile = postFile.replace('post', 'pre')

    postImg = imread(postFile) / 255
    preImg = imread(preFile) / 255

    input = postImg.reshape((1024*1024, 3))
    output = preImg.reshape((1024*1024, 3))
    inp = input[np.arange(0, 1024*1024, 100), :]
    oup = output[np.arange(0, 1024*1024, 100), :]
    history = model.fit(inp, oup, epochs=100, batch_size=200,
                        verbose=0, validation_split=0.2)
    postMatched = model.predict(input)
    postMatched = postMatched.reshape((1024, 1024, 3))

    diff_post_pre = postImg - preImg
    diff_matched_pre = postMatched - preImg

    samplePixels = row['samplePixels']
    for rc in samplePixels:
        postImg[rc[0], rc[1], :] = [1, 1, 0]
        preImg[rc[0], rc[1], :] = [1, 1, 0]
        diff_post_pre[rc[0], rc[1], :] = [1, 1, 0]
        diff_matched_pre[rc[0], rc[1], :] = [1, 1, 0]

    plt.figure()
    plt.suptitle(file.split('/')[-1])
    plt.subplot(221), plt.imshow(preImg), plt.xticks([]), plt.yticks([]), plt.title('pre image')
    plt.subplot(222), plt.imshow(postMatched), plt.xticks([]), plt.yticks([]), plt.title('post image')
    plt.subplot(223), plt.imshow(diff_post_pre), plt.xticks([]), plt.yticks([]), plt.title('"post-pre" image')
    plt.subplot(224), plt.imshow(diff_matched_pre), plt.xticks([]), plt.yticks([]), plt.title('"matched-pre" image')
    plt.tight_layout()
    plt.show()


