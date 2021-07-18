from tensorflow.keras.applications import NASNetMobile, InceptionResNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

# model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(1024, 1024, 3))
# model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(75, 75, 3))
model = NASNetMobile(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
img_path = r"D:/00.University/PhD Thesis Implementation/thesisEnv/dataset/train/images/hurricane-florence_00000001_post_disaster.png"
img = image.load_img(img_path)  # , target_size=(224, 224)
x = image.img_to_array(img)
x = x[540:540+224, 680:680+224, :]
# x = x[100:100+224, 120:120+224, :]
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
features = model.predict(x)
featuresFlat = GlobalAveragePooling2D()(features).numpy()[0]
print(features.shape, featuresFlat.shape)

plt.figure()
plt.plot(features[0, :, :, :].T.flatten(), 'b', alpha=0.5)
plt.plot(np.arange(0, len(featuresFlat) * 7 * 7, 7 * 7), featuresFlat, 'k')
plt.show()


"""
https://towardsdatascience.com/convolutional-neural-network-feature-map-and-filter-visualization-f75012a5a49c
"""
# # Show weights
# features = features[0, :, :, :]
# for layer in model.layers:
#     if 'conv' in layer.name:
#         a = layer.get_weights()[0]
#         print(layer.name, a.shape)
#         plt.figure()
#         filter_cnt = 1
#         for i in range(a.shape[3]):
#             filt = a[:, :, :, i]
#             for j in range(a.shape[0]):
#                 ax = plt.subplot(a.shape[3], a.shape[0], filter_cnt)
#                 ax.set_xticks([])
#                 ax.set_yticks([])
#                 plt.imshow(filt[:, :, j])
#                 filter_cnt += 1
#         plt.show()

# # Show convolutional features
successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = Model(inputs=model.input, outputs=successive_outputs)
successive_feature_maps = visualization_model.predict(x)

layer_names = [layer.name for layer in model.layers]
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    print(feature_map.shape)
    if len(feature_map.shape) == 4:
        n_features = feature_map.shape[-1]
        size = feature_map.shape[1]
        display_grid = np.zeros((size, size * n_features))
        for i in range(n_features):
            a = feature_map[0, :, :, i]
            a -= a.mean()
            a /= a.std()
            a *= 64
            a += 128
            a = np.clip(a, 0, 255).astype('uint8')
            display_grid[:, i * size: (i + 1) * size] = a
        scale = 20. / n_features
        plt.figure(figsize=(scale * n_features, scale))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()






