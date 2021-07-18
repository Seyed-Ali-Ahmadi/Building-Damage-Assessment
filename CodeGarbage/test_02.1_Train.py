import time
import pickle
import numpy as np
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt


root = 'D:/00.University/data/data sets/BD/train/'
imageDir = root + '1/'
labelDir = root + '2/'

class_names = ['destroyed', 'major-damage', 'minor-damage', 'no-damage', 'un-classified']
class_labels = [4, 3, 2, 1, -1]

FeatureVector = pickle.load(open(root + 'Features.p', 'rb'))
# plt.figure(), plt.hist(FeatureVector[:, -1].ravel()), plt.show()

FeatureManipulated = FeatureVector
for i in range(FeatureManipulated.shape[0]):
    if FeatureManipulated[i, -1] != 4:
        FeatureManipulated[i, -1] = 1

plt.figure(), plt.hist(FeatureVector[:, -1].ravel()), plt.show()
np.random.shuffle(FeatureManipulated)


print('5- Fitting classifier ...')
t0 = time.time()
# clf = SVC(probability=False)
clf = LinearSVC()
clf.fit(FeatureManipulated[:, 2:-1], FeatureManipulated[:, -1])
print('         ' + str(round(time.time() - t0, 2)))

print('---- Writing File.')
pickle.dump(clf, open(root + 'classifier_3.p', 'wb'))
