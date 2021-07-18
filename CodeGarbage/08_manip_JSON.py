# data['features']['xy'][0]['properties']['feature_type']
#
# data['features']['xy'][0]['properties']['subtype']

from Functions import *


root = 'D:/00.University/data/data sets/BD/train/'
imageDir = root + 'subset_images/'
labelDir = root + 'labels/'
jsonFiles = os.listdir(labelDir)

uniqueNames = read_jsondir(labelDir)

damageType = [0, 0, 0, 0, 0]

for name in uniqueNames:
    pre = name + '_pre_disaster.json'
    post = name + '_post_disaster.json'

    f = open(labelDir + post, )
    data_post = json.load(f)

    num_building_pre = len(data_post['features']['xy'])
    for item in data_post['features']['xy']:
        type = item['properties']['subtype']
        if type == 'destroyed':
            damageType[0] += 1
        elif type == 'major-damage':
            damageType[1] += 1
        elif type == 'minor-damage':
            damageType[2] += 1
        elif type == 'no-damage':
            damageType[3] += 1
        else:
            damageType[4] += 1

print(damageType)
print(sum(damageType))



