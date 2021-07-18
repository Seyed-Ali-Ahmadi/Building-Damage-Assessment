import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.draw import polygon
import datetime as dt

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# directory = 'D:/00.University/data/data sets/BD/'
# train = 'train/labels/'
# test = 'test/labels/'
# tier3 = 'tier3/labels/'
#
# trainFiles = os.listdir(directory + train)
# print('There are {} train *.json files.'.format(len(trainFiles)))
# trainFiles = [directory + train + file for file in trainFiles]
#
# testFiles = os.listdir(directory + test)
# print('There are {} test *.json files.'.format(len(testFiles)))
# testFiles = [directory + test + file for file in testFiles]
#
# tier3Files = os.listdir(directory + tier3)
# print('There are {} tier3 *.json files.'.format(len(tier3Files)))
# tier3Files = [directory + tier3 + file for file in tier3Files]
#
# labelFiles = trainFiles + testFiles + tier3Files
#
# labelsDB = pd.DataFrame(columns=['Group', 'img_name', 'Pre_Post', 'capture_date',
#                                  'sensor', 'gsd', 'disaster', 'disaster_type',
#                                  'off_nadir_angle', 'sun_azimuth', 'sun_elevation',
#                                  'pan_resolution', 'target_azimuth', 'buildings#',
#                                  'destroyed#', 'minor-damage#', 'major-damage#',
#                                  'no-damage#', 'un-classified#', 'buildings#pix',
#                                  'destroyed#pix', 'minor-damage#pix', 'major-damage#pix',
#                                  'no-damage#pix', 'un-classified#pix', 'latitude',
#                                  'longitude'], index=np.arange(0, len(labelFiles), 1))
#
# count = 0
# start_time = dt.datetime.now()
# for file in labelFiles:
#     file_df = pd.read_json(file)
#     num_nodamage, num_minor, num_major, num_destroyed, num_uncls = 0, 0, 0, 0, 0
#     pix_nodamage, pix_minor, pix_major, pix_destroyed, pix_uncls = 0, 0, 0, 0, 0
#     building_pix = 0
#
#     try:
#         lng_lat = file_df['features']['lng_lat'][0]['wkt'].partition('POLYGON ((')[2].partition('))')[0].split(', ')[0].split(' ')
#     except IndexError:
#         lng_lat = [np.nan, np.nan]
#
#     group = None
#     if 'train' in file:
#         group = 'Train'
#     elif 'test' in file:
#         group = 'Test'
#     elif 'tier3' in file:
#         group = 'Tier3'
#
#     for item in file_df['features']['xy']:
#         if len(file_df['features']['xy']) > 0:
#             vertices = item['wkt'].partition('POLYGON ((')[2].partition('))')[0].split(', ')
#             rows = []
#             cols = []
#             for vertex in vertices:
#                 cols.append(float(vertex.split(' ')[0]))
#                 rows.append(float(vertex.split(' ')[1]))
#             rr, cc = polygon(rows, cols, (1024, 1024))
#             building_pix += len(rr)
#
#             if file.split('_')[-2] == 'post':
#                 if item['properties']['subtype'] == 'no-damage':
#                     num_nodamage += 1
#                     pix_nodamage += len(rr)
#                 elif item['properties']['subtype'] == 'minor-damage':
#                     num_minor += 1
#                     pix_minor += len(rr)
#                 elif item['properties']['subtype'] == 'major-damage':
#                     num_major += 1
#                     pix_major += len(rr)
#                 elif item['properties']['subtype'] == 'destroyed':
#                     num_destroyed += 1
#                     pix_destroyed += len(rr)
#                 elif item['properties']['subtype'] == 'un-classified':
#                     num_uncls += 1
#                     pix_uncls += len(rr)
#             else:
#                 pass
#         else:
#             pass
#
#     labelsDB.loc[count] = {
#         'Group': group,
#         'img_name': file[:-5],
#         'Pre_Post': file.split('_')[-2],
#         'capture_date': pd.to_datetime(file_df['metadata']['capture_date']),
#         'sensor': file_df['metadata']['sensor'],
#         'gsd': file_df['metadata']['gsd'],
#         'disaster': file_df['metadata']['disaster'],
#         'disaster_type': file_df['metadata']['disaster_type'],
#         'off_nadir_angle': file_df['metadata']['off_nadir_angle'],
#         'sun_azimuth': file_df['metadata']['sun_azimuth'],
#         'sun_elevation': file_df['metadata']['sun_elevation'],
#         'pan_resolution': file_df['metadata']['pan_resolution'],
#         'target_azimuth': file_df['metadata']['target_azimuth'],
#         'buildings#': len(file_df['features']['xy']),
#         'destroyed#': num_destroyed,
#         'minor-damage#': num_minor,
#         'major-damage#': num_major,
#         'no-damage#': num_nodamage,
#         'un-classified#': num_uncls,
#         'buildings#pix': building_pix,
#         'destroyed#pix': pix_destroyed,
#         'minor-damage#pix': pix_minor,
#         'major-damage#pix': pix_major,
#         'no-damage#pix': pix_nodamage,
#         'un-classified#pix': pix_uncls,
#         'latitude': float(lng_lat[1]),
#         'longitude': float(lng_lat[0])
#     }
#     count += 1
#
# end_time = dt.datetime.now()
# print(end_time - start_time)
#
# labelsDB.to_csv('All_Data_Props.csv')
# labelsDB.to_pickle('All_Data_Props.pkl')
# ------------------------------------------------------------
labelsDB = pd.read_pickle('All_Data_Props.pkl')
print(labelsDB)

labelsDB_pre = labelsDB[labelsDB['Pre_Post'] == 'pre']
labelsDB_post = labelsDB[labelsDB['Pre_Post'] == 'post']
print(labelsDB_pre.shape, labelsDB_post.shape)
# ------------------------------------------------------------
# How many images pairs in each group (train/test/tier3)
group_counts = labelsDB_post['Group'].value_counts()
plt.figure()
plt.bar(group_counts.index, group_counts.values, align='center', width=0.5, edgecolor='black')
plt.title('Number of image pairs (pre/post) in each group.')
plt.grid(True, linestyle='--', color='grey', alpha=.25)
plt.tight_layout()
plt.show()
# ------------------------------------------------------------
# Time difference between pre and post images.
tDeltas = []
for i in range(len(labelsDB_post)):
    tDeltas.append((labelsDB_post['capture_date'].iloc[i] - labelsDB_pre['capture_date'].iloc[i]).days)
tdeltas = pd.Series(tDeltas).value_counts()
plt.figure()
plt.bar(tdeltas.index, tdeltas.values, align='center', width=5, edgecolor='black')
plt.title('Frequency of time differences between Pre and Post images (Days)\n'
          'Total number of image pairs {}'.format(int(len(labelsDB) / 2)))
plt.xlabel('Time difference in days')
plt.ylabel('Number of images with specific time difference')
plt.grid(True, linestyle='--', color='grey', alpha=.25)
plt.xticks(np.arange(0, 2000, 100), rotation='vertical')
plt.xlim([min(tdeltas.index) - 10, max(tdeltas.index) + 10])
plt.tight_layout()
plt.show()
# ------------------------------------------------------------
# How many images are acquired in each date?
import matplotlib.dates as mdates

plt.figure()
plt.vlines(x=labelsDB_pre['capture_date'].value_counts().index,
           ymin=0, ymax=labelsDB_pre['capture_date'].value_counts().values,
           colors='r', alpha=0.65, linewidth=2, label='Pre disaster images')
plt.vlines(x=labelsDB_post['capture_date'].value_counts().index,
           ymin=0, ymax=labelsDB_post['capture_date'].value_counts().values,
           colors='b', alpha=0.65, linewidth=2, label='Post disaster images')
plt.title('Number of images in each date')
plt.xlabel('Date')
plt.ylabel('Number of images')
plt.ylim([0, 1900])
plt.xlim(["2006-01-01", "2020-01-01"])
ax = plt.gca()
ax.xaxis.set_minor_locator(mdates.YearLocator())
plt.grid(True, linestyle='--', color='grey', alpha=.25)
plt.tight_layout()
plt.legend()
plt.show()
# ------------------------------------------------------------
# plot the capture date variations by month of year.
months_pre = []
months_post = []
for i in range(int(len(labelsDB) / 2)):
    months_pre.append(labelsDB_pre['capture_date'].iloc[i].month)
    months_post.append(labelsDB_post['capture_date'].iloc[i].month)
months_pre = pd.Series(months_pre).value_counts()
months_post = pd.Series(months_post).value_counts()

plt.figure()
plt.bar(months_pre.index, months_pre.values, label='Pre disaster images',
        width=1, align='center', color='r', alpha=0.65, edgecolor='black')
plt.bar(months_post.index, months_post.values, label='Post disaster images',
        width=1, align='center', color='b', alpha=0.65, edgecolor='black')
ax = plt.gca()
plt.xticks(ticks=range(1, 13),
           labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.title('Distribution of image acquisition dates over the years.')
plt.xlabel('Month of year')
plt.ylabel('Number of images')
plt.legend()
plt.grid(True, linestyle='--', color='grey', alpha=.25)
plt.tight_layout()
plt.show()
# ------------------------------------------------------------
# Pre disaster dates vs. Post disaster dates
unique_pre_post_pairs = pd.concat((labelsDB_pre.reset_index()['capture_date'],
                                   labelsDB_post.reset_index()['capture_date'],
                                   labelsDB_pre.reset_index()['disaster_type']),
                                  axis=1).drop_duplicates()
color_marker = {'volcano': ['r', '*'], 'flooding': ['g', 'o'],
                'wind': ['b', 's'], 'earthquake': ['k', 'p'],
                'tsunami': ['c', 'v'], 'fire': ['m', 'x']}
plt.figure()
for c in color_marker:
    plt.scatter(x=unique_pre_post_pairs[unique_pre_post_pairs['disaster_type'] == c].iloc[:, 0],
                y=unique_pre_post_pairs[unique_pre_post_pairs['disaster_type'] == c].iloc[:, 1],
                color=color_marker[c][0], label=c, marker=color_marker[c][1])
plt.title('Distribution of image acquisition dates and types (pre-post pair)')
plt.xlabel('Pre disasters acquisition dates')
plt.ylabel('Post disasters acquisition dates')
plt.xlim(["2006-01-01", "2020-01-01"])
plt.ylim(["2011-01-01", "2020-01-01"])
plt.legend()
plt.grid(True, linestyle='--', color='grey', alpha=.25)
plt.tight_layout()
plt.show()
# ------------------------------------------------------------
# Number of image pairs per disaster type
type_counts = labelsDB_post['disaster_type'].value_counts()
plt.figure()
barlist = plt.bar(type_counts.index, type_counts.values,
                  align='center', width=0.5)
for i in range(len(barlist)):
    barlist[i].set_color(color_marker[type_counts.index[i]][0])
plt.title('Number of image pairs (pre/post) in each disaster type.')
plt.xlabel('Disaster type')
plt.ylabel('Number of image pairs')
plt.grid(True, linestyle='--', color='grey', alpha=.25)
plt.tight_layout()
plt.show()
# ------------------------------------------------------------
# Number of image pairs per unique disaster
dis_counts = labelsDB_post['disaster'].value_counts()
plt.figure()
barlist = plt.bar(dis_counts.index, dis_counts.values, align='center', width=0.5)
for i in range(len(barlist)):
    barlist[i].set_color(color_marker[pd.unique(labelsDB[labelsDB['disaster'] == dis_counts.index[i]]['disaster_type'])[0]][0])
plt.title('Number of image pairs (pre/post) in each disaster.')
plt.xlabel('Disaster')
plt.ylabel('Number of image pairs')
plt.xticks(rotation='vertical')
plt.grid(True, linestyle='--', color='grey', alpha=.25)
plt.tight_layout()
plt.show()
# ------------------------------------------------------------
# Number of images per sensor
plt.figure()
sensor_counts = labelsDB_pre['sensor'].value_counts()
plt.bar(np.arange(len(sensor_counts)) - 0.15, sensor_counts.values, label='Pre disaster images',
        width=0.3, align='center', color='r', alpha=0.65, edgecolor='black')
sensor_counts = labelsDB_post['sensor'].value_counts()
plt.bar(np.arange(len(sensor_counts)) + 0.15, sensor_counts.values, label='Post disaster images',
        width=0.3, align='center', color='b', alpha=0.65, edgecolor='black')
plt.title('Number of images captured by each sensor.')
plt.xlabel('Sensor')
plt.ylabel('Number of images')
ax = plt.gca()
ax.set_xticks(np.arange(len(labelsDB['sensor'].value_counts())))
ax.set_xticklabels(labelsDB['sensor'].value_counts().index)
plt.grid(True, linestyle='--', color='grey', alpha=.25)
plt.tight_layout()
plt.legend()
plt.show()
# ------------------------------------------------------------
# Number of images per sensor per disaster type
unique_sensor = pd.unique(labelsDB['sensor'])
unique_type = pd.unique(labelsDB['disaster_type'])
plt.figure()
w = 0.85
for i in range(len(unique_sensor)):
    subset = labelsDB[labelsDB['sensor'] == unique_sensor[i]]
    for j in range(len(unique_type)):
        subset_ = subset[subset['disaster_type'] == unique_type[j]]
        try:
            if i == 2:
                plt.bar(j*w/len(unique_type) + i,
                        subset_['disaster_type'].value_counts().values,
                        color=color_marker[unique_type[j]][0], width=w/len(unique_type),
                        label=unique_type[j])
            else:
                plt.bar(j * w / len(unique_type) + i,
                        subset_['disaster_type'].value_counts().values,
                        color=color_marker[unique_type[j]][0], width=w / len(unique_type))
        except ValueError:
            continue
plt.title('Number of images taken by each sensor per disaster type')
plt.xlabel('Sensor')
plt.ylabel('Number of images')
plt.legend()
ax = plt.gca()
ax.set_xticks(np.arange(len(unique_sensor)) + w/2)
ax.set_xticklabels(unique_sensor)
plt.grid(True, linestyle='--', color='grey', alpha=.25)
plt.tight_layout()
plt.show()
# ------------------------------------------------------------
plt.figure()
plt.hist(labelsDB_pre['gsd'], bins=40, range=(1, 3.3), color='r',
         alpha=0.6, edgecolor='r', hatch='//', histtype='stepfilled')
plt.hist(labelsDB_post['gsd'], bins=40, range=(1, 3.3), color='b',
         alpha=0.6, edgecolor='b', hatch='\\\\', histtype='stepfilled')
plt.title('Histogram of GSD values')
plt.xlabel('Ground Sample Distance (m)')
plt.ylabel('Number of images')
plt.xticks(ticks=np.arange(1, 3.3, 0.1))
plt.grid(True, linestyle='--', color='grey', alpha=.25)
plt.tight_layout()
plt.show()
# ------------------------------------------------------------
builing_per_class = np.sum(np.array(labelsDB_post[['destroyed#', 'minor-damage#',
                                                   'major-damage#', 'no-damage#',
                                                   'un-classified#']]), axis=0)
plt.figure()
plt.bar(x=np.arange(5), height=builing_per_class)
plt.xticks(ticks=np.arange(5),
           labels=['destroyed', 'minor-damage',
                   'major-damage', 'no-damage',
                   'un-classified'], rotation=10)
plt.title('Total number of buildings is ' + str(sum(builing_per_class)))
plt.xlabel('Classes')
plt.ylabel('Number of buildings')
plt.grid(True, linestyle='--', color='grey', alpha=.25)
plt.tight_layout()
plt.show()

pixels_per_class = np.sum(np.array(labelsDB_post[['destroyed#pix', 'minor-damage#pix',
                                                  'major-damage#pix', 'no-damage#pix',
                                                  'un-classified#pix']]), axis=0)
plt.figure()
plt.bar(x=np.arange(5), height=pixels_per_class)
plt.xticks(ticks=np.arange(5),
           labels=['destroyed', 'minor-damage',
                   'major-damage', 'no-damage',
                   'un-classified'], rotation=10)
plt.title('Total number of building pixels is ' + str(sum(pixels_per_class)))
plt.xlabel('Classes')
plt.ylabel('Number of buildings')
plt.grid(True, linestyle='--', color='grey', alpha=.25)
plt.tight_layout()
plt.show()
# ------------------------------------------------------------
pix_dest = np.array(labelsDB_post[labelsDB_post['destroyed#pix'] > 0]['destroyed#pix'])
pix_mino = np.array(labelsDB_post[labelsDB_post['minor-damage#pix'] > 0]['minor-damage#pix'])
pix_majo = np.array(labelsDB_post[labelsDB_post['major-damage#pix'] > 0]['major-damage#pix'])
pix_nodm = np.array(labelsDB_post[labelsDB_post['no-damage#pix'] > 0]['no-damage#pix'])
area_dest = pix_dest * np.array(labelsDB_post[labelsDB_post['destroyed#pix'] > 0]['gsd']) ** 2
area_mino = pix_mino * np.array(labelsDB_post[labelsDB_post['minor-damage#pix'] > 0]['gsd'])
area_majo = pix_majo * np.array(labelsDB_post[labelsDB_post['major-damage#pix'] > 0]['gsd'])
area_nodm = pix_nodm * np.array(labelsDB_post[labelsDB_post['no-damage#pix'] > 0]['gsd'])

plt.figure()
plt.hist(area_dest, bins=30, color='r', range=(0, 6*1e5),
         alpha=0.3, edgecolor='r', label='Destroyed')
plt.hist(area_mino, bins=30, color='g', range=(0, 6*1e5),
         alpha=0.3, edgecolor='g', label='Minor-Damage')
plt.hist(area_majo, bins=30, color='b', range=(0, 6*1e5),
         alpha=0.3, edgecolor='b', label='Major-Damage')
plt.hist(area_nodm, bins=30, color='k', range=(0, 6*1e5),
         alpha=0.3, edgecolor='k', label='No-Damage')
plt.xlim([-0.1*1e5, 0.6*1e6])
plt.title('Histogram of buildings area')
plt.xlabel('Area of buildings ($m^2$)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, linestyle='--', color='grey', alpha=.25)
plt.tight_layout()
plt.show()

plt.figure()
plt.boxplot([area_dest, area_mino, area_majo, area_nodm],
            labels=['Destroyed', 'Minor-Damage', 'Major-Damage', 'No-Damage'],
            showfliers=False, notch=True, widths=0.2, showmeans=True)
plt.title('Box plot of buildings'' area')
plt.xlabel('Damage type')
plt.ylabel('Area of buildings ($m^2$)')
plt.grid(True, linestyle='--', color='grey', alpha=.25)
plt.tight_layout()
plt.show()
# ------------------------------------------------------------
# Number of buildings in each disaster type per class
classes = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']
color4class = ['tab:purple', 'tab:orange', 'tab:brown', 'goldenrod']
plt.figure()
w = 0.75
for i in range(len(unique_type)):
    subset = labelsDB[labelsDB['disaster_type'] == unique_type[i]]
    for j in range(len(classes)):
        num_building_class_disaster = sum(subset[classes[j] + '#'])
        if i == 0:
            plt.bar((j - 1) * w / len(classes) + i,
                    num_building_class_disaster,
                    color=color4class[j], width=w / len(classes),
                    label=classes[j])
        else:
            plt.bar((j - 1) * w / len(classes) + i,
                    num_building_class_disaster,
                    color=color4class[j], width=w / len(classes))

plt.title('Number of buildings per disaster-type per class')
plt.xlabel('Disaster Types')
plt.ylabel('Number of buildings')
plt.legend()
ax = plt.gca()
ax.set_xticks(np.arange(len(unique_type)))
ax.set_xticklabels(unique_type)
plt.grid(True, linestyle='--', color='grey', alpha=.25)
plt.tight_layout()
plt.show()



