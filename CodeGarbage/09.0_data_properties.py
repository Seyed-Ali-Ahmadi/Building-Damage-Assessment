import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cartopy.crs as ccrs

path = 'D:/00.University/data/data sets/BD/train/labels/'
# -----------------------------------------------------------------------
# # Get metadata
# labelFiles = os.listdir(path)
#
# all_df = pd.DataFrame()
# for file in labelFiles:
#     file_df = pd.read_json(path + file)
#     file_df = pd.DataFrame(file_df['metadata'])
#     all_df = pd.concat((all_df, file_df.transpose()), axis=0, ignore_index=True)
#
# print(all_df)
# print(all_df.shape)
# print(list(all_df.columns))
#
# all_df.to_pickle('./properties.pkl')
# -----------------------------------------------------------------------
# Get location features
# labelFiles = os.listdir(path)
#
# all_locs = []
# for file in labelFiles:
#     file_df = pd.read_json(path + file)
#     try:
#         temp = file_df['features']['lng_lat'][0]['wkt']
#         loc_str = temp.partition('POLYGON ((')[2].partition('))')[0].split(', ')[0]
#         lng = float(loc_str.split(' ')[0])
#         lat = float(loc_str.split(' ')[1])
#         all_locs.append([lng, lat])
#     except:
#         continue
#
# print(np.array(all_locs))
# print(np.array(all_locs).shape)
#
# np.save('./locations.npy', np.array(all_locs))
# -----------------------------------------------------------------------
all_df1 = pd.read_pickle('./properties.pkl')
all_df2 = pd.read_pickle('./properties_tier3.pkl')
all_df = pd.concat((all_df1, all_df2))
# -----------------------------------------------------------------------
# Separating Pre/Post data.
pre_df = pd.DataFrame()
post_df = pd.DataFrame()
for index, row in all_df.iterrows():
    if 'pre' in row['img_name']:
        pre_df = pd.concat((pre_df, row.transpose()), axis=1)
    elif 'post' in row['img_name']:
        post_df = pd.concat((post_df, row.transpose()), axis=1)

pre_df = pre_df.transpose()
post_df = post_df.transpose()
print(pre_df.shape)
print(post_df.shape)
# Unique sensor type
sensor_names = ['GEOEYE01', 'WORLDVIEW03_VNIR', 'WORLDVIEW02', 'QUICKBIRD02']
sensor_marker = ['o', '*', 'd', 's']
# Unique disaster type
disaster_names = ['earthquake', 'wind', 'tsunami', 'flooding', 'volcano', 'fire']
disaster_color = ['r', 'g', 'b', 'k', 'c', 'm']
# -----------------------------------------------------------------------
# Plot histograms
plt.figure()
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.subplot(231), plt.title('Pre/Post Sun Azimuth Angles (degrees)')
plt.hist(pre_df['sun_azimuth'], bins=20, color='r', range=(40, 180), label='Pre Sun Az', alpha=0.5)
plt.hist(post_df['sun_azimuth'], bins=20, color='b', range=(40, 180), label='Post Sun Az', alpha=0.5)
plt.legend(), plt.grid(True, alpha=0.5, linestyle='--', linewidth='0.5')
plt.subplot(232), plt.title('Pre/Post Sun Elevation Angles (degrees)')
plt.hist(pre_df['sun_elevation'], bins=20, color='r', range=(30, 75), label='Pre Sun Elev', alpha=0.5)
plt.hist(post_df['sun_elevation'], bins=20, color='b', range=(30, 75), label='Post Sun Elev', alpha=0.5)
plt.legend(), plt.grid(True, alpha=0.5, linestyle='--', linewidth='0.5')
plt.subplot(234), plt.title('Pre/Post Target Azimuth Angles (degrees)')
plt.hist(pre_df['target_azimuth'], bins=20, color='r', range=(0, 360), label='Pre Tgt Az', alpha=0.5)
plt.hist(post_df['target_azimuth'], bins=20, color='b', range=(0, 360), label='Post Tgt Az', alpha=0.5)
plt.legend(), plt.grid(True, alpha=0.5, linestyle='--', linewidth='0.5')
plt.subplot(235), plt.title('Pre/Post Satellite Off-Nadir Angles (degrees)')
plt.hist(pre_df['off_nadir_angle'], bins=20, color='r', range=(0, 45), label='Pre Sat Off-Ndr', alpha=0.5)
plt.hist(post_df['off_nadir_angle'], bins=20, color='b', range=(0, 45), label='Post Sat Off-Ndr', alpha=0.5)
plt.legend(), plt.grid(True, linestyle='--', linewidth='0.5')
plt.subplot(233), plt.title('Pre/Post Image Resolution GSD (meters)')
plt.hist(pre_df['gsd'], bins=30, range=(1, 3), label='Pre GSD', color='r', alpha=0.5)
plt.hist(post_df['gsd'], bins=30, range=(1, 3), label='Post GSD', color='b', alpha=0.5)
plt.legend(), plt.grid(True, linestyle='--', linewidth='0.5')
plt.subplot(236), plt.title('GSD vs. Off-Nadir Angle')
plt.grid(True, linestyle='--', linewidth='0.5', alpha=0.5)
plt.xlabel('Off-Nadir Angle (degrees)'), plt.ylabel('GSD (meters)')
for i in range(0, int(all_df.shape[0]/2), 10):
    plt.plot(pre_df['off_nadir_angle'].iloc[i], pre_df['gsd'].iloc[i], alpha=0.1,
             marker=sensor_marker[sensor_names.index(pre_df['sensor'].iloc[i])],
             color=disaster_color[disaster_names.index(pre_df['disaster_type'].iloc[i])],
             markersize=10, fillstyle='left')
    plt.plot(post_df['off_nadir_angle'].iloc[i], post_df['gsd'].iloc[i], alpha=0.1,
             marker=sensor_marker[sensor_names.index(post_df['sensor'].iloc[i])],
             color=disaster_color[disaster_names.index(post_df['disaster_type'].iloc[i])],
             markersize=10, fillstyle='right')
plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# Draw a hemisphere.
u = np.linspace(0, np.pi / 2, 30)
v = np.linspace(0, 2 * np.pi, 30)

x_sphere = np.outer(np.sin(u), np.sin(v))
y_sphere = np.outer(np.sin(u), np.cos(v))
z_sphere = np.outer(np.cos(u), np.ones_like(v))
# -----------------------------------------------------------------------
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_wireframe(x_sphere, y_sphere, z_sphere, alpha=0.05, color='k', linestyle='--')
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# Plot Sun position before disaster.
pre_sun_el = (pre_df['sun_elevation'] * np.pi / 180).tolist()
pre_sun_az = ((90 - pre_df['sun_azimuth']) * np.pi / 180).tolist()

x = np.cos(pre_sun_el) * np.cos(pre_sun_az)
y = np.cos(pre_sun_el) * np.sin(pre_sun_az)
z = np.sin(pre_sun_el)
ax.scatter3D(x, y, z, s=15, marker='o', c='y', alpha=0.1)
# -----------------------------------------------------------------------
# Plot Sun position after disaster.
post_sun_el = (post_df['sun_elevation'] * np.pi / 180).tolist()
post_sun_az = ((90 - post_df['sun_azimuth']) * np.pi / 180).tolist()

x = np.cos(post_sun_el) * np.cos(post_sun_az)
y = np.cos(post_sun_el) * np.sin(post_sun_az)
z = np.sin(post_sun_el)
ax.scatter3D(x, y, z, s=15, marker='*', c='c', alpha=0.1)
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# Plot satellite position in pre-disaster.
pre_sat_el = ((90 - pre_df['off_nadir_angle']) * np.pi / 180).tolist()
pre_sat_az = ((90 - pre_df['target_azimuth']) * np.pi / 180).tolist()

x = np.cos(pre_sat_el) * np.cos(pre_sat_az)
y = np.cos(pre_sat_el) * np.sin(pre_sat_az)
z = np.sin(pre_sat_el)
ax.scatter3D(x, y, z, s=15, marker='s', c='r', alpha=0.1)
# -----------------------------------------------------------------------
# Plot satellite position in post-disaster.
post_sat_el = ((90 - post_df['off_nadir_angle']) * np.pi / 180).tolist()
post_sat_az = ((90 - post_df['target_azimuth']) * np.pi / 180).tolist()

x = np.cos(post_sat_el) * np.cos(post_sat_az)
y = np.cos(post_sat_el) * np.sin(post_sat_az)
z = np.sin(post_sat_el)
ax.scatter3D(x, y, z, s=15, marker='d', c='b', alpha=0.1)
# -----------------------------------------------------------------------
# ----------------------------------------------------------------------- Plot in polar projection
# Plot previous points on polar coordinates.
axp = fig.add_subplot(1, 2, 2, projection='polar')
axp.set_theta_zero_location('N')
axp.set_theta_direction(-1)
# -----------------------------------------------------------------------
theta_sun_pre = list(np.pi / 2 - np.array(pre_sun_az))
axp.scatter(theta_sun_pre, np.cos(pre_sun_el), color='y', marker='o')
theta_sun_post = list(np.pi / 2 - np.array(post_sun_az))
axp.scatter(theta_sun_post, np.cos(post_sun_el), color='c', marker='*')
# -----------------------------------------------------------------------
theta_sat_pre = list(np.pi / 2 - np.array(pre_sat_az))
axp.scatter(theta_sat_pre, np.cos(pre_sat_el), color='r',
            s=list(np.power(list(pre_df['gsd']), 4)*pre_df['off_nadir_angle']/2),
            alpha=0.02)
theta_sat_post = list(np.pi / 2 - np.array(post_sat_az))
axp.scatter(theta_sat_post, np.cos(post_sat_el), color='b',
            s=list(np.power(list(post_df['gsd']), 4)*post_df['off_nadir_angle']/2),
            alpha=0.02)
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
ax.quiver(0, 0, 0, 1.2, 0, 0, color='r', arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, 0, 1.2, 0, color='k', arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, 0, 0, 1.2, color='b', arrow_length_ratio=0.1)
ax.text(1.3, 0, 0, 'East')
ax.text(0, 1.3, 0, 'North')
ax.text(0, 0, 1.3, 'Zenith')
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([0, 1.5])
fig.tight_layout()
plt.show()
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
plt.figure()
plt.subplot(121)
plt.plot(pd.to_datetime(pre_df['capture_date']), pre_df['gsd'], 'or', alpha=0.05)
plt.plot(pd.to_datetime(pre_df['capture_date']), pre_df['pan_resolution'], 'ob', alpha=0.05)
plt.subplot(122)
plt.plot(pd.to_datetime(post_df['capture_date']), post_df['gsd'], 'or', alpha=0.05)
plt.plot(pd.to_datetime(post_df['capture_date']), post_df['pan_resolution'], 'ob', alpha=0.05)
plt.show()
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# Plot locations
import cartopy
import cartopy.io.shapereader as shpreader

all_locs1 = np.load('./locations.npy')
all_locs2 = np.load('./locations_tier3.npy')
all_locs = np.vstack((all_locs1, all_locs2))

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
ax.set_global()
# ax.stock_img()
ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax.add_feature(cartopy.feature.OCEAN, facecolor='lightblue')
ax.add_feature(cartopy.feature.LAND, facecolor='green', alpha=0.6)
ax.coastlines(resolution='10m', linewidth=0.3)
for i in range(all_locs.shape[0]):
    ax.plot(all_locs[i, 0], all_locs[i, 1], transform=ccrs.PlateCarree(), alpha=0.5,
            marker=sensor_marker[sensor_names.index(all_df['sensor'].iloc[i])],
            color=disaster_color[disaster_names.index(all_df['disaster_type'].iloc[i])],
            markersize=np.power(all_df['gsd'].iloc[i], 1) * all_df['off_nadir_angle'].iloc[i] / 10)
plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
fig = plt.figure()
axp = fig.add_subplot(1, 1, 1, projection='polar')
axp.set_theta_zero_location('N')
axp.set_theta_direction(-1)
diff_gsd = np.abs(np.array(post_df['gsd']) - np.array(pre_df['gsd']))
diff_az = np.array(post_df['target_azimuth']) - np.array(pre_df['target_azimuth'])
diff_el = np.array(post_df['off_nadir_angle']) - np.array(pre_df['off_nadir_angle'])
axp.scatter(diff_az, np.cos(list(diff_el * 2 * np.pi/180)), color='k',
            s=list(np.power(diff_gsd*100, 1.2)), alpha=0.02)
plt.title('Relative pre-post satellite configuration.')
plt.show()






















