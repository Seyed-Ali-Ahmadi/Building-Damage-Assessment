import json
import numpy as np
import os
import time
import datetime
import matplotlib.pyplot as plt

"""
My PhD thesis is going to contain only Earthquakes and Floods. This is why I am decreasing my files to these 2 classes.
"""


def read_json(json_path):
    """
    :param json_path: path to load json from
    :returns: a python dictionary of json features
    """
    annotations = json.load(open(json_path))
    return annotations


def get_metadata(json_file):
    gsd = json_file['metadata']['gsd']
    sun_elev = json_file['metadata']['sun_elevation']
    off_nadir_angle = json_file['metadata']['off_nadir_angle']
    disaster_type = json_file['metadata']['disaster_type']
    date = json_file['metadata']['capture_date']
    return [gsd, sun_elev, off_nadir_angle, disaster_type, date]


path = 'D:/00.University/data/data sets/BD/train/labels/'
labelFiles = os.listdir(path)

Fld_Ertqk_files = []
for labelfile in labelFiles:
    [gsd, sun_elev, off_nadir_angle, disaster_type, date] = get_metadata(read_json(path + labelfile))
    if disaster_type == 'earthquake' or disaster_type == 'flooding':
        Fld_Ertqk_files.append(labelfile)

print('\n >>> There are {0} files which have disasters'
      ' of type {1} and {2}'.format(len(Fld_Ertqk_files), 'Earthquake', 'Flooding'))

with open(path + 'Floods & Earthquakes 2.txt', 'w') as f:
    for item in Fld_Ertqk_files:
        f.write('%s\n' % item[:-5])  # remove the .json extension from the names
