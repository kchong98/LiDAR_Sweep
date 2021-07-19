from pandaset import DataSet
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import pptk

# Importing dataset
dataset = DataSet('C:/Users/maste/4AI3/PandaSet')

# Determining sequences that have semantic segmentation labels
sequences = dataset.sequences(with_semseg=True)
# print(sequences)

j = ''
k = ''

# Loading dataset into DataFrame
for seq in sequences:
    for i in range(80):
        if i < 10:
            point = '0' + str(i)
        else:
            point = str(i)
        pfile_semseg = 'C:/Users/maste/4AI3/PandaSet/{}/annotations/semseg/{}.pkl'.format(seq, point)
        pfile_data = 'C:/Users/maste/4AI3/PandaSet/{}/lidar/{}.pkl'.format(seq, point)
        j = seq
        k = point
        # with open(pfile_semseg, 'rb') as fin:
        #     semseg = pickle.load(fin)
        # with open(pfile_data, 'rb') as fin:
        #     data = pickle.load(fin)
        #
        # temp = data
        # temp['class'] = semseg
        # lidar_data.append(temp)

seq_loaded = dataset[j]
seq_loaded.load()
class_dict = {}

# Saving classes to dict
for key in seq_loaded.semseg.classes:
    class_dict[key] = seq_loaded.semseg.classes[key]

# Saving classes to json file
with open('classes.json', 'w') as fp:
    json.dump(class_dict, fp)

# Loading color map
with open('color.json') as f:
    color_map = json.load(f)

# Loading data into DataFrame
with open(pfile_semseg, 'rb') as fin:
    semseg = pickle.load(fin)
with open(pfile_data, 'rb') as fin:
    data = pickle.load(fin)

print(type(color_map))

# Visualising point-cloud in 3D space
# rgb = pptk.rand(len(data.index), 3)
rgb = np.zeros((len(data.index), 3))
row = 0
for a in semseg.to_numpy():
    rgb[row,:] = np.reshape(np.array(color_map[class_dict[str(int(a))]]), (1,3))/255
    # print(color_map[class_dict[str(int(a))]])
    row = row + 1
v = pptk.viewer(data[['x','y','z']].to_numpy(), rgb)
v.color_map('cool', scale=[0, 42])
v.color_map([[0, 0, 0], [1, 1, 1]])

# pfile = 'C:/Users/maste/4AI3/PandaSet/001/annotations/semseg/05.pkl'
# with open(pfile, 'rb') as fin:
#     semseg = pickle.load(fin)

# print(np.arange(10))
print('Done')