from pandaset import DataSet
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf

# Importing dataset
dataset = DataSet('C:/Users/maste/4AI3/PandaSet')

# Determining sequences that have semantic segmentation labels
sequences = dataset.sequences(with_semseg=True)
# print(sequences)

data = dataset[sequences[1]]

lidar_data = []

# Loading dataset into DataFrame
for seq in sequences:
    for i in range(80):
        if i < 10:
            point = '0' + str(i)
        else:
            point = str(i)
        pfile_semseg = 'C:/Users/maste/4AI3/PandaSet/{}/annotations/semseg/{}.pkl'.format(seq, point)
        pfile_data = 'C:/Users/maste/4AI3/PandaSet/{}/lidar/{}.pkl'.format(seq, point)
        with open(pfile_semseg, 'rb') as fin:
            semseg = pickle.load(fin)
        with open(pfile_data, 'rb') as fin:
            data = pickle.load(fin)

        temp = data
        temp['class'] = semseg
        lidar_data.append(temp)


# pfile = 'C:/Users/maste/4AI3/PandaSet/001/annotations/semseg/05.pkl'
# with open(pfile, 'rb') as fin:
#     semseg = pickle.load(fin)

print('Done')