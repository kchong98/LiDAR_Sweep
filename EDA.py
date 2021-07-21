from pandaset import DataSet
import plotly.graph_objects as go
import csv
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import random
import os

dataset = DataSet('C:/Users/maste/4AI3/PandaSet')
seq002 = dataset['002']
seq002.load()

pfile = 'C:/Users/maste/4AI3/PandaSet/001/lidar/05.pkl'
with open(pfile, 'rb') as fin:
    data = pickle.load(fin)

print(data.head())
print('----------------------------------------------')
print(data.info())

pfile = 'C:/Users/maste/4AI3/PandaSet/001/annotations/semseg/05.pkl'
with open(pfile, 'rb') as fin:
    semseg = pickle.load(fin)

print(semseg.head())
print('----------------------------------------------')
print(semseg.info())

# print(seq002.__dict__)

headerColor = 'rgb(116,0,4)'

annon_class = []
for key in seq002.semseg.classes:
    annon_class.append(seq002.semseg.classes[key])

# fig = go.Figure(data=[go.Table(
#     header=dict(values=['<b>Semantic Segmentation Annotations for Sequence 002</b>'],
#                 line_color='darkslategray',
#                 fill_color=headerColor,
#                 align=['left', 'center'],
#                 font=dict(color='white', size=12)
#                 ),
#     cells=dict(values=[annon_class],
#                line_color='darkslategray',
#                align=['left', 'center'],
#                font=dict(color=headerColor, size=11)
#                ))
# ])
# fig.show()

lidar_poses = seq002.lidar._poses
# print(lidar_poses[0])

points3d_lidar_xyz = seq002.lidar.data


pfile = 'C:/Users/maste/4AI3/PandaSet/001/lidar/05.pkl'

with open(pfile, 'rb') as fin:
    data = pickle.load(fin)

print(data['d'].unique())

ax = plt.axes(projection='3d')

xyz = data[['x','y','z']].to_numpy()
color_maps = [(random.random(), random.random(), random.random()) for _ in range (1000+1)]

ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2])

# random subsampling of data
sub_data = data.sample(frac=0.25)
print(sub_data.info())


# figsize = plt.rcParams.get('figure.figsize')
# fig = plt.figure(figsize=(figsize[0] * 2, figsize[1]))
# ax1 = fig.add_subplot(1, 2, 1, projection = '3d')
# ax2 = fig.add_subplot(1, 2, 2, projection = '3d')
# ax1.axis("off")
# ax1.view_init(90, -90) # front view
# ax1.scatter(xyz[:,0], xyz[:,1], xyz[:,2])
# ax2.axis("off")
# ax2.view_init(90 + 90, -90) # top view
# ax2.scatter(xyz[:,0], xyz[:,1], xyz[:,2])
# plt.show()