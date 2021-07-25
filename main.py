from pandaset import DataSet
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import pptk

# Importing dataset
dataset = DataSet('C:/Users/maste/4AI3/PandaSet')

# Determining sequences that have semantic segmentation labels
sequences = dataset.sequences(with_semseg=True)
# print(sequences)

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

seq_loaded = dataset[j]
seq_loaded.load()
class_dict = {}

# Saving classes to dict
for key in seq_loaded.semseg.classes:
    class_dict[key] = seq_loaded.semseg.classes[key]

# Saving classes to json file
with open('classes.json', 'w') as fp:
    json.dump(class_dict, fp)

# Loading color map from json file
with open('color.json') as f:
    color_map = json.load(f)

# Loading pkl file into DataFrame
with open(pfile_semseg, 'rb') as fin:
    semseg = pickle.load(fin)
with open(pfile_data, 'rb') as fin:
    data = pickle.load(fin)

# Visualising point-cloud in 3D space
rgb = np.zeros((len(data.index), 3))
row = 0
for a in semseg.to_numpy():
    rgb[row,:] = np.reshape(np.array(color_map[class_dict[str(int(a))]]), (1,3))/255
    row = row + 1
# v = pptk.viewer(data[['x','y','z']].to_numpy(), rgb)
# v.color_map('cool', scale=[0, 42])
# v.color_map([[0, 0, 0], [1, 1, 1]])

# Preparing data
class oneHot:
    def __init__(self):
        print('Created One Hot Encoder')

    def fit(self, classes):
        self.classes = classes

    def transform(self, data):
        encoded = np.zeros((len(data.index), self.classes))
        row = 0
        for point in data.iloc[:,0]:
            encoded[row, int(point)-1] = 1
            # print(point)
            row += 1
        return encoded

encoder = oneHot()
encoder.fit(42)
one_hot_semseg = encoder.transform(semseg)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split

data = data[['x','y','z']]
data = data.sample(frac=0.25)

# X_train, X_test, y_train, y_test = train_test_split(data, one_hot_semseg)

X_train, X_test = train_test_split(data)

visible = Input(shape=(data.shape[1],))
# Encoder 1
encoder = Dense(data.shape[1]*2)(visible)
encoder = BatchNormalization()(encoder)
encoder = LeakyReLU()(encoder)

# Encoder 2
encoder = Dense(data.shape[1])(encoder)
encoder = BatchNormalization()(encoder)
encoder = LeakyReLU()(encoder)

# Bottleneck
bottleneck = Dense(data.shape[1])(encoder)

# Decoder 1
decoder = Dense(data.shape[1])(bottleneck)
decoder = BatchNormalization()(decoder)
decoder = LeakyReLU()(decoder)

# Decoder 2
decoder = Dense(data.shape[1]*2)(decoder)
decoder = BatchNormalization()(decoder)
decoder = LeakyReLU()(decoder)

# Output
output = Dense(data.shape[1], activation='linear')(decoder)

model = Model(inputs=visible, outputs = output)
model.compile(optimizer='adam', loss='mse')

history = model.fit(X_train, X_train, epochs=15, batch_size=16, verbose=2, validation_data=(X_test, X_test))

predict = model.predict(X_test)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

print('Done')