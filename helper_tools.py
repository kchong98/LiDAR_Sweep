import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import OneHotEncoder
import random

def load_data(n_scenes = 3, n_train_sweeps = 5, n_test_sweeps = 3):
    """
    Function to load data from PandaSet

    n_scenes: int
        number of scenes to sample from
    n_train_sweeps: int
        number of sweeps to include in train dataset
    n_test_sweeps: int
        number of sweeps to include in test dataset

    returns: training dataset, testing dataset
    """
    all_classes = np.arange(0,43, dtype=int).reshape(-1,1)
    semseg_files = ['001', '002', '003', '005', '011', '013', '015', '016', '017', '019', '021', '023', '024', '027', '028', '029', '030', '032', '033', '034', '035', '037', '038', '039', '040', '041', '042', '043', '044', '046']
    X_train, X_test, y_train, y_test = [], [], [], []
    oneHot = OneHotEncoder(sparse = False)
    oneHot.fit(all_classes)

    train_sweeps = random.sample(range(79), n_train_sweeps)
    test_sweeps = random.sample(range(79), n_test_sweeps)

    for scene in semseg_files[:n_scenes]:
        for a in train_sweeps:
            if a < 10:
                sweep = '0' + str(a)
            else:
                sweep = str(a)
            
            pfile_seg = './dataset/{}/annotations/semseg/{}.pkl'.format(scene, sweep)
            pfile_lid = './dataset/{}/lidar/{}.pkl'.format(scene, sweep)
            with open(pfile_seg, 'rb') as fin:
                label = pickle.load(fin)
            with open(pfile_lid, 'rb') as fin:
                lidar = pickle.load(fin)
            
            temp = lidar
            temp['class'] = label
            temp = temp.sample(2000)

            # print(temp['class'].unique())

            X_train.append(temp[['x','y','z']].to_numpy())
            y_train.append(oneHot.transform(temp['class'].to_numpy().reshape(-1,1)))

        for a in test_sweeps:
            if a < 10:
                sweep = '0' + str(a)
            else:
                sweep = str(a)

            pfile_seg = './dataset/{}/annotations/semseg/{}.pkl'.format(scene, sweep)
            pfile_lid = './dataset/{}/lidar/{}.pkl'.format(scene, sweep)
            with open(pfile_seg, 'rb') as fin:
                label = pickle.load(fin)
            with open(pfile_lid, 'rb') as fin:
                lidar = pickle.load(fin)

            temp = lidar
            temp['class'] = label
            temp = temp.sample(2000)

            X_test.append(temp[['x','y','z']].to_numpy())
            y_test.append(oneHot.transform(temp['class'].to_numpy().reshape(-1,1)))                

    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    return train_data, test_data

def conv_layer(x, n_filters):
    x = tf.keras.layers.Conv1D(n_filters, kernel_size=1, padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU()(x)

def dense_layer(x, neurons):
    x = tf.keras.layers.Dense(neurons)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU()(x)

class Regularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.features, self.features))
        xxt = tf.tensordot(x, x, axes = (2, 2))
        xxt = tf.reshape(xxt, (-1, self.features, self.features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

def t_net(inputs, features):
    bias = tf.keras.initializers.Constant(np.eye(features).flatten())
    reg = Regularizer(features)

    x = conv_layer(inputs, 32)
    x = conv_layer(x, 64)
    x = conv_layer(x, 512)
    x = tf.keras.layers.GlobalMaxPool1D()(x)
    x = dense_layer(x, 256)
    x = dense_layer(x, 128)
    x = tf.keras.layers.Dense(features*features, 
                    kernel_initializer='zeros', 
                    bias_initializer=bias, 
                    activity_regularizer=reg)(x)
    features_T = tf.keras.layers.Reshape((features, features))(x)
    return tf.keras.layers.Dot(axes = (2, 1))([inputs, features_T])

def NLLLoss(y_test, y_pred):
    """
    Negative log likelihood custom loss function for PointNet
    y_test: true values
    y_pred: predicted values
    """
    y_pred_mean = tf.reduce_mean(y_pred)
    y_pred_sd = tf.reduce_std(y_pred)
    square = tf.square(y_pred_mean - y_test)
    ms = tf.add(tf.divide(square, y_pred_sd), tf.log(y_pred_sd))
    ms = tf.reduce_mean(ms)
    return (ms)

def create_model(num_points):
    """
    Declaring model
    num_points: number of inputs points for input
    """
    inputs = tf.keras.layers.Input(shape=(num_points, 3))
    x = t_net(inputs, 3)
    x = conv_layer(x, 32)
    x = conv_layer(x, 32)
    t = t_net(x, 32)
    x = conv_layer(t, 32)
    x = conv_layer(x, 64)
    x = conv_layer(x, 512)
    concat = tf.keras.layers.Concatenate()([t, x])
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = conv_layer(concat, 512)
    x = conv_layer(x, 256)
    x = conv_layer(x, 128)

    x = conv_layer(x, 128)
    outputs = tf.keras.layers.Conv1D(1, kernel_size = 1, padding = 'valid')(x)

    # x = dense_layer(x, 256)
    # x = layers.Dropout(0.25)(x)
    # x = dense_layer(x, 128)
    # x = layers.Dropout(0.25)(x)
    # outputs = layers.Dense(42, activation = 'softmax')(x)

    model = tf.keras.models.Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer = 'adam', loss = NLLLoss, metrics = [tf.metrics.MeanIoU(num_classes=43)])
    return model

if __name__ == "__main__":
    train_data, test_data = load_data()
    print('Done')