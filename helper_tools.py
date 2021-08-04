import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import OneHotEncoder
import random

def fps(data, n_samples):
    """
    Farthest point sampling of point cloud data

    data: input point cloud
    n_samples: number of samples to return

    returns: farthest point sampled data
    """
    columns = ['x','y','z','class']
    data = data[columns]
    if (str(type(data)) == '<class \'pandas.core.frame.DataFrame\'>'):
        data = data.to_numpy()

    samples = np.empty((n_samples, 4))
    for i in range(n_samples):
        distances = []
        center_point_idx = np.random.randint(0,data.shape[0], size = 1)
        for a in data:
            distances.append(np.linalg.norm(data[center_point_idx,:3]-a[:3]))
        samples[i,:] = data[distances.index(max(distances)), :]
        if i % 100 == 0:
            print('Almost!')

    samples = pd.DataFrame(samples, columns=columns)
    return samples

def load_data(n_scenes = 2, n_train_sweeps = 15, n_test_sweeps = 3, n_samples = 16000):
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
            # temp = fps(temp, n_samples)
            temp = temp.sample(n_samples)

            # print(temp['class'].unique())

            X_train.append(temp[['x','y','z']].to_numpy().reshape(1,-1,3))
            y_train.append(oneHot.transform(temp['class'].to_numpy().reshape(-1,1)).reshape(1,-1,43))

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
            # temp = fps(temp, n_samples)
            temp = temp.sample(n_samples)

            X_test.append(temp[['x','y','z']].to_numpy().reshape(1,-1,3))
            y_test.append(oneHot.transform(temp['class'].to_numpy().reshape(-1,1)).reshape(1,-1,43))                
    
    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

    if __name__ == '__main__':
        print("-"*20)
        print(X_train.shape, X_test.shape)
        print(y_train.shape, y_test.shape)
        print("-"*20)

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

    returns: loss value
    """
    y_pred_mean = tf.reduce_mean(y_pred)
    y_pred_sd = tf.math.reduce_std(y_pred)
    square = tf.square(y_pred_mean - y_test)
    ms = tf.add(tf.divide(square, y_pred_sd), tf.math.log(y_pred_sd))
    ms = tf.reduce_mean(ms)
    return (ms)

def dice_loss(y_test, y_pred):
    y_test = tf.cast(y_test, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_test * y_pred)
    denominator = tf.reduce_sum(y_test + y_pred)
    loss = 1 - (numerator/denominator)
    return loss

def create_model(num_points):
    """
    Declaring model

    num_points: number of inputs points for input

    returns: tensorflow model
    """
    inputs = tf.keras.layers.Input(shape=(num_points,3))
    x = t_net(inputs, 3)
    x = conv_layer(x, 32)
    x = conv_layer(x, 32)
    x = t_net(x, 32)
    x = conv_layer(x, 32)
    t = conv_layer(x, 64)
    x = conv_layer(t, 512)
    # concat = tf.keras.layers.Concatenate()([t, x])
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.tile(tf.reshape(x, [1, 1,512]), [1, num_points,1], tf.int32)
    # print('-'*10)
    # print(x.shape)
    # print('-'*10)
    concat = tf.keras.layers.Concatenate(axis = 2)([t, x])
    # print('-'*10)
    # print(concat.shape)
    # print('-'*10)    
    x = conv_layer(concat, 512)
    x = conv_layer(x, 256)
    x = conv_layer(x, 128)

    x = conv_layer(x, 128)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Conv1D(1, kernel_size = 1, padding = 'valid')(x)
    x = dense_layer(x, 256)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = dense_layer(x, 128)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(43, activation='softmax')(x)
    
    # outputs = tf.reshape(outputs, [-1,1])

    model = tf.keras.models.Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer = 'adam', loss = dice_loss, metrics = ['accuracy']) #tf.metrics.MeanIoU(num_classes=43)
    return model

if __name__ == "__main__":
    n_samples = 80000

    train_data, test_data = load_data(n_samples = n_samples)
    model = create_model(n_samples)
    model.summary()
    print('Done')