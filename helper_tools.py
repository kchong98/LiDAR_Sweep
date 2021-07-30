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

def create_model():
    return model

if __name__ == "__main__":
    train_data, test_data = load_data()
    print('Done')