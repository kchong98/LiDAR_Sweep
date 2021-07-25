import numpy as np
import tensorflow as tf
import time
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors


class RandLANet(tf.keras.models.Model):
    def __init__(self, data, KNN, blocks):
        super(RandLANet, self).__init__()
        self.data = data
        self.neighbors = KNN
        self.blocks = blocks


class DilatedResBlock:
    """
    For local feature aggregation of 3D point cloud
    """
    def __init__(self):
        return 1

    def LoCSE(self, data):
        # Subsampling data
        sub_data = data.sample(frac=0.25)
        point_data = sub_data[['x', 'y', 'z']]
        knn = NearestNeighbors(n_neighbors=16)
        knn.fit(sub_data)
        r_k = []
        for point in point_data:
            r_k.append(knn.kneighbors(point, return_distance=True))
        return 1

    def attentivePooling(self):
        return 1


class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Encoder, self).__init__()

    def call(self, inputs):
        self.output = inputs
        return self.outputs


class Decoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Decoder, self).__init__()


if __name__ == "__main__":
    print('Done')