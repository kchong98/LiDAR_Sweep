import numpy as np
import tensorflow as tf
import time
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors


class RandLANet():
    def __init__(self, data, KNN, blocks):
        self.data = data
        self.neighbors = KNN
        self.blocks = blocks

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

    def attentive_pooling(self):
        return 1

    def DRB(self):
        return 1


if __name__ == "__main__":
    print('Done')