import numpy as np
import pandas as pd
import tensorflow as tf
import pptk
import argparse
import helper_tools
from sklearn.metrics import jaccard_score

if __name__ == "__main__":
    # Declaring argument parser for command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the number of GPUs to use [default: 0]')
    parser.add_argument('--mode', type=str, default='vis', help='options: train, test, vis')
    parser.add_argument('--test_area', type=str, default='02', help='options: 08,11,12,13,14,15,16,17,18,19,20,21')
    parser.add_argument('--model_path', type=str, default='/saved_models/point_net.h5', help='pretrained model path')
    parser.add_argument('--scene', type=int, default=1, help='select what scene to test/train on')
    FLAGS = parser.parse_args()

    mode = FLAGS.mode

    n_samples = 30000

    if mode == 'train':
        # Loading dataset
        train_data, test_data = helper_tools.load_data(n_samples=n_samples)
        model = helper_tools.create_model(n_samples)
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=2, restore_best_weights=True)
        model.fit(train_data, epochs=50, verbose=2, batch_size=1, validation_data=test_data, callbacks=[callback]) #, callbacks=[callback]
        # model.save('./saved_models/point_net.h5')
        model.save_weights('./saved_models/checkpoints')

        predict = model.predict(test_data)
        # print(predict.shape)
        # print(predict)

    elif mode == 'test':
        _, test_data = helper_tools.load_data(n_scenes = 1, n_train_sweeps = 0, n_test_sweeps = 1, n_samples = n_samples)
        model = helper_tools.create_model(n_samples)
        model.load_weights('./saved_models/checkpoints')
        predict = model.predict(test_data)
        # print(predict.shape)
        true = list(test_data.as_numpy_iterator())
        true = true[0][1]

        # print(predict.shape)
        # print(true.shape)
        true = (np.argmax(true, axis=2) + 1).flatten()
        predict = (np.argmax(predict, axis=2) + 1).flatten()
        print(true[:5])
        print(predict[:5])
        print('Mean IoU: {}'.format(np.average(jaccard_score(y_true = true, y_pred = predict, average=None))))
        print('Done')

    elif mode =='vis':
        print('Done')

    else:
        print('Invalid mode \"{}\"'.format(mode))