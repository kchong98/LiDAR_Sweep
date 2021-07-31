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
    # parser.add_argument('--test_area', type=str, default='14', help='options: 08,11,12,13,14,15,16,17,18,19,20,21')
    parser.add_argument('--model_path', type=str, default='/saved_models/point_net.h5', help='pretrained model path')
    parser.add_argument('--scene', type=int, default=1, help='select what scene to test/train on')
    FLAGS = parser.parse_args()

    mode = FLAGS.mode

    n_samples = 10000
    # Loading dataset
    train_data, test_data = helper_tools.load_data(n_samples=n_samples)

    if mode == 'train':
        model = helper_tools.create_model(n_samples)
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, restore_best_weights=True)
        model.fit(train_data, epochs=50, verbose=2, batch_size=1, validation_data=test_data, callbacks=[callback]) #, callbacks=[callback]
        # model.save('./saved_models/point_net.h5')
        model.save_weights('./saved_models/checkpoints')

        predict = model.predict(test_data)
        print(predict.shape)
        print(predict)

    elif mode == 'test':
        model = helper_tools.create_model(n_samples)
        model.load_weights('./saved_models/checkpoints')
        model.summary()
        predict = model.predict(test_data)
        print(predict.shape)
        print('Done')

    elif mode =='vis':
        print('Done')

    else:
        print('Invalid mode \"{}\"'.format(mode))