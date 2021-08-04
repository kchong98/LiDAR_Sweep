import numpy as np
import pandas as pd
import tensorflow as tf
import pptk
import json
import argparse
import pickle
import helper_tools
from sklearn.metrics import jaccard_score, accuracy_score

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

    n_samples = 10000

    if mode == 'train':
        # Loading dataset
        train_data, test_data = helper_tools.load_data(n_samples=n_samples)
        model = helper_tools.create_model(n_samples)
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=5, restore_best_weights=True)
        model.fit(train_data, epochs=100, verbose=2, batch_size=1, validation_data=test_data, callbacks=[callback]) #, callbacks=[callback]
        model.save_weights('./saved_models/checkpoints')

        true = list(train_data.as_numpy_iterator())
        test = true[0][0]
        true = true[0][1]
        # print(test.shape)
        predict = model.predict(test)
        true = (np.argmax(true, axis=2) + 1).flatten()
        predict = (np.argmax(predict, axis=2) + 1).flatten()
        print('Mean IoU: {}'.format(np.average(jaccard_score(y_true = true, y_pred = predict, average=None))))
        print('Accuracy: {}'.format(accuracy_score(y_true=true, y_pred = predict)))

        true = pd.DataFrame(true, columns=[['class']])
        predict = pd.DataFrame(predict, columns=['class'])
        test = np.squeeze(test)
        data = pd.DataFrame(test, columns=[['x','y','z']])
        with open('true.pk', 'wb') as f:
            pickle.dump(true, f)        
        with open('predictions.pk', 'wb') as f:
            pickle.dump(predict, f)
        with open('data.pk', 'wb') as f:
            pickle.dump(data, f)

    elif mode == 'test':
        _, test_data = helper_tools.load_data(n_scenes = 1, n_train_sweeps = 0, n_test_sweeps = 1, n_samples = n_samples)
        model = helper_tools.create_model(n_samples)
        model.load_weights('./saved_models/checkpoints')
        predict = model.predict(test_data)
        true = list(test_data.as_numpy_iterator())
        true = true[0][1]
        true = (np.argmax(true, axis=2) + 1).flatten()
        predict = (np.argmax(predict, axis=2) + 1).flatten()
        print(true[:5])
        print(predict[:5])
        print('Mean IoU: {}'.format(np.average(jaccard_score(y_true = true, y_pred = predict, average=None))))
        print('Accuracy: {}'.format(accuracy_score(y_true=true, y_pred = predict)))
        print('Done')

    elif mode =='sum':
        model = helper_tools.create_model(n_samples)
        model.load_weights('./saved_models/local_machine/checkpoints')
        model.summary()

    elif mode =='vis':
        # Loading classes from json file
        with open('classes.json') as f:
            class_dict = json.load(f)

        # Loading color map from json file
        with open('color.json') as f:
            color_map = json.load(f)      

        # Loading data from pk file
        with open('data.pk', 'rb') as fin:
            data = pickle.load(fin)   

        # Loading predictons from pk file
        with open('predictions.pk', 'rb') as fin:
            predict = pickle.load(fin)  

        # Loading groud truth classifications from pk file
        with open('true.pk', 'rb') as fin:
            true = pickle.load(fin)    

        # Visualising point-cloud in 3D space
        rgb = np.zeros((len(data.index), 3))
        row = 0
        for a in predict.to_numpy():
            rgb[row,:] = np.reshape(np.array(color_map[class_dict[str(int(a))]]), (1,3))/255
            row = row + 1
        v = pptk.viewer(data[['x','y','z']].to_numpy(), rgb)
        v.color_map('cool', scale=[0, 42])
        print('Done')

    else:
        print('Invalid mode \"{}\"'.format(mode))