#!/usr/bin/env python
# title           :simpleNet.py
# description     :
# author          :Michael Bowyer
# date            :20191123
# version         :0.0
# usage           :
# notes           :
# python_version  :Python 3.7.3
# ==============================================================================
import logging
import math
import argparse
import os
import pandas as pd

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from matplotlib import pyplot as plt
import create_train_test_dfs as cttds

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings


""" Setup logging config """
logging.basicConfig(level=logging.DEBUG, filemode='w',
                    format='%(levelname)s:: %(message)s')  # ,filename='app.log')
# logging.debug('This is a debug message')

""" Input Arguments """
parser = argparse.ArgumentParser(
    description='this script does what?.')
parser.add_argument('--input', '-i', type=str, required=True,
                    help='input training dataset with corresponding target values')
parser.add_argument('--output_model_name', '-o', required=True,
                    help='The name of the model to be saved')
parser.add_argument('--prev_months', '-pm', required=True,
                    help='How many months in the past should be used for training the model')
parser.add_argument('--future_months', '-fm', required=True,
                    help='How many months in the future to predict (1=Only predict current month, and no future months)')


if __name__ == "__main__":
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    from keras import backend as K
    K.tensorflow_backend._get_available_gpus()
    """ Parse Args """
    args = parser.parse_args()
    logging.info('Using input file ' + str(args.input))
    logging.info('Each row in the output file will contain ' +
                 str(args.prev_months) + ' months of  worth of data, and will contain ZHVI values of the current month and ' +
                 str(int(args.future_months)-1) + ' months in the future.')

    outputDir = os.path.dirname(args.output_model_name)
    outputModelName = os.path.basename(args.output_model_name)
    if outputModelName == '':
        pass
    else:
        outputDir = outputDir + '/' + outputModelName

    if not os.path.exists(outputDir):
        os.mkdir(outputDir)
    logging.info(
        "Output model will be saved to following folder: " + outputDir)

    """ Read in input data """
    inputDf = pd.read_csv(args.input)

    trainingDf = cttds.create_training_df(inputDf)
    targetDf = cttds.create_target_df(inputDf)

    """ Define Model """
    myModel = Sequential()
    myModel.add(Dense(trainingDf.shape[1], kernel_initializer='normal',
                      input_dim=trainingDf.shape[1], activation='relu'))
    myModel.add(
        Dense(math.ceil(trainingDf.shape[1]/2), kernel_initializer='normal', activation='relu'))
    myModel.add(
        Dense(math.ceil(trainingDf.shape[1]/4), kernel_initializer='normal', activation='relu'))
    myModel.add(
        Dense(2, kernel_initializer='normal', activation='linear'))
    myModel.compile(loss='mean_absolute_error', optimizer='adam',
                    metrics=['mean_absolute_error'])
    myModel.summary()

    """ Create checkpoints """
    checkpoint_name = outputDir + \
        'best_weights.hdf5'  # _experiments/Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
    checkpoint = ModelCheckpoint(
        checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]

    """ Train Model """
    myModel.fit(trainingDf, targetDf, epochs=5, batch_size=32,
                validation_split=0.2, callbacks=callbacks_list)

    """ Save Model """
    logging.info(
        "Saving final model structure and best weights to: " + outputDir)
    myModel.save(outputDir + "/model_and_best_wieghts.hdf5")
    # """ Reload model """
    # weights_file = 'Weights-046--800.24624.hdf5'  # choose the best checkpoint
    # myModel.load_weights(weights_file)  # load it
    # myModel.compile(loss='mean_absolute_error',
    #                 optimizer='adam', metrics=['mean_absolute_error'])

    # predictions = myModel.predict(trainingDf)
    # print(predictions)
