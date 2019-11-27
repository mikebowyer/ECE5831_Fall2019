#!/usr/bin/env python
# title           :reloadModel.py
# description     :
# author          :Michael Bowyer
# date            :20191126
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
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from matplotlib import pyplot as plt


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
    description='this script reloads a keras model.')
parser.add_argument('--input', '-i', type=str, required=True,
                    help='Input file of where model was saved in .hdf5 format')
# parser.add_argument('--output', '-o', required=True,
#                     help='The name of the model to be saved')
# parser.add_argument('--prev_months', '-pm', required=True,
#                     help='How many months in the past should be used for training the model')
# parser.add_argument('--future_months', '-fm', required=True,
#                     help='How many months in the future to predict (1=Only predict current month, and no future months)')

if __name__ == "__main__":
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    from keras import backend as K
    K.tensorflow_backend._get_available_gpus()
    """ Parse Args """
    args = parser.parse_args()
    logging.info('Using input model ' + str(args.input))

    model = load_model(args.input)
    model.summary()

    # """ Reload model """
    # weights_file = 'Weights-046--800.24624.hdf5'  # choose the best checkpoint
    # myModel.load_weights(weights_file)  # load it
    # myModel.compile(loss='mean_absolute_error',
    #                 optimizer='adam', metrics=['mean_absolute_error'])

    # predictions = myModel.predict(trainingDf)
    # print(predictions)
