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

import create_train_test_dfs as cttds
import warnings
import numpy as np
import matplotlib.pyplot as plt

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
from tensorflow.python.client import device_lib
from keras import backend as K

""" Setup logging config """
logging.basicConfig(level=logging.DEBUG, filemode='w',
                    format='%(levelname)s:: %(message)s')  # ,filename='app.log')
# logging.debug('This is a debug message')

""" Input Arguments """
parser = argparse.ArgumentParser(
    description='this script reloads a keras model.')
parser.add_argument('--model_to_reload', '-m', type=str, required=True,
                    help='Input file of where model was saved in .hdf5 format')
parser.add_argument('--input', '-i', type=str, required=True,
                    help='input training dataset with corresponding target values')
parser.add_argument('--use_gpu', '-gpu', action='store_true',
                    help='Use this argument when you would like to use GPU.')
# parser.add_argument('--output', '-o', required=True,
#                     help='The name of the model to be saved')
# parser.add_argument('--prev_months', '-pm', required=True,
#                     help='How many months in the past should be used for training the model')
# parser.add_argument('--future_months', '-fm', required=True,
#                     help='How many months in the future to predict (1=Only predict current month, and no future months)')

if __name__ == "__main__":

    """ Parse Args """
    args = parser.parse_args()
    logging.info('Using input file ' + str(args.input))
    logging.info('Using input model ' + str(args.model_to_reload))

    """ determine if user wants to use GPU """
    if(args.use_gpu):
        GPU_info = K.tensorflow_backend._get_available_gpus()
        logging.info('Using GPU with following information: ' + str(GPU_info))
    else:
        logging.info('Using CPU instead of GPU')

    """ Load model """
    logging.info(
        'Loading model from ' + str(args.model_to_reload) + ' With summary:')
    model = load_model(args.model_to_reload)
    model.summary()
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    logging.info(short_model_summary)

    """ Read in input data """
    logging.info(
        'Reading in input data and creating training and target values from ' + str(args.input))
    inputDf = pd.read_csv(args.input)
    trainingDf = cttds.create_training_df(inputDf)
    targetDf = cttds.create_target_df(inputDf)

    """ Predict all training examples """
    logging.info(
        'Generating Prediction for all training examples')
    predictions = model.predict(trainingDf)
    predictionDf = pd.DataFrame(predictions)
    # print(predictions)
    numFuturePredictions = targetDf.shape[1]
    print(numFuturePredictions)

    """ Create Data frame with Target values and predicted values """
    logging.info(
        'Combining predictions with target values')
    evaluationDf = inputDf[['Date', 'ZillowNeighborhood']]
    for i in range(0, numFuturePredictions):
        predStr = 'pred_ZHVI_t' + str(i)
        targStr = 'ZHVI_t' + str(i)
        evaluationDf[predStr] = predictionDf.iloc[:, i]
        evaluationDf[targStr] = targetDf.iloc[:, i]
        # print(predictionDf.iloc[:, i])
        # print(targetDf.iloc[:, i])
    print(evaluationDf.head())

    """ Save infered information"""
    outputFilename = args.model_to_reload + '_inferenceAndTarget.csv'
    evaluationDf.to_csv(
        outputFilename, index=None, header=True)
    logging.info("Saving inferences to " + outputFilename)
