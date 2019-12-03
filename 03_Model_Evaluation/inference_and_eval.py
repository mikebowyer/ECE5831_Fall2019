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
import tensorflow as tf

from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from matplotlib import pyplot as plt
from tensorflow.python.client import device_lib
from keras import backend as K

""" Setup logging config """
log = logging.basicConfig(level=logging.INFO, filemode='w',
                          format='%(levelname)s:: %(message)s')  # ,filename='app.log')
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

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
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    """ Load model """
    logging.info(
        'Loading model from ' + str(args.model_to_reload) + ' With summary:')
    model = load_model(args.model_to_reload)

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
    numFuturePredictions = targetDf.shape[1]

    """ Create Data frame with Target values and predicted values """
    logging.info(
        'Combining predictions with target values')
    evaluationDf = inputDf[['Date', 'ZillowNeighborhood']]
    for i in range(0, numFuturePredictions):
        """ Create new strings for new column headers """
        predStr = 'pred_ZHVI_t' + str(i)
        targStr = 'ZHVI_t' + str(i)
        MAE = 'MAE_t' + str(i)
        MAPE = 'MAPE_t' + str(i)

        """ Generate predicted and target data in form of dict """
        newColumnsDict = {}
        newColumnsDict[predStr] = list(predictionDf.iloc[:, i].values)
        newColumnsDict[targStr] = list(targetDf.iloc[:, i].values)
        newColDf = pd.DataFrame(newColumnsDict)
        evaluationDf = pd.concat([evaluationDf, newColDf], axis=1)

        """ Add in evaluation metrics columns """
        evaluationDf[MAE] = abs(evaluationDf[predStr] - evaluationDf[targStr])
        evaluationDf[MAPE] = abs(
            (evaluationDf[predStr] - evaluationDf[targStr])/evaluationDf[targStr])

    print(evaluationDf.describe())

    """ Save infered information"""
    outputFilename = args.model_to_reload + '_inferenceAndTarget.csv'
    evaluationDf.to_csv(
        outputFilename, index=None, header=True)
    logging.info("Saving inferences to " + outputFilename)

    """ Generate Error Metrics for overall Training data """
    EvalCols = evaluationDf.columns.values
    MAECols = [col for col in EvalCols if 'MAE' in col]
    MAPECols = [col for col in EvalCols if 'MAPE' in col]

    MAEMean = []
    MAEStd = []
    for MAECol in MAECols:
        MAEMean.append(evaluationDf[MAECol].mean())
        MAEStd.append(evaluationDf[MAECol].std())

    MAPEMean = []
    MAPEStd = []
    for MAPECol in MAPECols:
        MAPEMean.append(evaluationDf[MAPECol].mean())
        MAPEStd.append(evaluationDf[MAPECol].std())

    plt.subplot(121)
    plt.errorbar(MAECols, MAEMean, MAEStd, capsize=15,
                 capthick=3, barsabove=True, linestyle='None')
    plt.xlabel('Predicted Months (tx, with x=number of months in future)')
    plt.ylabel('Mean Absolute Error')
    plt.title('Overall Training Mean Asolute Error Mean and Standard Deviations')
    plt.subplot(122)
    plt.errorbar(MAPECols, MAPEMean, MAPEStd, capsize=15,
                 capthick=3, barsabove=True, linestyle='None')
    plt.xlabel('Predicted Months (tx, with x=number of months in future)')
    plt.ylabel('Mean Absolute Percentage Error')
    plt.title(
        'Overall Training Mean Asolute Percentage Error Mean and Standard Deviations')
    plt.show()

    # """ Find Errors Per Neighborhoods """
    # neighborhoods = list(evaluationDf['ZillowNeighborhood'].unique())
    # for neighborhood in neighborhoods:
    #     neighborhoodDF = evaluationDf[evaluationDf['ZillowNeighborhood']
    #                                   == neighborhood]
    #     MAPEMean = []
    #     MAPEStd = []
    #     for MAPECol in MAPECols:
    #         MAPEMean.append(neighborhoodDF[MAPECol].mean())
    #         MAPEStd.append(evaluationDf[MAPECol].std())
    #     print(MAPEMean)
    #     print(MAEMean)
    # # plt.scatter(Albany['Date'],Albany['ZHVI_t0'])
