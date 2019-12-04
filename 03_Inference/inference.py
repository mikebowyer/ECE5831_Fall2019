#!/usr/bin/env python
# title           : inference.py
# description     :
# author          : Michael Bowyer
# date            : 20191126
# version         : 0.0
# usage           : This script reloads a trained keras model, and runs inference (prediction) on the input data.
# python_version  :Python 3.7.3
# ==============================================================================

import logging
import argparse
import os
import pandas as pd

import create_train_target_dfs as cttds
import tensorflow as tf
from keras.models import load_model
from keras import backend as K

""" Setup logging config """
log = logging.basicConfig(level=logging.INFO, filemode='w',
                          format='%(levelname)s:: %(message)s')  # ,filename='app.log')
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

""" Input Arguments """
parser = argparse.ArgumentParser(
    description='This script reloads a trained keras model, and runs inference (prediction) on the input data. ')
parser.add_argument('--model_to_reload', '-m', type=str, required=True,
                    help='Input file of where model was saved in .hdf5 format')
parser.add_argument('--input', '-i', type=str, required=True,
                    help='input training dataset with corresponding target values')
parser.add_argument('--use_gpu', '-gpu', action='store_true',
                    help='Use this argument when you would like to use GPU.')

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
        'Reading in input data and creating input feature vectors and target value vectors from ' + str(args.input))
    inputDf = pd.read_csv(args.input)
    trainingDf = cttds.create_training_df(inputDf)
    targetDf = cttds.create_target_df(inputDf)

    """ Predict all input feature examples """
    logging.info(
        'Generating Prediction for input feature examples')
    predictions = model.predict(trainingDf)
    predictionDf = pd.DataFrame(predictions)
    numFuturePredictions = targetDf.shape[1]

    """ Create Data frame with Target + predicted values, Absolute Error, and Percentage Error """
    logging.info(
        'Combining predictions with target values')
    evaluationDf = inputDf[['Date', 'ZillowNeighborhood']]
    for i in range(0, numFuturePredictions):
        """ Create new strings for new column headers """
        predStr = 'pred_ZHVI_t' + str(i)
        targStr = 'ZHVI_t' + str(i)
        AbsErr = 'AbsErr_t' + str(i)
        AbsPercentErr = 'AbsPercentErr_t' + str(i)

        """ Generate predicted and target data in form of dict """
        newColumnsDict = {}
        newColumnsDict[predStr] = list(predictionDf.iloc[:, i].values)
        newColumnsDict[targStr] = list(targetDf.iloc[:, i].values)
        newColDf = pd.DataFrame(newColumnsDict)
        evaluationDf = pd.concat([evaluationDf, newColDf], axis=1)

        """ Add in evaluation metrics columns """
        evaluationDf[AbsErr] = abs(
            evaluationDf[predStr] - evaluationDf[targStr])
        evaluationDf[AbsPercentErr] = abs(
            (evaluationDf[predStr] - evaluationDf[targStr])/evaluationDf[targStr])

    """ Save infered information"""
    outputdir = os.path.split(args.model_to_reload)[0]
    inputDir, inputFile = os.path.split(args.input)
    inputFileBase = os.path.splitext(inputFile)[0]
    outputdir = outputdir + '\\predictions\\'
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    outputFilename = outputdir + \
        inputFileBase + '_PredictedAndTarget_ZHVI.csv'

    logging.info("Saving predicted and target values to " + outputFilename)
    evaluationDf.to_csv(
        outputFilename, index=None, header=True)
