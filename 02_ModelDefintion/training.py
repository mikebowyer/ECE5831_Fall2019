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
from tensorflow.python.client import device_lib
from keras import backend as K
import Models.ConeModel.coneModel as Cone

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings


""" Input Arguments """
parser = argparse.ArgumentParser(
    description='this script does what?.')
parser.add_argument('--input', '-i', type=str, required=True,
                    help='input training dataset with corresponding target values')
parser.add_argument('--output_model_name', '-o', required=True,
                    help='The name of the model to be saved')
# parser.add_argument('--prev_months', '-pm', required=True,
#                     help='How many months in the past should be used for training the model')
# parser.add_argument('--future_months', '-fm', required=True,
#                     help='How many months in the future to predict (1=Only predict current month, and no future months)')
parser.add_argument('--use_gpu', '-gpu', action='store_true',
                    help='Use this argument when you would like to use GPU.')
parser.add_argument('--epochs', '-e', default=5,
                    help='However many epochs you would like to train for.')

if __name__ == "__main__":

    """ Parse Args """
    args = parser.parse_args()

    """ determine output directory/file formats """
    outputDir = os.path.dirname(args.output_model_name)
    outputModelName = os.path.basename(args.output_model_name)
    if outputModelName == '':
        pass
    else:
        outputDir = outputDir + '/' + outputModelName

    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    """ Setup logging config """
    output_log = outputDir + '/training.log'
    logging.basicConfig(level=logging.DEBUG, filemode='w',
                        format='%(levelname)s:: %(message)s', filename=output_log)
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.info('Using input training/target data file ' + str(args.input))
    logging.info('Number of training epochs selected: ' + str(args.epochs))
    # logging.info('Each row in the output file will contain ' +
    #              str(args.prev_months) + ' months of  worth of data, and will contain ZHVI values of the current month and ' +
    #              str(int(args.future_months)-1) + ' months in the future.')
    logging.info(
        "Output model will be saved to following folder: " + outputDir)

    """ determine if user wants to use GPU """
    if(args.use_gpu):
        GPU_info = K.tensorflow_backend._get_available_gpus()
        logging.info('Using GPU with following information: ' + str(GPU_info))
    else:
        logging.info('Using CPU instead of GPU')

    """ Read in input data """
    logging.info(
        'Reading in input data and creating training and target values from ' + str(args.input))
    inputDf = pd.read_csv(args.input)

    trainingDf = cttds.create_training_df(inputDf)
    targetDf = cttds.create_target_df(inputDf)

    """ Define Model """
    logging.info(
        'Model is now being defined and will be summarized below:')
    model = Cone.generateModel(trainingDf.shape)
    model.compile(loss='mean_absolute_error', optimizer='adam',
                  metrics=['mean_absolute_error'])

    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    logging.info(short_model_summary)

    """ Create checkpoints """
    checkpoint_name = outputDir + \
        'best_weights.hdf5'  # _experiments/Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
    checkpoint = ModelCheckpoint(
        checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]

    """ Train Model """
    logging.info(
        'Beginnging to traing model with ' + str(args.epochs) + ' epochs.')
    history = model.fit(trainingDf, targetDf, epochs=int(args.epochs), batch_size=32,
                        validation_split=0.2, callbacks=callbacks_list)

    """ Save Model """
    logging.info(
        "Saving final model structure and best weights to: " + outputDir)
    model.save(outputDir + "/model_and_best_wieghts.hdf5")

    """ Print evaluation scoring metrics and save it """
    plt.figure(figsize=(20.0, 15.0))
    plt.plot(history.history['mean_absolute_error'])
    plt.scatter(range(0, int(args.epochs)),
                history.history['mean_absolute_error'])
    plt.title(outputDir + ' Validation Error Verus Training Epoch')
    plt.xlabel('Training Epoch Number')
    plt.ylabel('Validation Mean Absolute Error')
    plt.savefig(outputDir + '/ValidationScoreVersusEpoch.png')
    plt.show()
