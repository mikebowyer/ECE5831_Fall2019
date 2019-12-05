#!/usr/bin/env python
# title           :coneModel.py
# description     :This class takes input crime data and converts them to zillow neighborhoods
# author          :Michael Bowyer
# date            :20191130
# version         :0.0
# notes           :
# python_version  :Python 3.7.3
# ==============================================================================
import math
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
import numpy as np


def generateModel(inputTrainingShape, inputTargetShape):
    model = Sequential()

    model.add(Dense(inputTrainingShape[1], kernel_initializer='normal',
                    input_dim=inputTrainingShape[1], activation='relu'))
    model.add(Dropout(0.2, input_shape=(inputTrainingShape[1],)))
    model.add(
        Dense(math.ceil(inputTrainingShape[1]/2), kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(
        Dense(math.ceil(inputTrainingShape[1]/4), kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.05))
    model.add(
        Dense(inputTargetShape[1], kernel_initializer='normal', activation='linear'))

    return model
