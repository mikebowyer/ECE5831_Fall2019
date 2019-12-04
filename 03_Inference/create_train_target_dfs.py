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


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings


def create_training_df(inputDf):
    cols = inputDf.shape[1]
    currentZHVIColNum = inputDf.columns.get_loc("ZHVI_t0")

    trainingDf = inputDf.copy()

    trainingDf.drop(
        trainingDf.columns[currentZHVIColNum:cols], inplace=True, axis=1)
    trainingDf.drop(
        axis=1, columns=['Date', 'ZillowNeighborhood'], inplace=True)
    return trainingDf


def create_target_df(inputDf):
    currentZHVIColNum = inputDf.columns.get_loc("ZHVI_t0")
    targetDf = inputDf.copy()

    targetDf.drop(
        targetDf.iloc[:, :currentZHVIColNum], inplace=True, axis=1)

    return targetDf
