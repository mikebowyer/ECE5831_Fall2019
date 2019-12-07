#!/usr/bin/env python
# title           : evalution.py
# description     : This script takes in already infered data and generates evaluation metrics
# author          : Michael Bowyer
# date            : 20191126
# version         : 0.0
# usage           : python evaluation.py --inference_data data.csv
# notes           :
# python_version  :Python 3.7.3
# ==============================================================================
import logging
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import numpy as np
from datetime import datetime

""" Setup logging config """
log = logging.basicConfig(level=logging.INFO, filemode='w',
                          format='%(levelname)s:: %(message)s')  # ,filename='app.log')
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

""" Input Arguments """
parser = argparse.ArgumentParser(
    description='This script takes in already infered data and generates evaluation metrics for the predicted ZHVI values')
parser.add_argument('--inference_data', '-id', type=str, required=True,
                    help='Input csv file which contains infered and error data from some ZHVI dataset')
parser.add_argument('--output_dir', '-o', type=str, default='',
                    help='Output directory where output graphs and evaluations will be stored')

if __name__ == "__main__":

    """ Turn off matplot lib output """
    matplotlib.set_loglevel("warning")
    """ Parse Args """
    args = parser.parse_args()
    logging.info('Using input inferece file to generate metrics: ' +
                 str(args.inference_data))
    saveImages = False
    if not (args.output_dir == ''):
        inputDir, inputFile = os.path.split(args.inference_data)
        inputFileBase = os.path.splitext(inputFile)[0]
        evalDir = args.output_dir + '\\evaluations\\'
        if not os.path.exists(evalDir):
            os.mkdir(evalDir)
        outputdir = evalDir + inputFileBase
        if not os.path.exists(outputdir):
            os.mkdir(outputdir)
        logging.info('Saving all evaluation images and data to: ' +
                     str(outputdir))
        saveImages = True
    else:
        logging.info(
            'Output directory not set, so not saving any output evaluation images or data')

    """ Loading in infered dataset """
    inferredDf = pd.read_csv(args.inference_data)

    """ Grab Column names  """
    EvalCols = inferredDf.columns.values
    MAECols = [col for col in EvalCols if 'AbsErr' in col]
    MAPECols = [col for col in EvalCols if 'AbsPercentErr' in col]

    """ Create Plots for Mean Absolute Error """
    plt.figure(1)  # , figsize=(15.0, 10.0))
    plt.grid
    inferredVals = inferredDf[MAECols].values
    bplot = plt.boxplot(inferredVals, showfliers=False,
                        showmeans=True, patch_artist=True)
    # Create colors for all boxes, and lines.
    colors = cm.rainbow(np.linspace(.5, 1, 12))
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    for whisker, cap in zip(bplot['whiskers'], bplot['caps']):
        whisker.set(color='b', lw=2)
        cap.set(color='b', lw=2)
    # Setting Grib Info
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', color='black')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    # Set Graph Texts
    plt.xlabel('Predicted Months (tx, with x=number of months in future)')
    plt.ylabel('Mean Absolute Error')
    plt.title(
        'Overall Mean Absolute Error and Standard Deviations')
    # Save Graphs or View them?
    if saveImages:
        MAE_Image = outputdir + \
            '\\Overall_Mean_Absolute_Error_and_Standard_Deviations.png'
        plt.savefig(MAE_Image)
    else:
        plt.show()

    """ Create Plots for Mean Absolute Percentage Error """
    plt.figure(2)  # figsize=(15.0, 10.0))
    plt.grid
    inferredVals = inferredDf[MAPECols].values
    bplot = plt.boxplot(inferredVals, showfliers=False,
                        showmeans=True, patch_artist=True)
    # Create colors for all boxes, and lines.
    colors = cm.rainbow(np.linspace(.5, 1, 12))
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    for whisker, cap in zip(bplot['whiskers'], bplot['caps']):
        whisker.set(color='b', lw=2)
        cap.set(color='b', lw=2)
    # Setting Grib Info
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', color='black')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    # Set Graph Texts
    plt.xlabel('Predicted Months (tx, with x=number of months in future)')
    plt.ylabel('Mean Absolute Percentage Error')
    plt.title(
        'Overall Mean Absolute Percentage Error and Standard Deviations')

    # Save Graphs or View them?
    if saveImages:
        MAE_Image = outputdir + \
            '\\Overall_Mean_Absolute_Percentage_Error_and_Standard_Deviations.png'
        plt.savefig(MAE_Image)
    else:
        plt.show()

    """ Plot all ZHVI predictions for One Neighborhood """
    neighborhoods = list(inferredDf['ZillowNeighborhood'].unique())
    Dates = list(inferredDf['Date'].unique())
    xticksDates = [date for date in Dates if '-01' in date]
    inferredCols = inferredDf.columns.values
    predZHVICOls = [col for col in inferredCols if 'pred_ZHVI_t' in col]

    for neighborhood in neighborhoods:
        neighborhoodDF = inferredDf[inferredDf['ZillowNeighborhood']
                                    == neighborhood]

        inferredCols = inferredDf.columns.values

        mynewDf = neighborhoodDF[[
            'Date', 'ZillowNeighborhood', 'ZHVI_t0', 'pred_ZHVI_t0']]
        plt.figure(figsize=(16.0, 8))
        scatter = plt.scatter(mynewDf['Date'], mynewDf['ZHVI_t0'], c="g", marker="d",
                              label="Actual ZHVI")
        plt.xticks(xticksDates)

        # print(len(mynewDf['ZHVI_t0']))
        colors = cm.rainbow(np.linspace(0, 1, len(predZHVICOls)))

        for futureMonthNumber, predictionColName in enumerate(predZHVICOls, start=0):
            values = neighborhoodDF[predictionColName].values
            # add appropriate number of Nones to start of list
            print(type(values))
            for i in range(0, futureMonthNumber):
                values = np.insert(values, 0, np.nan)
            # chop off values which are predicting future dates which we don't have
            values = values[:len(values)-futureMonthNumber]

            legendStr = 'Predicted ZHVI ' + \
                str(futureMonthNumber) + " Months Ago"
            plt.scatter(mynewDf['Date'], values,
                        c=[colors[futureMonthNumber]], marker="1", label=legendStr)

        plt.xlabel("Date")
        plt.ylabel("Zillow Home Value Index")
        plt.title(
            "Zillow Home Value Index and Predictions for the Neighborhood: " + neighborhood)
        plt.legend(loc='upper right')
        plt.minorticks_on()
        # Customize the major grid
        plt.grid(which='major', linestyle='-', color='black')
        # Customize the minor grid
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

        if saveImages:
            if neighborhood == 'Schorsch Village':
                neighborhoodImageName = outputdir + \
                    '\\' + neighborhood + '_PredictedZHVIOverTime.png'
                plt.savefig(neighborhoodImageName)
                break
        else:
            plt.show()
            # break

    """ Find Errors Per Neighborhoods """
    neighborhoods = list(sorted(inferredDf['ZillowNeighborhood'].unique()))
    col = list(inferredDf.drop(['Date', 'ZillowNeighborhood'], axis=1))

    ErrCols = [col for col in col if 'AbsErr_' in col]

    newdf = pd.DataFrame()
    newdf['ZillowNeighborhood'] = neighborhoods

    cnt = 0
    for col in ErrCols:
        meanerrdf = inferredDf.groupby(
            'ZillowNeighborhood', as_index=False)[col].mean()
        stddevdf = pd.DataFrame(inferredDf.groupby(
            'ZillowNeighborhood', as_index=True)[col].std())
        newdf['MeanAbsErr_t' + str(cnt)] = meanerrdf[col]
        newdf['STDEVAbsErr_t' + str(cnt)] = stddevdf[col].values
        cnt += 1

    evalPerNeighborhoodDir = outputdir + \
        '\\Evaluation_Per_Neighborhood.csv'
    newdf.to_csv(evalPerNeighborhoodDir, index=False)
