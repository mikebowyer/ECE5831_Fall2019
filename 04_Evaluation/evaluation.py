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

    """ Parse Args """
    args = parser.parse_args()
    logging.info('Using input inferece file to generate metrics: ' +
                 str(args.inference_data))
    outputdir = ''
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
    else:
        logging.info(
            'Output directory not set, so not saving any output evaluation images or data')

    """ Loading in infered dataset """
    inferredDf = pd.read_csv(args.inference_data)

    """ Generate Plots for MAE and MAPE for entire inferred set """
    generatePlots = False
    if(generatePlots):
        """ Generate Error Metrics for overall Training data """
        EvalCols = inferredDf.columns.values
        MAECols = [col for col in EvalCols if 'AbsErr' in col]
        MAPECols = [col for col in EvalCols if 'AbsPercentErr' in col]

        MAEMean = []
        MAEStd = []
        for MAECol in MAECols:
            MAEMean.append(inferredDf[MAECol].mean())
            MAEStd.append(inferredDf[MAECol].std())
        MAPEMean = []
        MAPEStd = []
        for MAPECol in MAPECols:
            MAPEMean.append(inferredDf[MAPECol].mean())
            MAPEStd.append(inferredDf[MAPECol].std())

        """ Create Plots for Mean Absolute Error """
        plt.figure(1, figsize=(15.0, 10.0))
        plt.errorbar(MAECols, MAEMean, MAEStd, capsize=15,
                     capthick=3, barsabove=True, linestyle='None')
        plt.xlabel('Predicted Months (tx, with x=number of months in future)')
        plt.ylabel('Mean Absolute Error')
        plt.title(
            'Overall Mean Absolute Error and Standard Deviations')
        if not (args.output_dir == ''):
            MAE_Image = outputdir + \
                '\\Overall_Mean_Absolute_Error_and_Standard_Deviations.png'
            plt.savefig(MAE_Image)
        else:
            plt.show()

        """ Create Plots for Mean Absolute Percentage Error """
        plt.figure(2, figsize=(15.0, 10.0))
        plt.errorbar(MAPECols, MAPEMean, MAPEStd, capsize=15,
                     capthick=3, barsabove=True, linestyle='None')
        plt.xlabel('Predicted Months (tx, with x=number of months in future)')
        plt.ylabel('Mean Absolute Percentage Error')
        plt.title(
            'Overall Mean Absolute Percentage Error and Standard Deviations')

        if not (args.output_dir == ''):
            MAPE_Image = outputdir + \
                '\\Overall_Mean_Absolute_Percentage_Error_and_Standard_Deviations.png'
            plt.savefig(MAPE_Image)
        else:
            plt.show()

    """ Plot all ZHVI predictions for One Neighborhood """
    neighborhoods = list(inferredDf['ZillowNeighborhood'].unique())
    Dates = list(inferredDf['Date'].unique())
    xticksDates = [date for date in Dates if '-01' in date]
    for neighborhood in neighborhoods:
        neighborhoodDF = inferredDf[inferredDf['ZillowNeighborhood']
                                    == neighborhood]

        inferredCols = inferredDf.columns.values

        mynewDf = neighborhoodDF[[
            'Date', 'ZillowNeighborhood', 'ZHVI_t0', 'pred_ZHVI_t0']]

        scatter = plt.scatter(mynewDf['Date'], mynewDf['ZHVI_t0'], c="g", alpha=0.5, marker=r'$\clubsuit$',
                              label="Actual ZHVI")
        plt.xticks(xticksDates)
        # plt.scatter(mynewDf['Date'], mynewDf['pred_ZHVI_t0'], c="g", alpha=0.5, marker=r'+',
        # label = "Actual ZHVI")
        plt.xlabel("Date")
        plt.ylabel("Zillow Home Value Index")
        plt.title(
            "Zillow Home Value Index and Predictions for the Neighborhood: " + neighborhood)
        plt.legend(loc='upper left')
        plt.show()

        # Cols = [col for col in EvalCols if 'AbsErr' in col]
        # MAPECols = [col for col in EvalCols if 'AbsPercentErr' in col]

        break

    """ Find Errors Per Neighborhoods """
    # neighborhoods = list(inferredDf['ZillowNeighborhood'].unique())
    # for neighborhood in neighborhoods:
    #     neighborhoodDF = inferredDf[inferredDf['ZillowNeighborhood']
    #                                 == neighborhood]
    #     MAPEMean = []
    #     MAPEStd = []
    #     for MAPECol in MAPECols:
    #         MAPEMean.append(neighborhoodDF[MAPECol].mean())
    #         MAPEStd.append(inferredDf[MAPECol].std())
    #     print(MAPEMean)
    #     print(MAEMean)
    #     break
    # # plt.scatter(Albany['Date'],Albany['ZHVI_t0'])
