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


if __name__ == "__main__":

    """ Parse Args """
    args = parser.parse_args()
    logging.info('Using input inferece file to generate metrics: ' +
                 str(args.inference_data))

# """ Generate Error Metrics for overall Training data """
# EvalCols = evaluationDf.columns.values
# MAECols = [col for col in EvalCols if 'MAE' in col]
# MAPECols = [col for col in EvalCols if 'MAPE' in col]

# MAEMean = []
# MAEStd = []
# for MAECol in MAECols:
#     MAEMean.append(evaluationDf[MAECol].mean())
#     MAEStd.append(evaluationDf[MAECol].std())

# MAPEMean = []
# MAPEStd = []
# for MAPECol in MAPECols:
#     MAPEMean.append(evaluationDf[MAPECol].mean())
#     MAPEStd.append(evaluationDf[MAPECol].std())

# plt.subplot(121)
# plt.errorbar(MAECols, MAEMean, MAEStd, capsize=15,
#              capthick=3, barsabove=True, linestyle='None')
# plt.xlabel('Predicted Months (tx, with x=number of months in future)')
# plt.ylabel('Mean Absolute Error')
# plt.title('Overall Training Mean Asolute Error Mean and Standard Deviations')
# plt.subplot(122)
# plt.errorbar(MAPECols, MAPEMean, MAPEStd, capsize=15,
#              capthick=3, barsabove=True, linestyle='None')
# plt.xlabel('Predicted Months (tx, with x=number of months in future)')
# plt.ylabel('Mean Absolute Percentage Error')
# plt.title(
#     'Overall Training Mean Asolute Percentage Error Mean and Standard Deviations')
# plt.show()

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
