#!/usr/bin/env python
# title           :PastMonthsToTrain_FutureZHVITarget_Generator..py
# description     :This script will take in the csv file with monthly crime and ZHVI values for a neighbor hood, and create data set to train a neural net on.
# author          :Michael Bowyer
# date            :20191123
# version         :0.0
# usage           :python PastMonthsToTrain_FutureZHVITarget_Generator.py.py -i input_CrimePerMonthPerNeighborhood.csv -o 4MonthsTraining_2monthsPrediction.csv
# notes           :
# python_version  :Python 3.7.3
# ==============================================================================
import logging
import argparse
import os
import pandas as pd
import PastFutureDataGenerator as pfdg

""" Setup logging config """
logging.basicConfig(level=logging.DEBUG, filemode='w',
                    format='%(levelname)s:: %(message)s')  # ,filename='app.log')
# logging.debug('This is a debug message')

""" Input Arguments """
parser = argparse.ArgumentParser(
    description='This script will take in the csv file with monthly crime and ZHVI values for a neighbor hood, and create data set to train a neural net on.')
parser.add_argument('--input', '-i', type=str, required=True,
                    help='The input dataset which a row represents all crime categories for a neighborhood in a month.')
parser.add_argument('--output', '-o', default='output.csv',
                    help='The output file name where the output data to be used for trianing/evaluating a network is saved.')
parser.add_argument('--prev_months', '-pm', required=True,
                    help='How many months in the past should be used for training the model')
parser.add_argument('--future_months', '-fm', required=True,
                    help='How many months in the future to predict (1=Only predict current month, and no future months)')

if __name__ == "__main__":
    """ Parse Args """
    args = parser.parse_args()
    logging.info('Using input file ' + str(args.input))
    logging.info('Output tranform file will be saved to ' + str(args.output))
    logging.info('Each row in the output file will contain ' +
                 str(args.prev_months) + ' months of  worth of data, and will contain ZHVI values of the current month and ' +
                 str(int(args.future_months)-1) + ' months in the future.')

    """ Create input Df and drop date column """
    inputDf = pd.read_csv(args.input)
    inputDf = inputDf.drop(axis=1, columns=['Date'])

    """ Generate new data """
    dataGenerator = pfdg.PastFutureDataGenerator(
        inputDf, args.prev_months, args.future_months)
    dataGenerator.create_output_headings()
    dataGenerator.generate_past_future_data()
    # logging.info("Final Data frame outputis shown below")
    # logging.info(finalDf)

    outFileName = ""
    if args.output == 'output.csv':
        outFileName = os.path.splitext(args.input)[0] + '_transformed.csv'
    else:
        outFileName = args.output

    logging.info("Saving output dataframe to " + outFileName)
    # finalDf.to_csv(outFileName, index=None, header=True)
