#!/usr/bin/env python
#title           :ZHVI_to_row_tranformation.py
#description     :This script converts the Zillow Home Value Index into individual rows to be used for machine learning.
#author          :Michael Bowuer
#date            :20191109
#version         :0.0
#usage           :python ZHVI_to_row_tranformation.py -i input.csv -o output.csv
#notes           :
#python_version  :Python 3.7.3 
#==============================================================================
import logging
import argparse
import pandas as pd
import row_transform as rt

""" Setup logging config """
logging.basicConfig(level=logging.DEBUG, filemode='w', format='%(levelname)s:: %(message)s')#,filename='app.log')
#logging.debug('This is a debug message')

""" Input Arguments """
parser = argparse.ArgumentParser(description='This script converts the Zillow Home Value Index into individual rows to be used for machine learning.')
parser.add_argument('--input','-i', type=str,
                    help='The input ZHVI Input data to transform')
parser.add_argument('--output','-o', default='output_ZHVI_row_transformed.csv',
                    help='The output file to store the transformed ZHVI rows')


if __name__ == "__main__":
    args = parser.parse_args()
    logging.info('Using input file ' + str(args.input))
    logging.info('Output tranform file will be saved to ' + str(args.output))

    rt.row_transform(pd.read_csv(args.input))
