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
import row_transformer as rt

""" Setup logging config """
logging.basicConfig(level=logging.DEBUG, filemode='w', format='%(levelname)s:: %(message)s')#,filename='app.log')
#logging.debug('This is a debug message')

""" Input Arguments """
parser = argparse.ArgumentParser(description='This script converts the Zillow Home Value Index into individual rows to be used for machine learning.')
parser.add_argument('--input','-i', type=str, required=True,
                    help='The input ZHVI Input data to transform')
parser.add_argument('--output','-o', default='output_ZHVI_row_transformed.csv',
                    help='The output file to store the transformed ZHVI rows')
parser.add_argument('--first_tranform_row','-ftr', required=True,
                    help='The first row in the ZHVI which will be transformed')

if __name__ == "__main__":
    args = parser.parse_args()
    logging.info('Using input file ' + str(args.input))
    logging.info('Output tranform file will be saved to ' + str(args.output))
    logging.info('The first row which will be used for transformation is: ' + str(args.first_tranform_row))

    row_transform = rt.row_transformer(pd.read_csv(args.input),args.first_tranform_row)
    finalDf = row_transform.transform_rows()

    print(finalDf)
