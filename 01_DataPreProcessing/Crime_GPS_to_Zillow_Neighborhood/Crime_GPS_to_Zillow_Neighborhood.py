#!/usr/bin/env python
# title           :Crime_GPS_to_Zillow_Neighborhood.py
# description     :This script adds the zillow defined neighborhood which a crime took place in
# author          :Michael Bowyer
# date            :20191117
# version         :0.0
# usage           :python Crime_GPS_to_Zillow_Neighborhood.py -ic input_crime.csv -is input_shapes.shp -o output_crime.csv
# notes           :
# python_version  :Python 3.7.3
# ==============================================================================
import logging
import shapefile
import argparse
import os
import pandas as pd
import GPStoZNeighborhood as gps2zh

""" Setup logging config """
logging.basicConfig(level=logging.DEBUG, filemode='w',
                    format='%(levelname)s:: %(message)s')  # ,filename='app.log')
#logging.debug('This is a debug message')

""" Input Arguments """
parser = argparse.ArgumentParser(
    description='This script adds the zillow defined neighborhood which a crime took place in')
parser.add_argument('--input_crime', '-ic', type=str, required=True,
                    help='The input crime data csv file')
parser.add_argument('--output', '-o', default='output.csv',
                    help='The output file to store the crime data with appended Zillow Neighborhoods')
parser.add_argument('--input_shapes', '-is', required=True,
                    help='The file which contains definitions of the neighborhood shapes')

if __name__ == "__main__":
    args = parser.parse_args()
    logging.info('Using input crime file ' + str(args.input_crime))
    logging.info('Using input geographic shape file ' + str(args.input_shapes))

    outFileName = ""
    if args.output == 'output.csv':
        outFileName = os.path.splitext(
            args.input_crime)[0] + '_w_Zillow_Neighborhoods.csv'
    else:
        outFileName = args.output
    logging.info('Output tranform file will be saved to ' + str(outFileName))

    """ Read in the shapes defined in the Zillow Neighborhood shape file """
    r = shapefile.Reader(args.input_shapes)
    shapes = r.shapes()
    encodings = r.records()

    """ Create class instance for GPS to Zillow neighborhood to store shapes and crime data """
    neihgborhoodFinder = gps2zh.GPStoZNeighborhood(
        pd.read_csv(args.input_crime), shapes, encodings)

    """ Call the function which will find the zillow neighbor hood for each crime data """
    inputCrimeWZillowNH = neihgborhoodFinder.add_zillow_neighborhood_column()

    logging.info("Saving output dataframe to " + outFileName)
    inputCrimeWZillowNH.to_csv(outFileName, index=None, header=True)