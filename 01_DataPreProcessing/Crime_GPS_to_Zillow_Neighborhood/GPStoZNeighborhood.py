#!/usr/bin/env python
# title           :GPStoZNeighborhood.py
# description     :This class takes input crime data and converts them to zillow neighborhoods
# author          :Michael Bowuer
# date            :20191117
# version         :0.0
# notes           :
# python_version  :Python 3.7.3
# ==============================================================================
import logging
import shapefile
from multiprocessing import Pool
from shapely.geometry import shape, Point
import pandas as pd
import numpy as np


class GPStoZNeighborhood:

    def __init__(self, input_crime_df, input_zillow_neighborhood_shapes):
        """
        Saves input crime dataframe and ensures there is lat and long in it.
        Saves the input neighborhood shapes locally.

        Parameters:
        input_crime_df (pandas dataframe): Raw input crime data frame with lat and log columns
        input_zillow_neighborhood_shapes (shapes): shapes of all zillow neighborhoods

        Returns:
        Pandas dataframe: The input crime dataframe with an added column titled "ZillowNeighborhood"
        """
        if isinstance(input_crime_df, pd.DataFrame):
            self.crime_df = input_crime_df
            logging.debug('Input Dataframe is of shape ' +
                          str(input_crime_df.shape))
        else:
            logging.error(
                "The input_df argument is not a pandas dataframe.")
            exit - 1

        if not self.check_lat_long_exist():
            logging.error(
                "The input_df does not contain columns titled Lat and Long.")
            exit - 1

        self.nhShapes = input_zillow_neighborhood_shapes
        logging.debug('Input amount of neighborhood shapes is ' +
                      str(len(input_zillow_neighborhood_shapes)))
        polygons = []
        for nhShape in self.nhShapes:
            polygon = shape(nhShape)
            polygons.append(polygon)
        self.nhPolygons = polygons

    def check_lat_long_exist(self):
        logging.debug(
            'Checking to ensure lat and long rows exist.')

        lat_exists = "Latitude" in self.crime_df.columns
        lon_exists = "Longitude" in self.crime_df.columns
        if lat_exists and lon_exists:
            logging.debug(
                'Found column Latitude and Longitude columns in input data frame.')
            return True
        else:
            return False

    def find_zillow_neighborhood(self, input_row):
        d = dict(input_row[1])
        lati = d['Latitude']
        longi = d['Longitude']

        point = Point(longi, lati)
        count = 0
        polyIndex = 0
        for idx, polygon in enumerate(self.nhPolygons):
            if polygon.contains(point):
                count = count + 1
                polyIndex = idx
        inlist = list(input_row[1])
        inlist.append(count)
        inlist.append(polyIndex)
        return inlist

    def add_zillow_neighborhood_column(self):

        dataFrameHeadings = list(self.crime_df.columns.values)
        dataFrameHeadings.append("NumMatchedNeidhborhoods")
        dataFrameHeadings.append("ZillowNeighborhood")

        logging.info('Begging to append zillow neighborhood values.')
        p = Pool()
        results = p.map(self.find_zillow_neighborhood,
                        self.crime_df.iterrows())
        p.close()
        p.join()

        outputdataframe = pd.DataFrame(columns=dataFrameHeadings)

        for result in results:
            newRowDf = pd.DataFrame([result], columns=dataFrameHeadings)
            outputdataframe = outputdataframe.append(
                newRowDf, ignore_index=True)

        print(outputdataframe.head)

        return outputdataframe
