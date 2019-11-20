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
import tqdm
import sys


class GPStoZNeighborhood:

    def __init__(self, input_crime_df, input_zillow_neighborhood_shapes, input_zillow_neighborhood_encodings):
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

        """Initialize and save all zillow neighborhood shapes and encodings"""
        self.nhShapes = input_zillow_neighborhood_shapes
        self.nhEncodings = [
            item.name for item in input_zillow_neighborhood_encodings]
        logging.debug('Input amount of neighborhood shapes is ' +
                      str(len(input_zillow_neighborhood_shapes)))
        polygons = []
        for nhShape in self.nhShapes:
            polygon = shape(nhShape)
            polygons.append(polygon)
        self.nhPolygons = polygons

    def check_lat_long_exist(self):
        """
        Ensures that latitude and longitude columns exist in input crime dataset

        Parameters:

        Returns:
            True: When both latitude and longitude columns exist
            False: Otherwise
        """
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
        """
        Finds the associated zillow neighborhoods for the input rows latitude and longitudue.

        Parameters:
            input_row: Row from the crime data set. Contains 1 individual crime information.

        Returns:
            output_row_list: The input row with the NumMatchedNeidhborhoods and ZillowNeighborhood columns populated.
        """
        d = dict(
            input_row[1])  # First element of input row is the index of the input row, row 1 is the actually row data
        lati = d['Latitude']
        longi = d['Longitude']

        point = Point(longi, lati)
        count = 0
        polyIndex = -1
        for idx, polygon in enumerate(self.nhPolygons):
            if polygon.contains(point):
                count = count + 1
                polyIndex = idx

        """Append new columns to input list"""
        output_row_list = list(input_row[1])
        output_row_list.append(count)
        output_row_list.append(self.nhEncodings[polyIndex])
        return output_row_list

    def add_zillow_neighborhood_column(self):
        """
        Returns original crime data frame with appended NumMatchedNeidhborhoods and ZillowNeighborhood columns.

        Returns:
            Pandas data frame: The original crime dataframe with the NumMatchedNeidhborhoods and ZillowNeighborhood columns populated.
        """
        dataFrameHeadings = list(self.crime_df.columns.values)
        dataFrameHeadings.append("NumMatchedNeidhborhoods")
        dataFrameHeadings.append("ZillowNeighborhood")

        logging.info('Begging to append zillow neighborhood values.')
        p = Pool()
        results = []
        rows, cols = self.crime_df.shape

        """ multiprocess each row in the input crime dataset """
        for _ in tqdm.tqdm(p.imap_unordered(self.find_zillow_neighborhood, self.crime_df.iterrows(), chunksize=10000), total=rows):
            pass
            results.append(_)

        logging.info('Converting output results list into dataframe.')
        outputdataframe = pd.DataFrame(results, columns=dataFrameHeadings)

        return outputdataframe
