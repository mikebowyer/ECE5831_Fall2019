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
import pandas as pd


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
        else:
            logging.error(
                "The input_df argument is not a pandas dataframe.")
            exit - 1

        if not self.check_lat_long_exist():
            logging.error(
                "The input_df does not contain columns titled Lat and Long.")
            exit - 1

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
