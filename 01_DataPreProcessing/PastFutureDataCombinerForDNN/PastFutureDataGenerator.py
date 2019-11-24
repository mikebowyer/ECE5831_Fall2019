#!/usr/bin/env python
# title           :PastFutureDataGenerator.py
# description     :This class aids in the generation of feature data for previous months of zillow data + future zillow data to predict
# author          :Michael Bowyer
# date            :20191123
# version         :0.0
# python_version  :Python 3.7.3
# ==============================================================================
import logging
import pandas as pd
from multiprocessing import Pool
import tqdm
import itertools


class PastFutureDataGenerator:

    def __init__(self, input_df, prev_months, future_months):
        """
        Saves input dataframe, how many previous months to output, and how many future months to output.

        Parameters:
        input_df (pandas dataframe): Raw input data frame with each row containing all crime data for a given neighborhood in a given month.
        prev_months (int): How many previous months of data to generate in the output file
        future_months (int): How many months in the future are intended to be predicted
        """
        self.prev_months = int(prev_months)
        self.future_months = int(future_months)
        self.inputDf = input_df

    def create_output_headings(self):
        """
        Generates and saved the output data frame column headings for both output data frameswhich:
            1) contains previous crime+ZHVI data, and future ZHVI data
            2) contains previous ZHVI data, and future ZHVI data
        """
        allHeadings = list(self.inputDf.columns.values)
        allHeadings.remove('Date')
        allHeadings.remove('ZillowNeighborhood')
        headings_housingCrime = ['Date', 'ZillowNeighborhood']
        headings_housing = ['Date', 'ZillowNeighborhood']
        print(len(allHeadings))
        logging.info("Creating input headings now.")

        for i in range(0, self.prev_months):
            # print("i = " + str(i))
            """ Create new headings for housingCrime Data """
            tempHeadings = []
            timeStep = '_t-' + str(i+1)  # create time step string
            # for each feature name add times step value to it
            tempHeadings = tempHeadings + [s + timeStep for s in allHeadings]
            # add all feature values to ZHVI and crime data for this timestep
            headings_housingCrime = headings_housingCrime + tempHeadings
            """ Create new headings for just ZHVI Data """
            zhviTimestep = "ZHVI" + timeStep
            headings_housing.append(zhviTimestep)

        newtargetheadings = []
        for k in range(0, self.future_months):
            targetTimeStep = 'ZHVI_t' + str(k)
            newtargetheadings.append(targetTimeStep)

        headings_housingCrime = headings_housingCrime + newtargetheadings
        headings_housing = headings_housing + newtargetheadings

        logging.info("Output housing dataset will have " +
                     str(len(headings_housing)) + " columns.")
        logging.info("Output housing and crime dataset will have " +
                     str(len(headings_housingCrime)) + " columns.")

        self.housingHeadings = headings_housing
        self.housingCrimeHeadings = headings_housingCrime

    def neighborhood_get_past_future_data(self, neighborhood):
        """
        For a given neighborhood, finds all entries in inputDf with input neighborhood.
        Then creates DF to fill in a given row which would match the generated column headings in create_output_headings().

        Parameters:
        neighborhood (str): Name of Neighborhood

        Returns:
        housingCrime (list): list of data with all previous months crime+ZHVI values, and current + future ZHVI values
        housing (dataframe): list of data with all previous months ZHVI values, and current + future ZHVI values
        """
        """ Prepare data frame for only this neighborhood"""
        neighborhoodDf = self.inputDf[self.inputDf['ZillowNeighborhood']
                                      == neighborhood]

        # Need to be saved because date will be dropped from Df so it isn't replicated in output Df
        dates = neighborhoodDf['Date']
        neighborhoodDf = neighborhoodDf.drop(
            axis=1, columns=['ZillowNeighborhood', 'Date'])
        rows = neighborhoodDf.shape[0]

        """ Create output data frames using previous Crime+ZHVI data, and current crime+ZHVI data, and future ZHVI data"""
        housingCrime = []
        housing = []
        # Start loop prev_month-1 elements in so there is sufficient previous months to store, and end self.future_months early to have enough future data to predict
        for index, row in itertools.islice(neighborhoodDf.iterrows(), self.prev_months-1, (rows-self.future_months)):
            """ Start each row with the data and the neighborhood """
            newrowhousingCrime = [dates.loc[index], neighborhood]
            newrowHousing = [dates.loc[index], neighborhood]
            """ Add past data to new rows """
            for i in range(0, self.prev_months):
                newrowhousingCrime = newrowhousingCrime + \
                    list(neighborhoodDf.loc[index-i].values)
                newrowHousing = newrowHousing + \
                    [neighborhoodDf['ZHVI'].loc[index-i]]

            """ Add future ZHVI data to new row """
            for j in range(0, self.future_months):
                newrowhousingCrime = newrowhousingCrime + \
                    [neighborhoodDf['ZHVI'].loc[index+j]]
                newrowHousing = newrowHousing + \
                    [neighborhoodDf['ZHVI'].loc[index+j]]

            housingCrime.append(newrowhousingCrime)
            housing.append(newrowHousing)

        return housingCrime, housing

    def generate_past_future_data(self):
        """
        Calls neighborhood_get_past_future_data with each neighborhood as input to start N neighborhoods processes to generate
        the needed previous crime+ZHVI data and future ZHVI data.

        Returns:
        housingCrime (dataframe): Data frame with all previous months crime+ZHVI values, and current + future ZHVI values
        housing (dataframe): Data frame with all previous months ZHVI values, and current + future ZHVI values
        """
        """ Prepare data frame for only this neighborhood"""
        neighborhoods = list(self.inputDf['ZillowNeighborhood'].unique())
        logging.info(
            "Starting to generate past and future data now based on neighbor hoods.")
        logging.debug(
            "Working on a total of " + str(len(neighborhoods)) + " neighborhoods!")

        """ Process all data in as many pools as possible """
        p = Pool()
        results = []
        for _ in tqdm.tqdm(p.imap_unordered(self.neighborhood_get_past_future_data, iter(neighborhoods), chunksize=10), total=len(neighborhoods)):
            results.append(_)
            pass
        logging.info("Finished generating past and future data!")

        """ Compress output into two data frames """
        logging.info(
            "Now concatanating results and creating output dataframes.")
        housingCrime = [result[0] for result in results]
        housingCrime = list(itertools.chain.from_iterable(housingCrime))
        housing = [result[1] for result in results]
        housing = list(itertools.chain.from_iterable(housing))

        """ Convert outputs into data frames """
        housingCrimeDf = pd.DataFrame(housingCrime)
        housingCrimeDf.columns = self.housingCrimeHeadings
        housingDf = pd.DataFrame(housing)
        housingDf.columns = self.housingHeadings

        logging.info(
            "Final data set with crime and housing info is of size: " + str(housingCrimeDf.shape))
        logging.info(
            "Final data set with housing only info is of size: " + str(housingDf.shape))

        return housingCrimeDf, housingDf
