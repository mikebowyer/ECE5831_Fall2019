import logging
import pandas as pd
from multiprocessing import Pool
import tqdm
import itertools


class PastFutureDataGenerator:

    def __init__(self, input_df, prev_months, future_months):
        self.prev_months = int(prev_months)
        self.future_months = int(future_months)
        self.inputDf = input_df

    def create_output_headings(self):
        allHeadings = list(self.inputDf.columns.values)
        headings_housingCrime = []
        headings_housing = []
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
        neighborhoodDf = self.inputDf[self.inputDf['ZillowNeighborhood']
                                      == neighborhood]
        rows = neighborhoodDf.shape[0]

        housingCrime = []
        housing = []
        for index, row in itertools.islice(neighborhoodDf.iterrows(), self.prev_months, (rows-self.future_months)):
            # print(row['Date'])
            # print(index)
            newrowhousingCrime = []
            newrowHousing = []
            """ Add past data to new row """
            for i in range(0, self.prev_months):
                newrowhousingCrime = newrowhousingCrime + \
                    list(neighborhoodDf.loc[index-i].values)
                newrowHousing = newrowHousing + \
                    [neighborhoodDf['ZHVI'].loc[index-i]]

            """ Add target data to new row """
            for j in range(0, self.future_months):
                newrowhousingCrime = newrowhousingCrime + \
                    [neighborhoodDf['ZHVI'].loc[index+j]]
                newrowHousing = newrowHousing + \
                    [neighborhoodDf['ZHVI'].loc[index+j]]

            housingCrime.append(newrowhousingCrime)
            housing.append(newrowHousing)

        return housingCrime, housing

    def generate_past_future_data(self):
        neighborhoods = list(self.inputDf['ZillowNeighborhood'].unique())
        logging.info(
            "Starting to generate past and future data now based on neighbor hoods.")
        logging.debug(
            "Working on a total of " + str(len(neighborhoods)) + " neighborhoods!")

        p = Pool()
        results = []
        for _ in tqdm.tqdm(p.imap_unordered(self.neighborhood_get_past_future_data, iter(neighborhoods), chunksize=10), total=len(neighborhoods)):
            results.append(_)
            pass

        logging.info("Finished generating past and future data!")

        """ Compress output into two data frames """
        logging.info(
            "Now concatanating output results and creating output dataframes.")
        housingCrime = [result[0] for result in results]
        housingCrime = list(itertools.chain.from_iterable(housingCrime))

        housing = [result[1] for result in results]
        housing = list(itertools.chain.from_iterable(housing))

        # , self.housingCrimeHeadings)
        housingCrimeDf = pd.DataFrame(housingCrime)
        housingCrimeDf.columns = self.housingCrimeHeadings

        housingDf = pd.DataFrame(housing)
        housingDf.columns = self.housingHeadings

        logging.info(
            "Final data set with crime and housing info is of size: " + str(housingCrimeDf.shape))
        logging.info(
            "Final data set with housing only info is of size: " + str(housingDf.shape))

        return housingCrimeDf, housingDf
