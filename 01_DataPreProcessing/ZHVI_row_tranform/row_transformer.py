#!/usr/bin/env python
#title           :row_transform.py
#description     :This script converts the Zillow Home Value Index into individual rows to be used for machine learning.
#author          :Michael Bowuer
#date            :20191109
#version         :0.0
#usage           :python ZHVI_to_row_tranformation.py -i input.csv -o output.csv
#notes           :
#python_version  :Python 3.7.3 
#==============================================================================
import logging
import pandas as pd
class row_transformer:
    
    def __init__(self, input_df, start_row):
        """ 
        Saves input dataframe, verifies start_row, then transforms 
        all columns after start_row to new row entries. 
    
        Parameters: 
        input_df (pandas dataframe): Raw input data frame
        start_row (str): Name of first column to be transformed into new rows
    
        Returns: 
        Pandas dataframe: New pandas data frame with all colums after start_row
        """
        if isinstance(input_df, pd.DataFrame):
            self.input_df = input_df
            self.start_row = start_row
        else:
            logging.error("The input_df argument is not a pandas dataframe.")
            exit -1

        logging.debug('Input Dataframe is of shape ' + str(input_df.shape))

        self.check_and_find_start_row_index()

    def check_and_find_start_row_index(self):
        logging.debug('Checking to ensure input start row exists ' + str(self.start_row))
        if self.start_row in self.input_df.columns:
            logging.debug('Found column ' + str(self.start_row) + ' in input data frame.')
            self.start_row_ind = self.input_df.columns.get_loc(self.start_row)
            logging.debug('Column is column index ' + str(self.start_row_ind))
        else: 
            logging.error("Could not The input_df argument is not a pandas dataframe.")
            exit -1

    def transform_rows(self):
        logging.info('Starting row transformation using starting row ' + str(self.start_row))
        numCols = len(self.input_df.columns)
        allHeadings = list(self.input_df.columns.values)
        dataFrameHeadings =  allHeadings[:self.start_row_ind]
        dataFrameHeadings.append("Year-Month")
        dataFrameHeadings.append("ZHVI")
        
        outputdataframe = pd.DataFrame(columns=dataFrameHeadings)
        inputDfColNames = list(self.input_df.columns)

        for index, row in self.input_df.iterrows():
            newRowsInitialCols = []
            logging.debug("Processing row " + str(index) + "/" + str(self.input_df.shape[0]))

            for column in range(0,self.start_row_ind):
                newRowsInitialCols.append(row[column])        

            for column in range(self.start_row_ind,numCols):
                newRowData = newRowsInitialCols.copy()
                newRowData.append(inputDfColNames[column])
                newRowData.append(row[column])
                newRowDf = pd.DataFrame([newRowData],columns=dataFrameHeadings)
                outputdataframe = outputdataframe.append(newRowDf,ignore_index=True)

        print(outputdataframe.shape)
        return outputdataframe
        
        