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

class row_transform:
    
    def __init__(self, input_df):
        self.input_df = input_df
        print(self.input_df.shape)
        