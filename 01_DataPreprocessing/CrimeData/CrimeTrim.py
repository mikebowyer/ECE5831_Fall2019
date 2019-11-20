#!/usr/bin/env python
#title           :CrimeTrim.py
#description     :This script converts the Chicago Crime Data into a trimmed dataset
#author          :Bryan Sandoval
#date            :20191109
#version         :0.0
#usage           :
#notes           :
#python_version  :Python 3.7.3 
#==============================================================================

#Import Libraries
import pandas as pd

cf = pd.read_csv("Crimes_-_2001_to_present.csv")

#Eliminate unecessary columns
keep_col = ['Date','Primary Type','Location Description','Arrest', 'Domestic', 'Year', 'Latitude', 'Longitude'] #Columns to keep
crime_new = cf[keep_col]

#Drop any rows with uncomplete data
crime_new = crime_new.dropna()

#Write trimmed data to output file
crime_new.to_csv("ChicagoTrimmedCrimeData.csv", index=False)
