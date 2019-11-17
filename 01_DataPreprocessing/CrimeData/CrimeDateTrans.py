#!/usr/bin/env python
#title           :CrimeTrans.py
#description     :This script takes the 'Date' entries and converts them to the month-day format
#author          :Bryan Sandoval
#date            :20191117
#version         :0.0
#usage           :
#notes           :
#python_version  :Python 3.7.3 
#==============================================================================

#Import Python Libraries
import pandas as pd
import datetime

cf = pd.read_csv("ChicagoTrimmedCrimeData.csv")

#cf['Date'] = cf['Date'].apply(lambda x: datetime.datetime.strptime(x.split(' ')[0], '%m/%d/%Y').strftime('20%y-%m'))

#Eliminate timestamp from datatime
cf['Date'] = pd.to_datetime(cf['Date']) #returns values in datetime format
#cf['Date'] = cf['Date'].dt.date #returns only year-month-date
cf['Date'] = cf['Date'].dt.to_period('M') #returns only month-date

#Test
#new = cf[0:100]
#new['Date'] = pd.to_datetime(new['Date'])
#new['new_date_column'] = new['Date'].dt.date

#Write trimmed data to output file
cf.to_csv("ChicagoTrimmed_Date_CrimeData.csv", index=False)