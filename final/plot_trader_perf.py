import sys
import csv
import datetime
import numpy as np
import collections
import random
import itertools
import matplotlib.pyplot as plt
import pandas as pd

Data = [
    ('dj', None, None),
    ('qcom', None, None),
    ('rut', None, None),
    ('wmt', None, None),
    ('hd', None, None),
    ('low', None, None),
    ('tgt', None, None),
    ('cost', None, None),
    ('nke', None, None),
    ('ko', None, None),
    ('xom', None, None),
    ('cvx', None, None),
    ('cop', None, None),
    ('bp', None, None),
    ('ibm', None, None),
    ('aapl', None, None),
]

def load(key):
    dateToPrice = {}
    startDate = None
    endDate = None
    with open('../data/%s.csv' % key, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            date = np.datetime64(row[0])
            if endDate == None:
                endDate = date
            startDate = date
            price = float(row[6])
            dateToPrice[date] = price
    return (dateToPrice, startDate, endDate)

# The function returns the stock price for the day. It returns the
# last day price if the stock data for the day is not available
# (e.g. holiday)
def stockForDate(dateToPrice, date):
    while not date in dateToPrice:
        date = date - np.timedelta64(1, 'D')
    return dateToPrice[date]

def makeStockArray(dateToPrice, startDate, lastDate):
    stocks = []
    date = startDate
    while date <= lastDate:
        stocks += [stockForDate(dateToPrice, date)]
        date += np.timedelta64(1, 'D')
    return stocks

perfDF = None
for D in ['1', '7', '137']:
    perfFileName = 'trader_perf_100_%s.csv' % D
    df = pd.DataFrame.from_csv(perfFileName, index_col=None, header=None)
    df.columns = ['symbol', 'iteration', D]
    df[[D]] = df[[D]].apply(pd.to_numeric)
    if perfDF is not None:
        perfDF = pd.merge(perfDF, df[['symbol',D]], on='symbol')
    else:
        perfDF = df

perfDF = perfDF.append(perfDF.sum(numeric_only=True), ignore_index=True)        

print perfDF
print perfDF[perfDF.D=='1']
perfDF[perfDF.D=='1'].describe
for symbol, _, _ in Data:
    dateToPrice, startDate, lastDate = load(symbol)
    startPrice = stockForDate(dateToPrice, np.datetime64('2015-11-11'))
    
    
    
                              
    
