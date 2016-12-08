import sys
import csv
import datetime
import numpy as np
import collections
import random
import itertools
import matplotlib.pyplot as plt
from predictor import CheatPredictor, SimpleNNPredictor
from trader import RotQTrader

Data = [
    ('dj', None, None),
    ('gdx', None, None),
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

def getPriceChange(oldPrice, newPrice):
    return float(newPrice - oldPrice) / oldPrice

def trainPredictor(label, predictor, maxIndex):
    for index in range(maxIndex):
        phiX = predictor.extractFeatures(index)
        currentPrice = predictor.getPrice(index)
        predictor.train(phiX, [getPriceChange(currentPrice, predictor.getPrice(index + delta)) for delta in predictor.predictingDelta])

def trainTrader(label, trader, maxIndex):
    for i in range(10):
        startIndex = random.choice(range(0, maxIndex))
        endIndex = min(startIndex + 365, maxIndex)
        
        trader.train(startIndex, endIndex)

def main():
    for key, _, _ in Data:
        # dataToPrice[np.datetime64] := adjusted close price
        # startDate := The first date in the stock data
        # lastDate := The last date in the stock data
        dateToPrice, startDate, lastDate = load(key)

        # Fill up stocks, the prices array, so that we can look up by day-index.
        stocks = makeStockArray(dateToPrice, startDate, lastDate)

        def getPrice(index):
            if index < 0:
                return stocks[0]
            elif index >= len(stocks):
                return stocks[len(stocks) - 1]
            return stocks[index]

        predictors = [CheatPredictor(getPrice)]
        for predictor in predictors:
            # We learn from the first day up to one year ago.
            trainPredictor(key, predictor, len(stocks) - 365)

        traders = [RotQTrader(predictors[0], getPrice)]
        for trader in traders:
            # We learn from the first day up to one year ago.
            trainTrader(key, trader, len(stocks) - 365)

        for trader in traders:
            gain = trader.test(len(stocks) - 365, len(stocks))
            print '%s gain: %f' % (key, gain)
main()
