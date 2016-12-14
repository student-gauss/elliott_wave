import sys
import csv
import datetime
import numpy as np
import collections
import random
import itertools
#import matplotlib.pyplot as plt
from predictor import CheatPredictor, SimpleNNPredictor, PatternPredictor
from trader import RoteQTrader
from trader import QTrader

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

def getPriceChange(oldPrice, newPrice):
    return float(newPrice - oldPrice) / oldPrice

def trainPredictor(predictor, trainingLoopCount, startIndex, endIndex):
    for i in range(trainingLoopCount):
        testIndex = random.choice(range(startIndex, endIndex))
        for index in range(testIndex, testIndex + 128):
            phiX = predictor.extractFeatures(index)
            currentPrice = predictor.getPrice(index)
            predictor.train(phiX, getPriceChange(currentPrice, predictor.getPrice(index + predictor.predictionDelta)))

def trainTrader(label, trader, maxIndex, traderTrainingLoopCount):
    for i in range(traderTrainingLoopCount):
        startIndex = random.choice(range(0, maxIndex))
        endIndex = min(startIndex + 90, maxIndex)
        
        trader.train(startIndex, endIndex)

def priceGetterForStock(stocks):
    def getPrice(index):
        if index < 0:
            return stocks[0]
        elif index >= len(stocks):
            return stocks[len(stocks) - 1]
        return stocks[index]

    return getPrice

def test():
    stocks = [1, 2, 3, 4, 3, 2]
#    stocks = [3,2,1]
    
    predictor = CheatPredictor(1)
    predictor.getPrice = priceGetterForStock(stocks)
    trainPredictor(predictor, 1, 0, len(stocks))

    trader = QTrader([predictor])
    trader.getPrice = priceGetterForStock(stocks)
    trader.InitialMaxStocksToBuy = 2
    for i in range(50):
        trader.train(0, len(stocks))
        
    print 'Testing'
    gain = trader.test(0, len(stocks))
    print gain
    

def main(predictorTrainingLoopCount, traderTrainingLoopCount):
    # Train predictors
    predictors = [PatternPredictor(1), PatternPredictor(3), PatternPredictor(7)]
#    predictors = [PatternPredictor(1)]
#    predictors = [CheatPredictor(1)]
    for symbol, _, _ in Data:
        # dataToPrice[np.datetime64] := adjusted close price
        # startDate := The first date in the stock data
        # lastDate := The last date in the stock data
        dateToPrice, startDate, lastDate = load(symbol)
        stocks = makeStockArray(dateToPrice, startDate, lastDate)

        for predictor in predictors:
            predictor.getPrice = priceGetterForStock(stocks)
            predictor.startDate = startDate
            trainPredictor(predictor, predictorTrainingLoopCount, 0, len(stocks) - 356)

    # Train traders
    traders = [QTrader(predictors)]
    for symbol, _, _ in Data:
        dateToPrice, startDate, lastDate = load(symbol)

        # Fill up stocks, the prices array, so that we can look up by day-index.
        stocks = makeStockArray(dateToPrice, startDate, lastDate)

        for trader in traders:
            # We learn from the first day up to one year ago.
            trader.getPrice = priceGetterForStock(stocks)
            trainTrader(symbol, trader, len(stocks) - 365, traderTrainingLoopCount)

    # Test traders
    for symbol, _, _ in Data:
        dateToPrice, startDate, lastDate = load(symbol)

        # Fill up stocks, the prices array, so that we can look up by day-index.
        stocks = makeStockArray(dateToPrice, startDate, lastDate)
            
        for trader in traders:
            trader.getPrice = priceGetterForStock(stocks)
            gain = trader.test(len(stocks) - 365, len(stocks))
            print '%s, %d, %10.2f' % (symbol, traderTrainingLoopCount, gain)

# for i in np.arange(1, 5, 0.2):
#     trainingLoopCount = int(10 ** i)
#     main(100, trainingLoopCount)
main(100, 100)
# test()
