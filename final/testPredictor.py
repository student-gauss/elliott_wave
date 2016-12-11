import sys
import csv
import datetime
import numpy as np
import collections
import random
import itertools
import matplotlib.pyplot as plt
from predictor import CheatPredictor, SimpleNNPredictor, LinearPredictor

Data = [
    # ('dj', None, None),
    # ('gdx', None, None),
    # ('qcom', None, None),
    # ('rut', None, None),
    # ('wmt', None, None),
    # ('hd', None, None),
    # ('low', None, None),
    # ('tgt', None, None),
    # ('cost', None, None),
    # ('nke', None, None),
    # ('ko', None, None),
    # ('xom', None, None),
    # ('cvx', None, None),
    # ('cop', None, None),
    # ('bp', None, None),
    # ('ibm', None, None),
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

def testPredictor(label, predictor, maxIndex):
    # We learn from the first day up to one year ago.
    for i in range(10):
        for index in range(maxIndex - 365):
            phiX = predictor.extractFeatures(index)
            currentPrice = predictor.getPrice(index)
            predictor.train(phiX, [getPriceChange(currentPrice, predictor.getPrice(index + delta)) for delta in predictor.predictingDelta])

    # And test the learned weight performance by exercising in the
    # last one year.
    tp = 0.00001
    tn = 0.00001
    fp = 0.00001
    fn = 0.00001
    for index in range(maxIndex - 365, maxIndex):
        phiX = predictor.extractFeatures(index)
        y_prime = predictor.predict(phiX)
        currentPrice = predictor.getPrice(index)
        y = [getPriceChange(currentPrice, predictor.getPrice(index + delta)) for delta in predictor.predictingDelta]

        predictedMeanGain = sum(y_prime) / len(y_prime)
        trueMeanGain = sum(y) / len(y)

        if trueMeanGain <= 0 and predictedMeanGain <= 0:
            tn += 1

        if trueMeanGain <= 0 and predictedMeanGain > 0:
            fp += 1

        if trueMeanGain > 0 and predictedMeanGain > 0:
            tp += 1

        if trueMeanGain > 0 and predictedMeanGain <= 0:
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    print label
    print predictor
    print 'TP: %3f FP: %3f TN: %3f FN: %3f' % (tp / total, fp / total, tn / total, fn / total)
    print 'F1 score: %f' % f1
    print ''

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

        predictors = [LinearPredictor(getPrice)]
        for predictor in predictors:
            testPredictor(key, predictor, len(stocks))

main()
