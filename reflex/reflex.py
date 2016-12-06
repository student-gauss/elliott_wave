import sys
import csv
import datetime
import numpy as np
import collections
import random
import itertools
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
import numpy as np
import matplotlib.pyplot as plt

LookBack = [87, 54, 33, 21, 13, 8, 5, 3, 2, 1]
Data = [
    ('dj', None, None),
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

def extractFeatures(stocks, index):
    history = []
    currentPrice = stocks[index]
    for xi in range(len(LookBack)):
        lookBackIndex = max(index - LookBack[xi], 0)
        history += [float(currentPrice - stocks[lookBackIndex]) / currentPrice]

    return history

def learn(label, stocks, regressor, errorHistory):
    randomIndex = range(len(stocks))
    random.shuffle(randomIndex)
    for index in randomIndex:
        phiX = extractFeatures(stocks, index)
        X = np.array(phiX).reshape(1, -1)
        
        currentPrice = stocks[index]
        y = float(stocks[min(index + 7, len(stocks) - 1)] - currentPrice) / currentPrice
        
        regressor.partial_fit(X, [y])
        
        errorHistory += [regressor.predict(X) - y]

def test(label, stocks, regressor):
    total = 0.0
    tp = 0.00001
    tn = 0.00001
    fp = 0.00001
    fn = 0.00001
    
    for index in range(len(stocks)):
        phiX = extractFeatures(stocks, index)
        X = np.array(phiX).reshape(1, -1)
        
        currentPrice = stocks[index]
        y = float(stocks[min(index + 7, len(stocks) - 1)] - currentPrice) / currentPrice

        y_prime = regressor.predict(X)
        if y <= 0 and y_prime <= 0:
            tn += 1

        if y <= 0 and y_prime > 0:
            fp += 1

        if y > 0 and y_prime > 0:
            tp += 1

        if y > 0 and y_prime <= 0:
            fn += 1
            
        total += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
        
    print label
    print 'TP: %3f FP: %3f TN: %3f FN: %3f' % (tp / total, fp / total, tn / total, fn / total)
    print 'F1 score: %f' % f1
    print ''
    
def main():

    for key, _, _ in Data:
        regressor = MLPRegressor(hidden_layer_sizes=(3,3), activation='tanh', solver='sgd', learning_rate='invscaling')
#        print regressor.get_params()
        
        # dataToPrice[np.datetime64] := adjusted close price
        # startDate := The first date in the stock data
        # lastDate := The last date in the stock data
        dateToPrice, startDate, lastDate = load(key)
        
        # Fill up stocks, the prices array, so that we can look up by day-index.
        stocks = makeStockArray(dateToPrice, startDate, lastDate)

        # We learn from the first day up to one year ago.
        stocksToLearn = stocks[0:-365]
    
        # And test the learned weight performance by exercising in the
        # last one year.
        stocksToTest = stocks[-365:]

        errorHistory = []
        for i in range(5):
            learn(key, stocksToLearn, regressor, errorHistory)

        plt.plot(errorHistory)
        plt.savefig('error_%s.png' % key)
        plt.close()

        plt.plot(stocksToTest)
        plt.savefig('stock_%s.png' % key)
        plt.close()
        
        test(key, stocksToTest, regressor)
        
main()
