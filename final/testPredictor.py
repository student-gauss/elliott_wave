import sys
import csv
import datetime
import numpy as np
import collections
import random
import itertools
import matplotlib.pyplot as plt
from predictor import CheatPredictor, SimpleNNPredictor, LinearPredictor,SentimentPredictor

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

def testPredictor(label, predictor, startIndex, endIndex):
    # Test the learned weight performance by exercising in the
    # last one year.
    tp = 0.00001
    tn = 0.00001
    fp = 0.00001
    fn = 0.00001
    for index in range(startIndex, endIndex):
        phiX = predictor.extractFeatures(index)
        y_prime = predictor.predict(phiX)
        currentPrice = predictor.getPrice(index)
        y = getPriceChange(currentPrice, predictor.getPrice(index + predictor.predictionDelta))

        if y <= 0 and y_prime <= 0:
            tn += 1

        if y <= 0 and y_prime > 0:
            fp += 1

        if y > 0 and y_prime > 0:
            tp += 1

        if y > 0 and y_prime <= 0:
            fn += 1

    return np.array([tp, tn, fp, fn])

def trainPredictor(predictor, trainingLoopCount, startIndex, endIndex):
    for i in range(trainingLoopCount):
        testIndex = random.choice(range(startIndex, endIndex))
        for index in range(testIndex, testIndex + 128):
            phiX = predictor.extractFeatures(index)
            currentPrice = predictor.getPrice(index)
            predictor.train(phiX, getPriceChange(currentPrice, predictor.getPrice(index + predictor.predictionDelta)))

def priceGetterForStock(stocks):
    def getPrice(index):
        if index < 0:
            return stocks[0]
        elif index >= len(stocks):
            return stocks[len(stocks) - 1]
        return stocks[index]

    return getPrice

outputFile = open('predictor_perform.csv', 'w')

def main(trainingLoopCount):
#    predictors = [SimpleNNPredictor(1), SimpleNNPredictor(3), SimpleNNPredictor(7), LinearPredictor(1), LinearPredictor(3), LinearPredictor(7)]
    predictors = [SentimentPredictor(1, 'aapl')]
    for key, _, _ in Data:
        dateToPrice, startDate, lastDate = load(key)

        # Fill up stocks, the prices array, so that we can look up by day-index.
        stocks = makeStockArray(dateToPrice, startDate, lastDate)

        for predictor in predictors:
            predictor.getPrice = priceGetterForStock(stocks)
            predictor.startDate = startDate
            trainPredictor(predictor, trainingLoopCount, 0, len(stocks) - 356)

    for predictor in predictors:
        # [TP, TN, FP, FN]
        result = np.zeros(4)
        for key, _, _ in Data:
            dateToPrice, startDate, lastDate = load(key)
            stocks = makeStockArray(dateToPrice, startDate, lastDate)
            predictor.getPrice = priceGetterForStock(stocks)
            result += testPredictor(key, predictor, len(stocks) - 356, len(stocks))

        tp, tn, fp, fn = result
    
        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        
        tableStr=r'''
\begin{table}[h]
  \begin{tabular}{cc}
    \begin{tabular}{cc|cc}
      & & \multicolumn{2}{c}{Predicted} \\
      & & $+ $ & $-$ \\
      \hline
      \multirow{2}{*}{Actual}
      & $+$ & %4.2f & %4.2f \\
      & $-$ & %4.2f & %4.2f \\
      \hline
    \end{tabular}
    &
    \begin{tabular}{cc}
      Accuracy & %4.2f \\
      F1 Score & %4.2f \\
    \end{tabular}
  \end{tabular}
  \caption{%s with %d days delta, training loop %d}
\end{table}
''' % (tp / total, fn / total, fp / total, tn / total, (tp + tn) / total, f1,
       predictor.name, predictor.predictionDelta, trainingLoopCount
)
        log = '%24s %5d %5d %4.2f %4.2f %4.2f %4.2f %4.2f %4.2f\n' % (predictor.name, predictor.predictionDelta, trainingLoopCount, tp / total, fn / total, fp / total, tn / total, (tp + tn) / total, f1)
        print log
        
        outputFile.write(log)
        outputFile.flush()
#        print tableStr


for i in np.arange(0, 4, 0.2):
    trainingLoopCount = int(10 ** i)
    main(trainingLoopCount)
