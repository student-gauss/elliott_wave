import collections
import json
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
import numpy as np
import random

def getPriceChange(oldPrice, newPrice):
    return float(newPrice - oldPrice) / oldPrice

class Predictor:
    def __init__(self, predictionDelta):
        # The predictor predicts a price delta days later.
        self.predictionDelta = predictionDelta
        self.getPrice = None
        self.startDate = None
    
    def extractFeatures(self, dateIndex): raise NotImplementedError('Override me')
    def train(self, phiX, y): raise NotImplementedError('Override me')
    def predict(self, phiX): raise NotImplementedError('Override me')
    
class CheatPredictor(Predictor):
    def __init__(self, predictionDelta):
        Predictor.__init__(self, predictionDelta)
        self.name = "Cheat Predictor"
        
    def extractFeatures(self, dateIndex):
        return dateIndex

    def train(self, phiX, y):
        # NOOP
        return

    def predict(self, phiX):
        dateIndex = phiX
        currentPrice = self.getPrice(dateIndex)
        return getPriceChange(currentPrice, self.getPrice(dateIndex + self.predictionDelta))

class SimpleNNPredictor(Predictor):
    def __init__(self, predictionDelta):
        Predictor.__init__(self, predictionDelta)
        self.LookBack = [89, 55, 34, 21, 13, 8, 5, 3, 2, 1]
        self.regressor = MLPRegressor(hidden_layer_sizes=(3, 2), activation='tanh', solver='sgd', learning_rate='invscaling')
        self.name = "SimpleNNPredictor"

    def extractFeatures(self, dateIndex):
        history = []
        currentPrice = self.getPrice(dateIndex)
        for xi in self.LookBack:
            history += [getPriceChange(self.getPrice(dateIndex - xi), currentPrice)]
            
        return history

    def train(self, phiX, y):
        X = np.array(phiX).reshape(1, -1)
        Y = [y]
        self.regressor.partial_fit(X, Y)

    def predict(self, phiX):
        X = np.array(phiX).reshape(1, -1)
        result = self.regressor.predict(X)[0]
        return result

class LinearPredictor(Predictor):
    def __init__(self, predictionDelta):
        Predictor.__init__(self, predictionDelta)
        self.LookBack = [89, 55, 34, 21, 13, 8, 5, 3, 2, 1]
        self.regressor = SGDRegressor(alpha=0.05)
        self.name = "LinearPredictor"

    def extractFeatures(self, dateIndex):
        history = []
        currentPrice = self.getPrice(dateIndex)
        for xi in self.LookBack:
            change = getPriceChange(self.getPrice(dateIndex - xi), currentPrice)
            history += [change, change**2]
            
        return history

    def train(self, phiX, y):
        X = np.array(phiX).reshape(1, -1)
        Y = [y]
        self.regressor.partial_fit(X, Y)

    def predict(self, phiX):
        X = np.array(phiX).reshape(1, -1)
        result = self.regressor.predict(X)[0]
        return result
    
class SentimentPredictor(Predictor):
    def __init__(self, predictionDelta, tickerSymbol):
        Predictor.__init__(self, predictionDelta)
        self.name = 'SentimentPredictor'
        self.regressor = SGDRegressor(alpha=0.05)
        self.sentiment = collections.defaultdict(float)
        with open('nytimes/trends_with_sentiment.json', 'r') as jsonFile:
            j = json.load(jsonFile)
            if tickerSymbol in j:
                comments = j[tickerSymbol]['data']
                for comment in comments:
                    if 'sentiment' in comment:
                        self.sentiment[np.datetime64(comment['date'])] = comment['sentiment']

    def extractFeatures(self, dateIndex):
        priorSentiments = []

        # take sentiments of prior 30 days
        for i in range(30):
            date = self.startDate + np.timedelta64(dateIndex - i, 'D')
            priorSentiments += [self.sentiment[date]]

        return priorSentiments
    
    def train(self, phiX, y):
        X = np.array(phiX).reshape(1, -1)
        Y = [y]
        self.regressor.partial_fit(X, Y)

    def predict(self, phiX):
        X = np.array(phiX).reshape(1, -1)
        result = self.regressor.predict(X)[0]
        return result

class PatternPredictor(Predictor):
    def __init__(self, predictionDelta):
        Predictor.__init__(self, predictionDelta)
        self.LookBack = [1, 2, 3, 5, 8, 13, 21, 34]
        self.name = "PatternPredictor"
        self.patterns = collections.defaultdict(int)
        self.trainCount = collections.defaultdict(int)

    def extractFeatures(self, dateIndex):
        pattern = []
        prevPrice = self.getPrice(dateIndex)
        for xi in self.LookBack:
            price = self.getPrice(dateIndex - xi)
            change = getPriceChange(price, prevPrice)
            if change < 0:
                pattern += [-1]
            elif change > 0:
                pattern += [1]
            else:
                pattern += [random.choice([-1, +1])]

            prevPrice = price
            
        return (tuple(pattern), self.getPrice(dateIndex))

    def train(self, phiX, y):
        pattern, currentPrice = phiX

        change = y / abs(y) if y != 0 else 0
        trainCountForPattern = self.trainCount[pattern]
        eta = 1.0 / (trainCountForPattern + 1)
        currentValue = self.patterns[pattern]
        self.patterns[pattern] = (1 - eta) * currentValue + eta * change
        self.trainCount[pattern] += 1
    
    def predict(self, phiX):
        pattern, _ = phiX
        return self.patterns[pattern]
    
