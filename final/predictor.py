import collections
import json
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
import numpy as np

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
        Predictor.__init__(predictionDelta)
        self.name = "Cheat Predictor"
        
    def extractFeatures(self, dateIndex):
        return [dateIndex]

    def train(self, phiX, y):
        # NOOP
        return

    def predict(self, phiX):
        dateIndex = phiX[0]
        currentPrice = self.getPrice(dateIndex)
        return getPriceChange(currentPrice, self.getPrice(delta + self.predictionDelta))

class SimpleNNPredictor(Predictor):
    def __init__(self, predictionDelta):
        Predictor.__init__(self, predictionDelta)
        self.LookBack = [87, 54, 33, 21, 13, 8, 5, 3, 2, 1]
        self.regressor = MLPRegressor(hidden_layer_sizes=(3, 2), activation='tanh', solver='sgd', learning_rate='invscaling')
        self.name = "SimpleNNPredictor"

    def extractFeatures(self, dateIndex):
        history = []
        currentPrice = self.getPrice(dateIndex)
        for xi in range(len(self.LookBack)):
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
        self.LookBack = [87, 54, 33, 21, 13, 8, 5, 3, 2, 1]
        self.LookBack = [5, 3, 2, 1]
        self.regressor = SGDRegressor(alpha=0.05)
        self.name = "LinearPredictor"

    def extractFeatures(self, dateIndex):
        history = []
        currentPrice = self.getPrice(dateIndex)
        for xi in range(len(self.LookBack)):
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
