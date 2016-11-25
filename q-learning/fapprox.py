from sklearn.neural_network import MLPRegressor
import numpy as np

class Learner:
    def extractFeatures(state, action): raise NotImplementedError('Override me')
    def train(phiX, target): raise NotImplementedError('Override me')
    def predict(phiX): raise NotImplementedError('Override me')

class SimpleNNLearner(Learner):
    def __init__(self):
        self.regressor = MLPRegressor()
        self.hasFitted = False

    def extractFeatures(self, state, action):
        # prior pattern: [0, 1, 0, ...]
        # currentAssets: [(priceDiff, numStocks), ...]
        currentPrice, history, currentAssets = state

        features = []
        for priorPrice in history:
            features += [float(priorPrice - currentPrice) / currentPrice]
        
        priceDiffThreshold = 0.01
        while priceDiffThreshold < 128:
            nextThreshold = priceDiffThreshold * 2.0
            numStocks = 0
            for asset in currentAssets:
                purchasePrice, quantity = asset
                priceDiff = purchasePrice - currentPrice
                if priceDiffThreshold <= priceDiff and priceDiff < nextThreshold:
                    numStocks += quantity
            features += [numStocks]
            priceDiffThreshold = nextThreshold

        priceDiffThreshold = -0.01
        while priceDiffThreshold > -128:
            nextThreshold = priceDiffThreshold * 2.0
            numStocks = 0
            for asset in currentAssets:
                purchasePrice, quantity = asset
                priceDiff = purchasePrice - currentPrice
                if nextThreshold <= priceDiff and priceDiff < priceDiffThreshold:
                    numStocks += quantity
            features += [numStocks]
            priceDiffThreshold = nextThreshold

        features += [action[0]]
        return features
        
    def train(self, phiX, target):
        self.hasFitted = True
        X = np.array(phiX).reshape(1, -1)
        self.regressor.partial_fit(X, [target])

    def predict(self, phiX):
        result = 0
        if self.hasFitted:
            X = np.array(phiX).reshape(1, -1)
            result = self.regressor.predict(X)[0]
        return result
        
        
