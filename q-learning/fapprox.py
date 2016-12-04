from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
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
        # currentPrice: Float
        # history: [0, 1, 0, ...]
        # ownedStocks: Int
        currentPrice, history, ownedStocks, cash = state

        features = []
        for priorPrice in history:
            features += [float(priorPrice - currentPrice) / currentPrice]

        features += [ownedStocks]
        features += [cash]

        features += [action]
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
        
class SimpleSGDLearner(Learner):
    def __init__(self):
        self.regressor = SGDRegressor()
        self.hasFitted = False

    def extractFeatures(self, state, action):
        # currentPrice: Float
        # history: [0, 1, 0, ...]
        # ownedStocks: Int
        currentPrice, history, ownedStocks, cash = state

        features = []
        for priorPrice in history:
            features += [float(priorPrice - currentPrice) / currentPrice]

        features += [ownedStocks]
        features += [cash]
        features += [action]
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
