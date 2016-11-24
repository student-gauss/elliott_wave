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
        priorPattern, currentAssets = state

        return priorPattern
        
    def train(self, phiX, target):
        self.hasFitted = True
        X = np.array(phiX).reshape(1, -1)
        self.regressor.partial_fit(X, [target])

    def predict(self, phiX):
        if self.hasFitted:
            X = np.array(phiX).reshape(1, -1)
            return self.regressor.predict(X)[0]
        else:
            return 0.0
        
        
