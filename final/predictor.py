from sklearn.neural_network import MLPRegressor
import numpy as np

def getPriceChange(oldPrice, newPrice):
    return float(newPrice - oldPrice) / oldPrice

class Predictor:
    def __init__(self, getPrice):
        # The predictor predicts a price 1, 7, 30, and 90 days later.
#        self.predictingDelta = [1, 7, 30, 90]
        self.predictingDelta = [1, 2]
        self.getPrice = getPrice
    
    def extractFeatures(self, dateIndex): raise NotImplementedError('Override me')
    def train(self, phiX, y): raise NotImplementedError('Override me')
    def predict(self, phiX): raise NotImplementedError('Override me')
    
class CheatPredictor(Predictor):
    def extractFeatures(self, dateIndex):
        return [dateIndex]

    def train(self, phiX, y):
        # NOOP
        return

    def predict(self, phiX):
        dateIndex = phiX[0]
        currentPrice = self.getPrice(dateIndex)

        y_prime = []
        for delta in self.predictingDelta:
            y_prime += [getPriceChange(currentPrice, self.getPrice(delta + dateIndex))]

        return y_prime

class SimpleNNPredictor(Predictor):
    def __init__(self, getPrice):
        Predictor.__init__(self, getPrice)
        self.LookBack = [87, 54, 33, 21, 13, 8, 5, 3, 2, 1]
        self.regressor = MLPRegressor(hidden_layer_sizes=(3,), activation='tanh', solver='sgd', learning_rate='invscaling')

    def extractFeatures(self, dateIndex):
        history = []
        currentPrice = self.getPrice(dateIndex)
        for xi in range(len(self.LookBack)):
            history += [getPriceChange(self.getPrice(dateIndex - xi), currentPrice)]
            
        return history

    def train(self, phiX, y):
        X = np.array(phiX).reshape(1, -1)
        Y = np.array(y).reshape(1, -1)

        self.regressor.partial_fit(X, Y)

    def predict(self, phiX):
        X = np.array(phiX).reshape(1, -1)
        return self.regressor.predict(X)[0]
