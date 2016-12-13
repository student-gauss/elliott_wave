import collections
import numpy as np
import random
from scipy.special import expit
from sklearn.neural_network import MLPRegressor
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

class Trader:
    def __init__(self, predictors):
        self.predictors = predictors
        self.getPrice = None

    def train(self, startIndex, endIndex):raise NotImplementedError('Override me')
    def test(self, startIndex, endIndex):raise NotImplementedError('Override me')

def stateStr(state):
    ownedStocks, maxStocksToBuy, prediction = state
    shape, residual = prediction
    return 'o: %5.2f c: %5.2f m: %5.2f r: %4.2f' % (ownedStocks, maxStocksToBuy, shape, residual)
    
class QTrader(Trader):
    def __init__(self, predictors):
        Trader.__init__(self, predictors)
        
        self.Epsilon = 0.9
        self.Gamma = 0.9
        self.InitialMaxStocksToBuy = 10.0
        self.weights = np.zeros(4)
        self.hasFitted = False
        self.numOfUpdates = 0
    
    def getPrediction(self, index):
        predictionDelta = [0.0]
        futurePriceChanges = [0.0]

        for predictor in self.predictors:
            phiX = predictor.extractFeatures(index)
            futurePriceChanges += [predictor.predict(phiX)]
            predictionDelta += [predictor.predictionDelta]
        
        shape, residual, _, _, _ = np.polyfit(predictionDelta, futurePriceChanges, 1, full=True)
        if len(residual) == 0:
            residual = [0]

        return (shape[0], residual[0])

    def initState(self, index):
        # Suppose we have budget to buy InitialMaxStocksToBuy stocks initially.
        return (0, self.InitialMaxStocksToBuy, self.getPrediction(index))

    def getActions(self, state):
        ownedStocks, maxStocksToBuy, _ = state
        return range(-ownedStocks, int(maxStocksToBuy) + 1)

    def extractFeatures(self, state, action):
        ownedStocks, maxStocksToBuy, prediction = state
        shape, residual = prediction

        normalizedAsset = float(ownedStocks) / (ownedStocks + maxStocksToBuy)
        
        return np.array([
            (action ** 1) * (shape ** 1)
            ,(action ** 2) * (shape ** 1)
            ,(action ** 1) * (shape ** 2)
            ,(action ** 2) * (shape ** 2)
        ])

    def getQopt(self, phiX):
        Qopt = np.dot(phiX, self.weights)
        return Qopt
        
    def getVoptAndAction(self, state, debug=False):
        QoptAndActionList = []
        for action in self.getActions(state):
            phiX = self.extractFeatures(state, action)
            Qopt = self.getQopt(phiX)
            QoptAndActionList += [(Qopt, action)]

            if debug:
                print 'action = %4.2f Qopt = %4.2f' % (action, Qopt)
        return max(QoptAndActionList)

    def getAssetValue(self, state, index):
        ownedStocks, maxStocksToBuy, _ = state
        stockPrice = self.getPrice(index)
        return (ownedStocks + maxStocksToBuy) * stockPrice
        
    def takeAction(self, state, action, index):
        ownedStocks, maxStocksToBuy, _ = state

        budget = maxStocksToBuy * self.getPrice(index)
        budget += -action * self.getPrice(index)
        maxStocksToBuy = float(budget) / self.getPrice(index + 1)
        ownedStocks += action

        # update prediction
        prediction = self.getPrediction(index + 1)
        s_prime = (ownedStocks, maxStocksToBuy, prediction)
        reward = self.getAssetValue(s_prime, index + 1) - self.getAssetValue(state, index)
        return (s_prime, reward)
    
    def update(self, state, action, reward, s_prime):
        Vopt, _ = self.getVoptAndAction(s_prime)
        target = reward + self.Gamma * Vopt
        phiX = self.extractFeatures(state, action)
        eta = 1.0 / (self.numOfUpdates + 1)
        newWeights = self.weights - eta * (self.getQopt(phiX) - target) * phiX
        # if self.numOfUpdates % 10000 == 0:
        #     print '%10.8f' % ((self.getQopt(phiX) - target) ** 2)
        #     print self.weights
        self.weights = newWeights
        self.numOfUpdates += 1

    def train(self, startIndex, endIndex):
        state = None

        for index in range(startIndex, endIndex):
            if state == None:
                state = self.initState(index)

            actions = self.getActions(state)
            if len(actions) == 1:
                # No stocks to sell, No money to buy
                break

            if random.random() < self.Epsilon:
                action = random.choice(actions)
            else:
                # pick optimal action
                _, action = self.getVoptAndAction(state)

            s_prime, reward = self.takeAction(state, action, index)
            
            self.update(state, action, reward, s_prime)

            state = s_prime

    def test(self, startIndex, endIndex):
        state = None
        for index in range(startIndex, endIndex):
            if state == None:
                state = self.initState(index)

            actions = self.getActions(state)
            if len(actions) == 1:
                # No stocks to sell, No money to buy
                break

            # pick optimal action
            _, action = self.getVoptAndAction(state, debug=False)
            s_prime, reward = self.takeAction(state, action, index)
#            print 'Pick optimal action from state = %s, action = %3d, reward = %6.2f s_prime: %s' % (stateStr(state), action, reward, stateStr(s_prime))
            state = s_prime

        ownedStocks, maxStocksToBuy, _ = state
        return (ownedStocks + maxStocksToBuy) * self.getPrice(endIndex) - self.InitialMaxStocksToBuy * self.getPrice(startIndex)


# ===============================================================    
# RoteQTrader(Trader):
# ===============================================================    
class RoteQTrader(Trader):
    def __init__(self, predictors):
        Trader.__init__(self, predictors)
        
        self.Epsilon = 0.9
        self.Gamma = 1.0
        self.InitialMaxStocksToBuy = 10.0
        self.Qopt = collections.defaultdict(float)
        self.updateCount = collections.defaultdict(float)

    def getPrediction(self, index):
        predictionDelta = [0.0]
        futurePriceChanges = [0.0]

        for predictor in self.predictors:
            phiX = predictor.extractFeatures(index)
            futurePriceChanges += [predictor.predict(phiX)]
            predictionDelta += [predictor.predictionDelta]
        
        m, _ = np.polyfit(predictionDelta, futurePriceChanges, 1)
#        print 'Prediction. Future Prices: %s, slope: %f' % (futurePrices, m)
        if m < -0.01:
            return -1
        elif m > 0.01:
            return 1
        else:
            return 0
        
    def initState(self, index):
        # Suppose we have budget to buy InitialMaxStocksToBuy stocks initially.
        return (0, self.InitialMaxStocksToBuy, self.getPrediction(index))

    def getActions(self, state):
        ownedStocks, maxStocksToBuy, _ = state
        return range(-ownedStocks, int(maxStocksToBuy) + 1)

    def getQoptKey(self, state, action):
        ownedStocks, maxStocksToBuy, prediction = state
        return (ownedStocks, int(maxStocksToBuy), prediction, action)

    def getQopt(self, state, action):
        return self.Qopt[self.getQoptKey(state, action)]
    
    def getVoptAndAction(self, state):
        QoptAndActionList = []
        for action in self.getActions(state):
            Qopt = self.getQopt(state, action)
            QoptAndActionList += [(Qopt, action)]

        return max(QoptAndActionList)

    def getAssetValue(self, state, index):
        ownedStocks, maxStocksToBuy, prediction = state
        stockPrice = self.getPrice(index)

        return (ownedStocks + maxStocksToBuy) * stockPrice
        
    def takeAction(self, state, action, index):
        ownedStocks, maxStocksToBuy, prediction = state

        budget = maxStocksToBuy * self.getPrice(index)
        budget += -action * self.getPrice(index)
        maxStocksToBuy = float(budget) / self.getPrice(index + 1)
        ownedStocks += action

        # update prediction
        prediction = self.getPrediction(index + 1)

        s_prime = (ownedStocks, maxStocksToBuy, prediction)
        reward = self.getAssetValue(s_prime, index + 1) - self.getAssetValue(state, index)
        return (s_prime, reward)

    def update(self, state, action, reward, s_prime):
        Vopt, _ = self.getVoptAndAction(s_prime)
        eta = 1 / (self.updateCount[self.getQoptKey(state, action)] + 1.0)

        old = self.getQopt(state, action)
        self.Qopt[self.getQoptKey(state, action)] -= eta * (self.getQopt(state, action) - (reward + self.Gamma * Vopt))

#        print 'Update Qopt(%s): %s -> %s' % (self.getQoptKey(state, action), old, self.getQopt(state, action))
        
        self.updateCount[self.getQoptKey(state, action)] += 1
        
    def train(self, startIndex, endIndex):
        state = None

        for index in range(startIndex, endIndex):
            if state == None:
                state = self.initState(index)

            actions = self.getActions(state)
            if len(actions) == 1:
                # No stocks to sell, No money to buy
                break

            if random.random() < self.Epsilon:
                action = random.choice(actions)
            else:
                # pick optimal action
                _, action = self.getVoptAndAction(state)

            s_prime, reward  = self.takeAction(state, action, index)

            self.update(state, action, reward, s_prime)

            state = s_prime

    def test(self, startIndex, endIndex):

        stat = collections.defaultdict(int)
        for key, value in self.Qopt.iteritems():
            ownedStocks, maxStocksToBuy, prediction, action = key
#            print 'ownedStocks: %4d, maxStocksToBuy: %4.2f, prediction: %2d, action: %2d -> %4.2f' % (ownedStocks, maxStocksToBuy, prediction, action, value)

        state = None
        for index in range(startIndex, endIndex):
            if state == None:
                state = self.initState(index)

            actions = self.getActions(state)
            if len(actions) == 1:
                # No stocks to sell, No money to buy
                break

            # pick optimal action
            _, action = self.getVoptAndAction(state)
            s_prime, reward = self.takeAction(state, action, index)
#            print 'Pick optimal action from state = %s, action = %s, reward = %4.2f, s_prime: %s' % (state, action, reward, s_prime)
            state = s_prime

        ownedStocks, maxStocksToBuy, prediction = state
        return (ownedStocks + maxStocksToBuy) * self.getPrice(endIndex) - self.InitialMaxStocksToBuy * self.getPrice(startIndex)
