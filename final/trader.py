import collections
import numpy as np
import random
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Trader:
    def __init__(self, predictor, getPrice):
        self.predictor = predictor
        self.getPrice = getPrice

    def train(self, startIndex, endIndex):raise NotImplementedError('Override me')
    def test(self, startIndex, endIndex):raise NotImplementedError('Override me')

def stateStr(state):
    ownedStocks, maxStocksToBuy, prediction = state
    return '%4.2f %4.2f %5.2f' % (ownedStocks, maxStocksToBuy, prediction)
    
class QTrader(Trader):
    def __init__(self, predictor, getPrice):
        Trader.__init__(self, predictor, getPrice)
        
        self.Epsilon = 0.9
        self.Gamma = 0.9
        self.InitialMaxStocksToBuy = 10.0
        self.regressor = MLPRegressor(hidden_layer_sizes=(2,), activation='tanh', solver='sgd', learning_rate='invscaling')
        self.hasFitted = False
        self.statPrediction = []
        self.statAction = []
        self.statQopt = []
    
    def getPrediction(self, index):
        phiX = self.predictor.extractFeatures(index)
        futurePrices = [0]
        futurePrices += self.predictor.predict(phiX)
        
        m, _ = np.polyfit([0] + self.predictor.predictingDelta, futurePrices, 1)
        return m

    def initState(self, index):
        # Suppose we have budget to buy InitialMaxStocksToBuy stocks initially.
        return (0, self.InitialMaxStocksToBuy, self.getPrediction(index))

    def getActions(self, state):
        ownedStocks, maxStocksToBuy, _ = state
        return range(-ownedStocks, int(maxStocksToBuy) + 1)

    def extractFeatures(self, state, action):
        ownedStocks, maxStocksToBuy, prediction = state
        
        normalizedAsset = float(ownedStocks) / (ownedStocks + maxStocksToBuy)
        normalizedAction = float(action) / self.InitialMaxStocksToBuy
        
        return [normalizedAsset, normalizedAction, prediction]
    
    def getVoptAndAction(self, state):
        QoptAndActionList = []
        for action in self.getActions(state):
            phiX = self.extractFeatures(state, action)
            X = np.array(phiX).reshape(1, -1)

            if self.hasFitted:
                Qopt = self.regressor.predict(X)[0]
            else:
                Qopt = 0.0;

            QoptAndActionList += [(Qopt, action)]

        print QoptAndActionList
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
        target = reward + self.Gamma * Vopt
        phiX = self.extractFeatures(state, action)
        X = np.array(phiX).reshape(1, -1)

        self.regressor.partial_fit(X, [target])
        self.hasFitted = True

        self.statPrediction += [state[2]]
        self.statAction += [action]
        self.statQopt += [target]

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
            _, action = self.getVoptAndAction(state)
            s_prime, reward = self.takeAction(state, action, index)
            print 'Pick optimal action from state = %s, action = %s, reward = %4.2f s_prime: %s' % (state, action, reward, s_prime)
            state = s_prime

        fig = plt.figure()
        ax = fig.gca(projection='3d')
#        ax.scatter(self.statPrediction, self.statAction, self.statQopt)
#        plt.show()
        
        ownedStocks, maxStocksToBuy, prediction = state
        return (ownedStocks + maxStocksToBuy) * self.getPrice(endIndex) - self.InitialMaxStocksToBuy * self.getPrice(startIndex)


# ===============================================================    
# RoteQTrader(Trader):
# ===============================================================    
class RoteQTrader(Trader):
    def __init__(self, predictor, getPrice):
        Trader.__init__(self, predictor, getPrice)
        
        self.Epsilon = 0.9
        self.Gamma = 1.0
        self.InitialMaxStocksToBuy = 10.0
        self.Qopt = collections.defaultdict(float)
        self.updateCount = collections.defaultdict(float)

    def getPrediction(self, index):
        phiX = self.predictor.extractFeatures(index)
        futurePrices = [0]
        futurePrices += self.predictor.predict(phiX)
        
        m, _ = np.polyfit([0] + self.predictor.predictingDelta, futurePrices, 1)
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
