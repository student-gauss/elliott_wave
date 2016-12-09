import collections
import numpy as np
import random

class Trader:
    def __init__(self, predictor, getPrice):
        self.predictor = predictor
        self.getPrice = getPrice

    def train(self, startIndex, endIndex):raise NotImplementedError('Override me')
    def test(self, startIndex, endIndex):raise NotImplementedError('Override me')

class RotQTrader(Trader):
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
        return (ownedStocks, maxStocksToBuy, prediction, action)

    def getQopt(self, state, action):
        return self.Qopt[self.getQoptKey(state, action)]
    
    def getVoptAndAction(self, state):
        QoptAndActionList = []
        for action in self.getActions(state):
            Qopt = self.getQopt(state, action)
            QoptAndActionList += [(Qopt, action)]

        return max(QoptAndActionList)

    def takeAction(self, state, action, index):
        ownedStocks, maxStocksToBuy, prediction = state

        budget = maxStocksToBuy * self.getPrice(index)
        budget += -action * self.getPrice(index)
        maxStocksToBuy = float(budget) / self.getPrice(index + 1)
        ownedStocks += action

        # update prediction
        prediction = self.getPrediction(index + 1)

        return (ownedStocks, maxStocksToBuy, prediction)

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

            s_prime = self.takeAction(state, action, index)

            if index + 1 == endIndex:
                ownedStocks, maxStocksToBuy, _ = state
                reward = self.getPrice(index + 1) * (ownedStocks + maxStocksToBuy)
            else:
                reward = 0

            self.update(state, action, reward, s_prime)

            state = s_prime

    def test(self, startIndex, endIndex):

        stat = collections.defaultdict(int)
        for key, value in self.Qopt.iteritems():
            ownedStocks, maxStocksToBuy, prediction, action = key
            print 'ownedStocks: %4d, maxStocksToBuy: %4.2f, prediction: %2d, action: %2d -> %4.2f' % (ownedStocks, maxStocksToBuy, prediction, action, value)

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
            s_prime = self.takeAction(state, action, index)
            print 'Pick optimal action from state = %s, action = %s, s_prime: %s' % (state, action, s_prime)
            state = s_prime

        ownedStocks, maxStocksToBuy, prediction = state
        return (ownedStocks + maxStocksToBuy) * self.getPrice(endIndex) - self.InitialMaxStocksToBuy * self.getPrice(startIndex)
