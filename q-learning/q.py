import sys
import csv
import datetime
import numpy as np
import collections
import random
import itertools
import matplotlib.pyplot as plt
import fapprox

Gamma = 0.9
LookBack = [87, 54, 33, 21, 13, 8, 5, 3, 2, 1]
Epsilon = 0.5
Data = [
    #('dj', None, None),
    # ('gdx', None, None),
    # ('qcom', None, None),
    # ('rut', None, None),
    # ('wmt', None, None),
    # ('hd', None, None),
    # ('low', None, None),
    # ('tgt', None, None),
    # ('cost', None, None),
    # ('nke', None, None),
    # ('ko', None, None),
    # ('xom', None, None),
    # ('cvx', None, None),
    # ('cop', None, None),
    # ('bp', None, None),
    # ('ibm', None, None),
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

# Utility function to return the stock price for the day. It returns
# the last day price if the stock data for the day is not available
# (e.g. holiday)
def stockForDate(dateToPrice, date):
    while not date in dateToPrice:
        date = date - np.timedelta64(1, 'D')
    return dateToPrice[date]

def makeStockArray(dateToPrice, startDate, lastDay):
    stocks = []
    date = startDate
    while date <= lastDate:
        stocks += [stockForDate(dateToPrice, date)]
        date += np.timedelta64(1, 'D')
    return stocks

# Initial state.
def initState(initialPrice):
    history = [0] * (len(LookBack))
    ownedStocks = 0
    return tuple([initialPrice, history, ownedStocks])

def moveStateToIndex(currentState, stocks, newIndex):
    if newIndex == 0:
        return currentState
    
    history = []
    for xi in range(len(LookBack)):
        lookBackIndex = max(newIndex - LookBack[xi], 0)
        history += [stocks[lookBackIndex]]

    _, _, ownedStocks = currentState
    return tuple([stocks[newIndex], history, ownedStocks])

def buyStock(currentState, numberOfStocksToBuy):
    if numberOfStocksToBuy == 0:
        return currentState

    currentPrice, history, ownedStocks = currentState

#    print 'Buy %d -> %d' % (numberOfStocksToBuy, ownedStocks)
    return tuple([currentPrice, history, ownedStocks + numberOfStocksToBuy])

def sellStock(currentState, numberOfStocksToSell):
    currentPrice, history, ownedStocks = currentState

    newOwnedStocks = ownedStocks - numberOfStocksToSell
#    print 'Sell %d -> %d ' % (numberOfStocksToSell, ownedStocks)
    
    return tuple([currentPrice, history, newOwnedStocks])

def getActions(state):
    actions = []

    currentPrice, history, ownedStocks = state
    for numberOfStocksToBuy in range(0, 2 + 1):
        for numberOfStocksToSell in range(0, ownedStocks + 1):
            if numberOfStocksToBuy != numberOfStocksToSell:
                actions += [(numberOfStocksToBuy, numberOfStocksToSell)]

    actions += [(0, 0)]
    return actions

errorHistory = []
def trainLearner(state, action, reward, s_prime, learner):
    V_opt, _ = getVoptAndAction(s_prime, learner)
    target = reward + Gamma * V_opt
    
    errorHistory.append(learner.predict(learner.extractFeatures(state, action)) - target)
    
    learner.train(learner.extractFeatures(state, action), target)

def getVoptAndAction(state, learner):
    QoptAndActionList = []
    for action in getActions(state):
        phiX = learner.extractFeatures(state, action)
        Qopt = learner.predict(phiX)
        QoptAndActionList += [(Qopt, action)]

    return max(QoptAndActionList)
    
def takeAction(state, action):
    currentPrice, history, ownedStocks = state
    reward = 0
    s_prime = state

    numberOfStocksToBuy, numberOfStocksToSell = action
    
    # Buy
    reward -= currentPrice * numberOfStocksToBuy
    s_prime = buyStock(s_prime, numberOfStocksToBuy)

    # Sell
    reward += currentPrice * numberOfStocksToSell
    s_prime = sellStock(s_prime, numberOfStocksToSell)
    return (reward, s_prime)

def learn(stocks, learner):
    state = None
    totalReward = 0
    for index in range(len(stocks) - 1):
        if state == None:
            state = initState(stocks[index])
        
        actions = getActions(state)
        if random.random() < Epsilon:
            # pick random 
            action = random.choice(actions)
        else:
            # pick optimal action
            _, action = getVoptAndAction(state, learner)

        reward, s_prime = takeAction(state, action)
        totalReward += reward

#        print '%s -> %s: Reward %d' % (actions, action, reward)
#        _, _, currentAssets = s_prime
#        print 'Reward %d, assets: %d' % (reward, len(currentAssets))
        
        s_prime = moveStateToIndex(s_prime, stocks, index + 1)
        trainLearner(state, action, reward, s_prime, learner)
        state = s_prime
        
    print "In-test reward: ", totalReward
    
    
def test(stocks, learner):
    state = None
    totalReward = 0
    for index in range(len(stocks) - 1):
        if state == None:
            state = initState(stocks[index])

        actions = getActions(state)
        # optimal action
        _, action = getVoptAndAction(state, learner)
        reward, s_prime = takeAction(state, action)
        totalReward += reward

        s_prime = moveStateToIndex(s_prime, stocks, index + 1)
        state = s_prime

    return totalReward

for key, _, _ in Data:
    learner = fapprox.SimpleSGDLearner()
    
    # dataToPrice[np.datetime64] := adjusted close price
    # startDate := The first date in the stock data
    # lastDate := The last date in the stock data
    dateToPrice, startDate, lastDate = load(key)

    # Fill up stocks, the prices array, so that we can look up by day-index.
    stocks = makeStockArray(dateToPrice, startDate, lastDate)

    # We learn from the first day up to one year ago.
    stocksToLearn = stocks[0:-365]

    # And test the learned weight performance by exercising in the
    # last one year.
    stocksToTest = stocks[-365:]

    errorHistory = []
    for i in range(3):
        learn(stocksToLearn, learner)

    # Reward := How much money I got/lost.
    reward = test(stocksToTest, learner)
    print '%s & %f' % (key, reward)

    plt.plot(errorHistory)
    plt.show()
    plt.close()
