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
Epsilon = 0.1
Data = [
    ('dj', None, None),
    ('gdx', None, None),
    ('qcom', None, None),
    ('rut', None, None),
    ('wmt', None, None),
    ('hd', None, None),
    ('low', None, None),
    ('tgt', None, None),
    ('cost', None, None),
    ('nke', None, None),
    ('ko', None, None),
    ('xom', None, None),
    ('cvx', None, None),
    ('cop', None, None),
    ('bp', None, None),
    ('ibm', None, None),
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

# Initial state.
def initState():
    priorPattern = [0] * (len(LookBack) - 1)
    assets = []
    return tuple([tuple(priorPattern), tuple(assets)])

def advanceIndex(currentState, stocks, newIndex):
    if newIndex == 0:
        return currentState
    
    priorPattern = []
    for xi in range(1, len(LookBack)):
        prev = stocks[0]
        cur = stocks[1]
        if newIndex - LookBack[xi - 1] > 0 and newIndex - LookBack[xi] > 0:
            prev = stocks[newIndex - LookBack[xi - 1]]
            cur  = stocks[newIndex - LookBack[xi]]

        priorPattern += [+1 if prev < cur else 0]

    _, currentAssets = currentState
    newAssets = []
    for priceDiff, numStocks in currentAssets:
        newPriceDiff = priceDiff + (stocks[newIndex] - stocks[newIndex - 1])
        newAssets += [(newPriceDiff, numStocks)]

    return tuple([tuple(priorPattern), tuple(newAssets)])

def buyStock(currentState):
    priorPattern, currentAssets = currentState
    
    newAssets = list(currentAssets)
    newAssets += [(0, 1)]

    return tuple([tuple(priorPattern), tuple(newAssets)])

def sellStock(currentState, assetIndex):
    priorPattern, currentAssets = currentState
    
    newAssets = list(currentAssets)
    del newAssets[assetIndex]

    return tuple([tuple(priorPattern), tuple(newAssets)])

def getActions(state):
    _, assets = state
    
    actions = []
    actions += [-2]   # buy
    actions += [-1]   # no action
    actions += random.sample(range(len(assets)), min(5, len(assets)))  # sell
    return actions

def trainLearner(state, action, reward, s_prime, learner):
    V_opt, _ = getVoptAndAction(s_prime, learner)
    target = reward + Gamma * V_opt
    learner.train(learner.extractFeatures(state, action), target)

def getVoptAndAction(state, learner):
    QoptAndActionList = []
    for action in getActions(state):
        phiX = learner.extractFeatures(state, action)
        Qopt = learner.predict(phiX)
        QoptAndActionList += [(Qopt, action)]

    return max(QoptAndActionList)
    
def takeAction(stocks, index, state, action):
    reward = 0
    if action == -2:
        # Buy
        reward = -stocks[index]
        s_prime = buyStock(state)
    elif action == -1:
        # No action
        reward = 0
        s_prime = state
    else:
        # Sell
        _, currentAssets = state
        reward = stocks[index]
        s_prime = sellStock(state, action)

    return (reward, s_prime)

def learn(stocks, learner):
    state = initState()
    totalReward = 0
    for index in range(len(stocks) - 1):
        actions = getActions(state)
        if random.random() < Epsilon:
            # pick random 
            action = random.choice(actions)
        else:
            # pick optimal action
            _, action = getVoptAndAction(state, learner)

        reward, s_prime = takeAction(stocks, index, state, action)
        totalReward += reward
        
        s_prime = advanceIndex(s_prime, stocks, index + 1)
        trainLearner(state, action, reward, s_prime, learner)
        state = s_prime
        
    print "In-test reward: ", totalReward
    
    
def test(stocks, learner):
    state = initState()
    totalReward = 0
    for index in range(len(stocks) - 1):
        actions = getActions(state)
        # optimal action
        _, action = getVoptAndAction(state, learner)
        reward, s_prime = takeAction(stocks, index, state, action)
        totalReward += reward

        s_prime = advanceIndex(s_prime, stocks, index + 1)
        state = s_prime

    return totalReward

def makeStockArray(dateToPrice, startDate, lastDay):
    stocks = []
    date = startDate
    while date <= lastDate:
        stocks += [stockForDate(dateToPrice, date)]
        date += np.timedelta64(1, 'D')
    return stocks

for key, _, _ in Data:
    learner = fapprox.SimpleNNLearner()
    
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

    learn(stocksToLearn, learner)

    # Reward := How much money I got/lost.
    reward = test(stocksToTest, learner)
    print '%s & %f' % (key, reward)

    plt.show()
    plt.close()
