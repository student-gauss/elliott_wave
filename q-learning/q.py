import sys
import csv
import datetime
import numpy as np
import collections
import random
import itertools
import matplotlib.pyplot as plt
import fapprox

Gamma = 1.0
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

def makeStockArray(dateToPrice, startDate, lastDay):
    stocks = []
    date = startDate
    while date <= lastDate:
        stocks += [stockForDate(dateToPrice, date)]
        date += np.timedelta64(1, 'D')
    return stocks

# Initial state.
def initState(initialPrice):
    history = [0] * (len(LookBack) - 1)
    assets = []
    return tuple([initialPrice, history, assets])

def advanceIndex(currentState, stocks, newIndex):
    if newIndex == 0:
        return currentState
    
    history = []
    for xi in range(1, len(LookBack)):
        lookBackIndex = newIndex - LookBack[xi]
        lookBackIndex = max(lookBackIndex, 0)
        history += [stocks[lookBackIndex]]

    _, _, currentAssets = currentState

    return tuple([stocks[newIndex], history, currentAssets])

def buyStock(currentState, quantity):
    if quantity == 0:
        return currentState

    currentPrice, history, currentAssets = currentState
    
    newAssets = list(currentAssets)
    newAssets += [(currentPrice, quantity)]

#    print 'Buy %d -> %s' % (quantity, newAssets)
    return tuple([currentPrice, history, newAssets])

def sellStock(currentState, assetIndex, quantityToSell):
    currentPrice, history, assets = currentState

    purchasePrice, currentQuantity = assets[assetIndex]
    newQuantity = currentQuantity - quantityToSell
#    print 'Current assets: %s' % assets
    assets[assetIndex] = tuple([purchasePrice, newQuantity])

#    print 'Sell index %d for %d -> %s ' % (assetIndex, quantityToSell, assets)
    
    return tuple([currentPrice, history, assets])

def generateSellAction(state, buddingAction, assetIndex, actions):
    _, _, assets = state
    if assetIndex == len(assets):
        actions += [buddingAction]
        return
    
    for stocksToSell in range(assets[assetIndex][1] + 1):
        nextBuddingAction = list(buddingAction)
        nextBuddingAction += [stocksToSell]
        generateSellAction(state, nextBuddingAction, assetIndex + 1, actions)

def generateBuyAction(state, actions):
    for stocksToBuy in range(0, 3):
        buddingAction = [stocksToBuy]
        generateSellAction(state, buddingAction, 0, actions)
    
def getActions(state):
    actions = []
    generateBuyAction(state, actions)
    return actions

def trainLearner(state, action, reward, s_prime, learner):
    V_opt, _ = getVoptAndAction(s_prime, learner)
    target = reward + Gamma * V_opt

    currentPrediction = learner.predict(learner.extractFeatures(state, action))
    learner.train(learner.extractFeatures(state, action), target)

def getVoptAndAction(state, learner):
    QoptAndActionList = []
    for action in getActions(state):
        phiX = learner.extractFeatures(state, action)
        Qopt = learner.predict(phiX)
        QoptAndActionList += [(Qopt, action)]

    return max(QoptAndActionList)
    
def takeAction(state, action):
    currentPrice, history, assets = state
    reward = 0
    s_prime = state
    for actionIndex, quantity in enumerate(action):
        if actionIndex == 0:
            # Buy
            reward -= currentPrice * quantity
            s_prime = buyStock(s_prime, action[0])
        else:
            # Sell
            reward += currentPrice * quantity
            s_prime = sellStock(s_prime, actionIndex - 1, quantity)

    newPrice, newHistory, newAssets = s_prime
    newAssets = [(purchasePrice, quantity) for purchasePrice, quantity in newAssets if quantity != 0]
    s_prime = (newPrice, newHistory, newAssets)
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
        
        s_prime = advanceIndex(s_prime, stocks, index + 1)
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

        s_prime = advanceIndex(s_prime, stocks, index + 1)
        state = s_prime

    return totalReward

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
