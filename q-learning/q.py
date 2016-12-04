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
Epsilon = 0.9
Data = [
    ('dj', None, None),
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
    # ('aapl', None, None),
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

# The function returns the stock price for the day. It returns the
# last day price if the stock data for the day is not available
# (e.g. holiday)
def stockForDate(dateToPrice, date):
    while not date in dateToPrice:
        date = date - np.timedelta64(1, 'D')
    return dateToPrice[date]

def makeStockArray(dateToPrice, startDate, lastDate):
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
    cash = 1000    # $1000 as the initial budget
    return tuple([initialPrice, history, ownedStocks, cash])

def moveStateToIndex(currentState, stocks, newIndex):
    history = []
    for xi in range(len(LookBack)):
        lookBackIndex = max(newIndex - LookBack[xi], 0)
        history += [stocks[lookBackIndex]]

    _, _, ownedStocks, cash = currentState
    return tuple([stocks[newIndex], history, ownedStocks, cash])

def getActions(state):
    currentPrice, history, ownedStocks, cash = state
    actions = range(-ownedStocks, int(cash / currentPrice) + 1)
    return actions

errorHistory = []
def trainLearner(state, action, reward, s_prime, learner):
    V_opt, Action_opt = getVoptAndAction(s_prime, learner)
    target = reward + Gamma * V_opt
    
    errorHistory.append(learner.predict(learner.extractFeatures(state, action)) - target)
    
    print '[Learn] State = %s / Action = %s, Vopt = %f -> %f' % (state, action, target, learner.predict(learner.extractFeatures(state, action)))
    learner.train(learner.extractFeatures(state, action), target)

def getVoptAndAction(state, learner):
    QoptAndActionList = []
    for action in getActions(state):
        phiX = learner.extractFeatures(state, action)
        Qopt = learner.predict(phiX)
        QoptAndActionList += [(Qopt, action)]

    return max(QoptAndActionList)
    
def takeAction(state, action):
    currentPrice, history, ownedStocks, cash = state
    reward = 0
    s_prime = state

    # positive action: buy
    # negative action: sell
    reward  -= currentPrice * action
    s_prime = (currentPrice, history, ownedStocks + action, cash + reward)
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

#        print 'On state = %s, we took action = %s and got %d' % (state, action, reward)
        
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

        print 'On state = %s, we should take action = %s' % (state, action)
        
        reward, s_prime = takeAction(state, action)
        totalReward += reward

        s_prime = moveStateToIndex(s_prime, stocks, index + 1)
        state = s_prime

    return totalReward

def simpleTest():
    learner = fapprox.SimpleNNLearner()
    stocks = [100, 200, 50, 1]
    for i in range(500):
        learn(stocks, learner)

    reward = test(stocks, learner)
    print '%f' % reward
    
def main():
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
        
        errorHistory = []
        for i in range(1):
            learn(stocksToLearn, learner)
            
            # Reward := How much money I got/lost.
        reward = test(stocksToTest, learner)
        print '%s & %f' % (key, reward)
            
        plt.plot(errorHistory)
        plt.savefig('errorHistory.png')
        plt.close()


# main()
simpleTest()
