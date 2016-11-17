import csv
import datetime
import numpy as np
import collections
import random
import itertools

Eta = 0.1
Gamma = 0.9
LookBack = [87, 54, 33, 21, 13, 8, 5, 3, 2, 1]
Epsilon = 0.5
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

def stockForDate(dateToPrice, date):
    while not date in dateToPrice:
        date = date - np.timedelta64(1, 'D')
    return dateToPrice[date]


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
    actions += range(len(assets))  # sell
    return actions

def extractFeature(state, action):
    priorPattern, currentAssets = state
    return list(priorPattern) + [len(currentAssets)] + [action]
    
def Q_opt(state, action, weights):
    product = 0.0
    for index, feature in enumerate(extractFeature(state, action)):
        product += weights[index] * feature

    return product

def updateWeights(state, action, reward, s_prime, weights):
    V_opt = max([Q_opt(s_prime, a_prime, weights) for a_prime in getActions(s_prime)])
    delta = Q_opt(state, action, weights) - (reward + Gamma * V_opt)

    for index, feature in enumerate(extractFeature(state, action)):
        weights[index] -= Eta * delta * feature

def learn(stocks, weights):
    state = initState()
    for index in range(len(stocks) - 1):
        actions = getActions(state)
        if random.random() < Epsilon:
            # pick random 
            action = random.choice(actions)
        else:
            # pick optimal action
            _, action = max([(Q_opt(state, a, weights), a) for a in actions])

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

        s_prime = advanceIndex(s_prime, stocks, index + 1)
        updateWeights(state, action, reward, s_prime, weights)
        state = s_prime

def test(stocks, weights):
    state = initState()
    r = 0
    for index in range(len(stocks) - 1):
        actions = getActions(state)
        # optimal action
        q, action = max([(Q_opt(state, a, weights), a) for a in actions])
        if action == -2:
            # Buy
            r -= stocks[index]
            s_prime = buyStock(state)
        elif action == -1:
            # No action
            r += 0
            s_prime = state
        else:
            # Sell
            _, currentAssets = state
            r += stocks[index]
            s_prime = sellStock(state, action)

        s_prime = advanceIndex(s_prime, stocks, index + 1)
        state = s_prime

    return r
        
for key, dataStartDate, dataEndDate in Data:
    weights = collections.defaultdict(int)
    dateToPrice, startDate, endDate = load(key)
    stocks = []
    date = startDate
    while date < endDate:
        stocks += [stockForDate(dateToPrice, date)]
        date += np.timedelta64(1, 'D')
    stocksToLearn = stocks[0:-365]
    stocksToTest = stocks[-365:]
    
    learn(stocksToLearn, weights)
    reward = test(stocksToTest, weights)
    print '%s & %f \\\\ ' % (key, reward)
