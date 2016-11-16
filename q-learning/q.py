import csv
import datetime
import numpy as np
import collections
import random

Eta = 0.1
Gamma = 0.9
LookBack = [87, 54, 33, 21, 13, 8, 5, 3, 2, 1]
LookAhead = [1, 7, 14, 30]
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

learnCount = collections.defaultdict(int)
def getState(stocks, index):
    state = []
    for xi in range(1, len(LookBack)):
        prev = stocks[0]
        cur = stocks[1]
        if index - LookBack[xi - 1] > 0 and index - LookBack[xi] > 0:
            prev = stocks[index - LookBack[xi - 1]]
            cur  = stocks[index - LookBack[xi]]

            state += [+1 if prev < cur else -1]

    return tuple(state)

def actions(s):
    return [(t, d) for t in [1, -1] for d in LookAhead]

def learn(stocks, Q):
    for index in range(len(stocks) - max(LookAhead)):
        state = getState(stocks, index)
        for action in actions(state):
            predict, forcastDuration = action
            r = (stocks[index + forcastDuration] - stocks[index]) * predict
            s_prime = getState(stocks, index + forcastDuration)

            Vopt = max([Q[s_prime, a_prime] for a_prime in actions(s_prime)])
            Q[(state, action)] = (1 - Eta) * Q[(state, action)] + Eta * (r + Vopt)
            learnCount[(state, action)] += 1

def test(stocks, Q):
    reward = 0.0
    total = 0.0
    win = 0.0
    unknown = 0
    index = 0
    while index < len(stocks) - max(LookAhead):
        s = getState(stocks, index)
        _, action = max([(Q[(s, a)], a) for a in actions(s)])
        trade, duration = action
        reward += (stocks[index + duration] - stocks[index]) * trade
        index += duration
    return reward
        
Q = collections.defaultdict(int)
for key, dataStartDate, dataEndDate in Data:
    dateToPrice, startDate, endDate = load(key)
    stocks = []
    date = startDate
    while date < endDate:
        stocks += [stockForDate(dateToPrice, date)]
        date += np.timedelta64(1, 'D')
    stocksToLearn = stocks[0:-365]
    stocksToTest = stocks[-365:]
    
    learn(stocksToLearn, Q)
    reward = test(stocksToTest, Q)
    print reward
