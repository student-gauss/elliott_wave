import csv
import datetime
import numpy as np
import collections
import random

forcastDuration = 30
eta = 0.1

def load(key):
    dateToPrice = {}
    with open('../data/%s.csv' % key, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            date = np.datetime64(row[0])

            ratio = 1.0
            if key.startswith('aapl') and date < np.datetime64('2014-06-09'):
                ratio = 7.0    # split
            price = float(row[1]) / ratio

            dateToPrice[date] = price
    return dateToPrice

def stockForDate(dateToPrice, date):
    while not date in dateToPrice:
        date = date - np.timedelta64(1, 'D')
    return dateToPrice[date]

X = [87, 54, 33, 21, 13, 8, 5, 3, 2, 1]

learnData = [
#    ('dj', np.datetime64('2016-02-11'), np.datetime64('2016-06-27')),
    ('dj', np.datetime64('1985-01-29'), np.datetime64('2015-11-11')),
#    ('gdx', np.datetime64('2006-10-02'), np.datetime64('2008-03-10')),
#    ('qcom', np.datetime64('2002-08-05'), np.datetime64('2006-05-01')),
#    ('rut', np.datetime64('2011-10-03'), np.datetime64('2016-02-08')),
#    ('wmt', np.datetime64('2011-11-14'), np.datetime64('2016-11-11')),
#    ('hd', np.datetime64('2011-11-14'), np.datetime64('2016-11-11')),
#    ('low', np.datetime64('2011-11-14'), np.datetime64('2016-11-11')),
#    ('tgt', np.datetime64('1980-03-17'), np.datetime64('2016-11-11')),
#    ('cost', np.datetime64('1986-07-09'), np.datetime64('2016-11-11')),
#    ('nke', np.datetime64('1980-12-02'), np.datetime64('2016-11-11')),
#    ('ko', np.datetime64('1962-01-02'), np.datetime64('2016-11-11')),
#    ('xom', np.datetime64('1970-01-02'), np.datetime64('2016-11-11')),
#    ('cvx', np.datetime64('1970-01-02'), np.datetime64('2016-11-11')),
#    ('cop', np.datetime64('1981-12-31'), np.datetime64('2016-11-11')),
#    ('bp', np.datetime64('1977-01-03'), np.datetime64('2016-11-11')),
#    ('ibm', np.datetime64('1962-01-02'), np.datetime64('2015-05-02')),
#    ('aapl', np.datetime64('1980-12-12'), np.datetime64('2015-05-02')),
]


def getState(stocks, index):
    state = []
    for xi in range(1, len(X)):
        prev = stocks[0]
        cur = stocks[1]
        if index - X[xi - 1] > 0 and index - X[xi] > 0:
            prev = stocks[index - X[xi - 1]]
            cur  = stocks[index - X[xi]]

            state += [+1 if prev < cur else -1]

    return tuple(state)

def learn(stocks, Q):
    if len(stocks) < forcastDuration * 2:
        print "Ignore this"
        return
    
    for index in range(len(stocks) - forcastDuration * 2):
        s = getState(stocks, index)
        for a in [1, -1]:
            r = (stocks[index + forcastDuration] - stocks[index]) * a
            s_prime = getState(stocks, index + forcastDuration)
            a_prime = random.choice([+1, -1])
            
            sa = (s, a)
            
            Q[(s, a)] = (1 - eta) * Q[(s, a)] + eta * (r + Q[(s_prime, a_prime)])

Q = collections.defaultdict(int)

for key, dataStartDate, dataEndDate in learnData:
    dateToPrice = load(key)

    stocks = []
    date = dataStartDate
    while date < dataEndDate:
        stocks += [stockForDate(dateToPrice, date)]
        date += np.timedelta64(1, 'D')

    learn(stocks, Q)

testData = [
    ('dj', np.datetime64('2015-11-11'), np.datetime64('2016-11-11'))]

print "Size of Q ", len(Q)
print "Size of features ", len(X)

def test(stocks, Q):
    if len(stocks) < forcastDuration:
        print "Ignore this"
        return

    total = 0.0
    win = 0.0
    unknown = 0
    for index in range(len(stocks) - forcastDuration):
        s = getState(stocks, index)
        if (s, 1) in Q and (s, -1) in Q:
            total += 1
            positiveAction = Q[(s, +1)]
            negativeAction = Q[(s, -1)]

            r = stocks[index + forcastDuration] - stocks[index]
            if positiveAction < negativeAction and r < 0:
                # Stock should go negative
                win += 1
            elif positiveAction >= negativeAction and r >= 0:
                win += 1
        else:
            unknown += 1
    print "Accuracy: ",win / total
    print "Unknown SA: ", unknown
    print "Total SA: ", total
    
for key, dataStartDate, dataEndDate in testData:
    dateToPrice = load(key)

    stocks = []
    date = dataStartDate
    while date < dataEndDate:
        stocks += [stockForDate(dateToPrice, date)]
        date += np.timedelta64(1, 'D')

    test(stocks, Q)

