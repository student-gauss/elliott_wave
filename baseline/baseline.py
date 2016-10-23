import csv
import datetime
import numpy as np
import algorithm
import model
import matplotlib.pyplot as plt

aapl = {}

with open('aapl.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        date = np.datetime64(row[0])
        price = float(row[1])

        aapl[date] = price

def stockForDate(date):
    ratio = 7.0
    while not date in aapl:
        date = date - np.timedelta64(1, 'D')

    if date >= np.datetime64('2014-06-09'):
        ratio = 1.0

    return aapl[date] / ratio


date = np.datetime64('1980-12-12')
stocks = []

while date < np.datetime64('2016-10-14'):
    stocks += [stockForDate(date)]
    date += np.timedelta64(1, 'D')

    
problem = model.ElliottWaveProblem(0, len(stocks), lambda x:stocks[x], step=100, partialSequence=True)
ucs = algorithm.UniformCostSearch()
ucs.solve(problem)

cutIndexes = []
lastEndIndex = 0
for action in ucs.actions:
    cutDate = np.datetime64('1980-12-12') + np.timedelta64(action, 'D')
    cutIndexes += [action]
    print cutDate
#    print action - lastEndIndex
    lastEndIndex = action


fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111)
ax.plot(range(0, len(stocks)), stocks)

pricesAtCutPoint = [stocks[cutIndex] for cutIndex in cutIndexes]
print pricesAtCutPoint
ax.plot(cutIndexes, pricesAtCutPoint, linestyle='-', color='red')

plt.savefig("ucs_aapl.png")
