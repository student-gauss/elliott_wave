import csv
import datetime
import numpy as np
import algorithm
import model
import matplotlib.pyplot as plt
import cProfile

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


date = np.datetime64('2016-07-15')
stocks = []

while date < np.datetime64('2016-10-14'):
    stocks += [stockForDate(date)]
    date += np.timedelta64(1, 'D')
    
problem = model.ElliottWaveProblem(0, len(stocks), lambda x:stocks[x], step=30, partialSequence=True)
ucs = algorithm.UniformCostSearch()
cProfile.run('ucs.solve(problem)')

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111)
ax.plot(range(0, len(stocks)), stocks)

for action in ucs.actions:
    waveType, endIndex, subsequence = action
    print '%s %d' % (waveType, endIndex)
    if len(subsequence) != 0:
        print '    %s' % subsequence


# plt.savefig("ucs_aapl.png")
