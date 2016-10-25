import csv
import datetime
import numpy as np
import algorithm
import model
import matplotlib.pyplot as plt

aapl = {}
startDate = np.datetime64('2014-01-01')


def stockForDate(date):
    ratio = 7.0
    while not date in aapl:
        date = date - np.timedelta64(1, 'D')

    if date >= np.datetime64('2014-06-09'):
        ratio = 1.0

    return aapl[date] / ratio

def analyze(startIndex, endIndex, step, partialSequence, endIndexes):
    print "Analyzing [%d, %d) with step %d" % (startIndex, endIndex, step)
    
    if step == 1:
        endIndexes += [endIndex]
    
    problem = model.ElliottWaveProblem(startIndex, endIndex, lambda x:stocks[x], step, partialSequence)
    ucs = algorithm.UniformCostSearch()
    ucs.solve(problem)
    
    if ucs.actions == None:
        print "No substructure in [%d, %d)" % (startIndex, endIndex)
        return
    
    lastSegmentEndIndex = startIndex
    for action in ucs.actions:
        waveType, segmentEndIndex = action

        if step != 1 and segmentEndIndex != startIndex:
            newStep = step / 6
            if newStep == 0:
                newStep = 1

            analyze(lastSegmentEndIndex, segmentEndIndex, newStep, False, endIndexes)

        lastSegmentEndIndex = segmentEndIndex

        
with open('aapl.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        date = np.datetime64(row[0])
        price = float(row[1])

        aapl[date] = price

stocks = []
date = startDate
while date < np.datetime64('2016-10-14'):
    stocks += [stockForDate(date)]
    date += np.timedelta64(1, 'D')

endIndexes = []
analyze(0, len(stocks), 180, True, endIndexes)

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111)
ax.plot(range(0, len(stocks)), stocks)

prevEndIndex = 0
for endIndex in endIndexes:
    line = ax.plot([prevEndIndex, endIndex], [stocks[prevEndIndex], stocks[endIndex]])
    plt.setp(line, color='r', linewidth=2.0)
    prevEndIndex = endIndex

plt.savefig("ucs_aapl.png")

prevEndIndex = 0
meanSquaredSum = 0.0
for endIndex in endIndexes:
    for index in range(prevEndIndex, endIndex):
        slope = float(stocks[endIndex] - stocks[prevEndIndex]) / (endIndex - prevEndIndex) 
        predicted = slope * (index - prevEndIndex) + stocks[prevEndIndex]
        actual = stocks[index]

        meanSquaredSum += (predicted - actual) ** 2 / len(stocks)
        
    prevEndIndex = endIndex

print "MSS: ", meanSquaredSum
