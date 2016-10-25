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

def analyze(containingWaveType, startIndex, endIndex, step, openStart, openEnd, waveSequence):
    print "Analyzing wave %s [%d, %d) with step %d" % (containingWaveType, startIndex, endIndex, step)
    
    ucs = algorithm.UniformCostSearch()
    if step != 0:
        problem = model.ElliottWaveProblem(startIndex, endIndex, lambda x:stocks[x], step, openStart, openEnd)
        ucs.solve(problem)
    
    if ucs.actions == None:
        print "No substructure in [%d, %d)" % (startIndex, endIndex)
        waveSequence += [(containingWaveType, endIndex)]
        return
    
    lastSegmentEndIndex = startIndex
    for action in ucs.actions:
        waveType, segmentEndIndex = action
        print "Found wave %s till %d with step %d" % (waveType, segmentEndIndex, step)
        
        
        if step != 1 and segmentEndIndex != startIndex:
            newStep = step / 6

            if segmentEndIndex == endIndex:
                print "It's the last wave in the containing wave"
                analyze(waveType, lastSegmentEndIndex, segmentEndIndex, newStep, openStart=False, openEnd=True, waveSequence = waveSequence)
            else:
                analyze(waveType, lastSegmentEndIndex, segmentEndIndex, newStep, openStart=False, openEnd=False, waveSequence = waveSequence)

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

waveSequence = []
analyze(None, 0, len(stocks), 180, openStart=True, openEnd=True, waveSequence = waveSequence)

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111)
ax.plot(range(0, len(stocks)), stocks)

prevEndIndex = 0
for wave in waveSequence:
    waveType, endIndex = wave

    endPrice = stocks[endIndex] if endIndex != len(stocks) else stocks[endIndex - 1]
    line = ax.plot([prevEndIndex, endIndex], [stocks[prevEndIndex], endPrice])
    plt.setp(line, color='r', linewidth=2.0)
    prevEndIndex = endIndex

plt.savefig("ucs_aapl.png")

prevEndIndex = 0
meanSquaredSum = 0.0
for wave in waveSequence:
    waveType, endIndex = wave
    
    for index in range(prevEndIndex, endIndex):
        endPrice = stocks[endIndex] if endIndex != len(stocks) else stocks[endIndex - 1]
        slope = float(endPrice - stocks[prevEndIndex]) / (endIndex - prevEndIndex) 
        predicted = slope * (index - prevEndIndex) + stocks[prevEndIndex]
        actual = stocks[index]

        meanSquaredSum += (predicted - actual) ** 2 / len(stocks)
        
    prevEndIndex = endIndex

print "MSS: ", meanSquaredSum
