import csv
import datetime
import numpy as np
import algorithm
import model
import matplotlib.pyplot as plt

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

def analyze(containingWaveType, startIndex, endIndex, dataEndIndex, step, openStart, openEnd, verbose = False):
    if verbose: print "Analyzing wave %s [%d, %d) with step %d" % (containingWaveType, startIndex, endIndex, step)
    waveSequence = []
    
    ucs = algorithm.UniformCostSearch()
    if step != 0:
        problem = model.ElliottWaveProblem(startIndex, endIndex, lambda x:stocks[x], step, openStart, openEnd)
        ucs.solve(problem)
    
    if ucs.actions == None:
        if verbose: print "No substructure in [%d, %d)" % (startIndex, endIndex)
        return waveSequence
    
    lastSegmentEndIndex = startIndex
    for action in ucs.actions:
        waveType, segmentEndIndex = action
        if verbose: print "Found wave %s till %d with step %d" % (waveType, segmentEndIndex, step)

        subWaveSequence = []
        if step != 1 and segmentEndIndex != startIndex:
            newStep = step / 3

            if segmentEndIndex == dataEndIndex:
                if verbose: print "It's the last wave in the data"
                subWaveSequence = analyze(waveType, lastSegmentEndIndex, segmentEndIndex, dataEndIndex, newStep, openStart=False, openEnd=True, verbose=verbose)

        waveSequence += [(waveType, lastSegmentEndIndex, segmentEndIndex, step, subWaveSequence)]
        lastSegmentEndIndex = segmentEndIndex

    return waveSequence

def printLastWave(waveSequence):
    if len(waveSequence) != 0:
        waveType, startIndex, endIndex, step, subWaveSequence = waveSequence[len(waveSequence) - 1]
        print step, waveType
        printLastWave(subWaveSequence)

def predictionError(waveSequence, futureDays):
    if len(waveSequence) != 0:
        waveType, startIndex, endIndex, step, subWaveSequence = waveSequence[len(waveSequence) - 1]

        if subWaveSequence == []:
            # This is the minimal step we found.
            startPrice = stocks[startIndex]
            endPrice = stocks[endIndex] if endIndex != len(stocks) else stocks[endIndex - 1]
            slope = float(endPrice - startPrice) / (endIndex - startIndex)
            prediction = endPrice + slope * futureDays
            
            truth = stocks[endIndex + futureDays] if endIndex + futureDays < len(stocks) else -1
            return prediction - truth
        else:
            return predictionError(subWaveSequence, futureDays)

testData = [
    ('dj', np.datetime64('2016-02-11'), np.datetime64('2016-06-27')),
    ('aapl1', np.datetime64('2011-06-20'), np.datetime64('2012-09-17')),
    ('aapl2', np.datetime64('2013-06-24'), np.datetime64('2016-05-09')),
    ('gdx', np.datetime64('2006-10-02'), np.datetime64('2008-03-10')),
    ('qcom', np.datetime64('2002-08-05'), np.datetime64('2006-05-01')),
    ('rut', np.datetime64('2011-10-03'), np.datetime64('2016-02-08'))]

for key, dataStartDate, dataEndDate in testData:
    dateToPrice = load(key)

    stocks = []
    date = dataStartDate
    while date < dataEndDate:
        stocks += [stockForDate(dateToPrice, date)]
        date += np.timedelta64(1, 'D')

    analysisDuration = (dataEndDate - dataStartDate) / 3
        
    analysisEndDate = dataStartDate + analysisDuration
    
    predictionErrors = []
    while analysisEndDate < (dataEndDate - np.timedelta64(30, 'D')):
        analysisStartDate = analysisEndDate - analysisDuration
        analysisStartIndex = int((analysisStartDate - dataStartDate) / np.timedelta64(1,'D'))
        analysisEndIndex = int((analysisEndDate - dataStartDate) / np.timedelta64(1,'D'))
    
        waveSequence = analyze(None, analysisStartIndex, analysisEndIndex, analysisEndIndex, 90, openStart=True, openEnd=True, verbose=False)
        predictionErrors.append(predictionError(waveSequence, 30))
        analysisEndDate += np.timedelta64(7, 'D')
    
    print key, sum(map(lambda x:x ** 2, predictionErrors)) / len(predictionErrors)











    # fig = plt.figure(figsize=(9, 6))
    # ax = fig.add_subplot(111)
    # ax.plot(range(analysisStartIndex, analysisEndIndex), stocks[analysisStartIndex:analysisEndIndex])
    
    # prevEndIndex = 0
    # for wave in waveSequence:
    #     waveType, startIndex, endIndex, step, subWaveSequence  = wave
    #     if prevEndIndex != 0:
    #         endPrice = stocks[endIndex] if endIndex != len(stocks) else stocks[endIndex - 1]
    #         line = ax.plot([prevEndIndex, endIndex], [stocks[prevEndIndex], endPrice])
    #         plt.setp(line, color='r', linewidth=2.0)

    #     prevEndIndex = endIndex

    # plt.savefig("ucs_aapl.png")
    # plt.close()
    
# prevEndIndex = 0
# meanSquaredSum = 0.0
# for wave in waveSequence:
#     waveType, startIndex, endIndex, step, subWaveSequence = wave
    
#     for index in range(prevEndIndex, endIndex):
#         endPrice = stocks[endIndex] if endIndex != len(stocks) else stocks[endIndex - 1]
#         slope = float(endPrice - stocks[prevEndIndex]) / (endIndex - prevEndIndex) 
#         predicted = slope * (index - prevEndIndex) + stocks[prevEndIndex]
#         actual = stocks[index]

#         meanSquaredSum += (predicted - actual) ** 2 / len(stocks)
        
#     prevEndIndex = endIndex

#print "MSS: ", meanSquaredSum
