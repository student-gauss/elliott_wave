import algorithm
import math

class ElliottWaveProblem(algorithm.SearchProblem):
    
    def __init__(self, startIndex, endIndex, stockForDateIndex, step, partialSequence):
        self.WAVE_0 = 0
        self.WAVE_1 = 1
        self.WAVE_2 = 2
        self.WAVE_3 = 3
        self.WAVE_4 = 4
        self.WAVE_5 = 5
        self.WAVE_A = 6
        self.WAVE_B = 7
        self.WAVE_C = 8

        self.stockForDateIndex = stockForDateIndex
        self.startIndex = startIndex
        self.endIndex = endIndex - step
        self.step = step
        self.partialSequence = partialSequence
        self.cacheMinPoint = {}
        self.cacheMaxPoint = {}
        
    def startState(self):
        # state := (currentWaveType, currentWaveEndIndex, isPositiveTrend, LongestDuration, startPriceW1, endPriceW1, startPriceWA)
        return (None, None, None, 0, None, None, None)
        
    def isEnd(self, state):
        return state[0] == self.WAVE_C

    def getCost(self, startPrice, endPrice):
        # +1.1 to avoid div-by-zero when startPrice == endPrice
        return 1 / math.log(abs(startPrice - endPrice) + 1.1)

    def getMin(self, startIndex, endIndex):
        result = self.cacheMinPoint.get((startIndex, endIndex))
        if result == None:
            result = min((self.stockForDateIndex(index), index) for index in range(startIndex, endIndex))
            self.cacheMinPoint[(startIndex, endIndex)] = result
        return result

    def getMax(self, startIndex, endIndex):
        result = self.cacheMaxPoint.get((startIndex, endIndex))
        if result == None:
            result = max((self.stockForDateIndex(index), index) for index in range(startIndex, endIndex))
            self.cacheMaxPoint[(startIndex, endIndex)] = result
        return result
    
    def makeNextState(self, waveType, endIndex, isPositiveTrend, longestDurationSoFar, startPriceW1, endPriceW1, startPriceWA):
        return (waveType, endIndex, isPositiveTrend, longestDurationSoFar, startPriceW1, endPriceW1, startPriceWA)
    
    def succAndCost(self, state):
        currentWaveType, currentWaveEndIndex, isPositiveTrend, longestDurationSoFar, startPriceW1, endPriceW1, startPriceWA = state
        result = []
        
        if currentWaveType == None:
            # When partial sequence is allowed, the first wave can be anything
            result += [(0, self.makeNextState(self.WAVE_0, 0, True, longestDurationSoFar, startPriceW1, endPriceW1, startPriceWA), 0)]
            result += [(0, self.makeNextState(self.WAVE_0, 0, False, longestDurationSoFar, startPriceW1, endPriceW1, startPriceWA), 0)]
            
            if self.partialSequence:
                result += [(0, self.makeNextState(self.WAVE_1, 0, True, longestDurationSoFar, startPriceW1, endPriceW1, startPriceWA), 1)]
                result += [(0, self.makeNextState(self.WAVE_1, 0, False, longestDurationSoFar, startPriceW1, endPriceW1, startPriceWA), 1)]

                result += [(0, self.makeNextState(self.WAVE_2, 0, True, longestDurationSoFar, startPriceW1, endPriceW1, startPriceWA), 2)]
                result += [(0, self.makeNextState(self.WAVE_2, 0, False, longestDurationSoFar, startPriceW1, endPriceW1, startPriceWA), 2)]

                result += [(0, self.makeNextState(self.WAVE_3, 0, True, longestDurationSoFar, startPriceW1, endPriceW1, startPriceWA), 3)]
                result += [(0, self.makeNextState(self.WAVE_3, 0, False, longestDurationSoFar, startPriceW1, endPriceW1, startPriceWA), 3)]

                result += [(0, self.makeNextState(self.WAVE_4, 0, True, longestDurationSoFar, startPriceW1, endPriceW1, startPriceWA), 4)]
                result += [(0, self.makeNextState(self.WAVE_4, 0, False, longestDurationSoFar, startPriceW1, endPriceW1, startPriceWA), 4)]

                result += [(0, self.makeNextState(self.WAVE_5, 0, True, longestDurationSoFar, startPriceW1, endPriceW1, startPriceWA), 5)]
                result += [(0, self.makeNextState(self.WAVE_5, 0, False, longestDurationSoFar, startPriceW1, endPriceW1, startPriceWA), 5)]

                result += [(0, self.makeNextState(self.WAVE_A, 0, True, longestDurationSoFar, startPriceW1, endPriceW1, startPriceWA), 6)]
                result += [(0, self.makeNextState(self.WAVE_A, 0, False, longestDurationSoFar, startPriceW1, endPriceW1, startPriceWA), 6)]

                result += [(0, self.makeNextState(self.WAVE_B, 0, True, longestDurationSoFar, startPriceW1, endPriceW1, startPriceWA), 7)]
                result += [(0, self.makeNextState(self.WAVE_B, 0, False, longestDurationSoFar, startPriceW1, endPriceW1, startPriceWA), 7)]

                result += [(0, self.makeNextState(self.WAVE_C, 0, True, longestDurationSoFar, startPriceW1, endPriceW1, startPriceWA), 8)]
                result += [(0, self.makeNextState(self.WAVE_C, 0, False, longestDurationSoFar, startPriceW1, endPriceW1, startPriceWA), 8)]

        if currentWaveType == self.WAVE_0:
            # Let's find wave 1
            #
            startPriceW1 = self.stockForDateIndex(self.startIndex)
            for endIndex in range(0, self.endIndex, self.step):
                endPriceW1, endIndex = self.getMax(endIndex, endIndex + self.step) if isPositiveTrend else self.getMin(endIndex, endIndex + self.step)

                if (startPriceW1 <= endPriceW1) != isPositiveTrend:
                    continue

                action = endIndex
                newState = self.makeNextState(self.WAVE_1, endIndex, isPositiveTrend, endIndex, startPriceW1, endPriceW1, startPriceWA)
                cost = self.getCost(startPriceW1, endPriceW1)
                result += [(action, newState, cost)]

        elif currentWaveType == self.WAVE_1:
            # Find wave 2.
            #
            # At any point in wave 2, the price shall be in w1's territory
            startPriceW2 = self.stockForDateIndex(currentWaveEndIndex)
            for endIndex in range(currentWaveEndIndex, self.endIndex, self.step):
                endPriceW2, endIndex = self.getMin(endIndex, endIndex + self.step) if isPositiveTrend else self.getMax(endIndex, endIndex + self.step)
                if (startPriceW2 >= endPriceW2) != isPositiveTrend:
                    # trend check
                    continue
                
                if startPriceW1 != None:
                    if (startPriceW1 < endPriceW2) != isPositiveTrend:
                        break
                    
                if longestDurationSoFar < endIndex - currentWaveEndIndex:
                    longestDurationSoFar = endIndex - currentWaveEndIndex
                    
                action = endIndex
                newState = self.makeNextState(self.WAVE_2, endIndex, isPositiveTrend, longestDurationSoFar, startPriceW1, endPriceW1, startPriceWA)
                cost = self.getCost(startPriceW2, endPriceW2)
                result += [(action, newState, cost)]
                
        elif currentWaveType == self.WAVE_2:
            # Find wave 3
            #
            # It must be longer than W1 and W2.
            #
            startPriceW3 = self.stockForDateIndex(currentWaveEndIndex)
            for endIndex in range(currentWaveEndIndex, self.endIndex, self.step):
                endPriceW3, endIndex = self.getMax(endIndex, endIndex + self.step) if isPositiveTrend else self.getMin(endIndex, endIndex + self.step)

                if endIndex - currentWaveEndIndex < longestDurationSoFar:
                    continue

                if (startPriceW3 <= endPriceW3) != isPositiveTrend:
                    continue
                
                action = endIndex
                newState = self.makeNextState(self.WAVE_3, endIndex, isPositiveTrend, endIndex - currentWaveEndIndex, startPriceW1, endPriceW1, startPriceWA)
                cost = self.getCost(startPriceW3, endPriceW3)
                result += [(action, newState, cost)]
        elif currentWaveType == self.WAVE_3:
            # Find wave 4
            #
            # It must not enter W1's territory.
            
            startPriceW4 = self.stockForDateIndex(currentWaveEndIndex)
            for endIndex in range(currentWaveEndIndex, self.endIndex, self.step):
                endPriceW4, endIndex = self.getMin(endIndex, endIndex + self.step) if isPositiveTrend else self.getMax(endIndex, endIndex + self.step)
                if (startPriceW4 >= endPriceW4) != isPositiveTrend:
                    continue

                if (endPriceW1 < endPriceW4) != isPositiveTrend:
                    break
                
                action = endIndex
                newState = self.makeNextState(self.WAVE_4, endIndex, isPositiveTrend, longestDurationSoFar, startPriceW1, endPriceW1, startPriceWA)
                cost = self.getCost(startPriceW4, endPriceW4)
                result += [(action, newState, cost)]
            
        elif currentWaveType == self.WAVE_4:
            # Find wave 5
            #
            # It cannot be longer than the longest one so far
            startPriceW5 = self.stockForDateIndex(currentWaveEndIndex)
            for endIndex in range(currentWaveEndIndex, self.endIndex, self.step):
                endPriceW5, endIndex = self.getMax(endIndex, endIndex + self.step) if isPositiveTrend else self.getMin(endIndex, endIndex + self.step)
                
                if (startPriceW5 <= endPriceW5) != isPositiveTrend:
                    continue
                
                action = endIndex
                newState = self.makeNextState(self.WAVE_5, endIndex, isPositiveTrend, longestDurationSoFar, startPriceW1, endPriceW1, startPriceWA)
                cost = self.getCost(startPriceW5, endPriceW5)
                result += [(action, newState, cost)]

        elif currentWaveType == self.WAVE_5:
            # Find wave A
            #
            # It cannot be longer than the longest one so far
            startPriceWA = self.stockForDateIndex(currentWaveEndIndex)
            for endIndex in range(currentWaveEndIndex, self.endIndex, self.step):
                endPriceWA, endIndex = self.getMin(endIndex, endIndex + self.step) if isPositiveTrend else self.getMax(endIndex, endIndex + self.step)

                if (startPriceWA >= endPriceWA) != isPositiveTrend:
                    continue

                action = endIndex
                newState = self.makeNextState(self.WAVE_A, endIndex, isPositiveTrend, longestDurationSoFar, startPriceW1, endPriceW1, startPriceWA)
                cost = self.getCost(startPriceWA, endPriceWA)
                result += [(action, newState, cost)]

        elif currentWaveType == self.WAVE_A:
            # Find wave B
            #
            # It cannot go beyond startPriceWA
            startPriceWB = self.stockForDateIndex(currentWaveEndIndex)
            for endIndex in range(currentWaveEndIndex, self.endIndex, self.step):
                endPriceWB, endIndex = self.getMax(endIndex, endIndex + self.step) if isPositiveTrend else self.getMin(endIndex, endIndex + self.step)
                
                if (startPriceWB <= endPriceWB) != isPositiveTrend:
                    continue

                action = endIndex
                newState = self.makeNextState(self.WAVE_B, endIndex, isPositiveTrend, longestDurationSoFar, startPriceW1, endPriceW1, startPriceWA)
                cost = self.getCost(startPriceWB, endPriceWB)
                result += [(action, newState, cost)]
                
        elif currentWaveType == self.WAVE_B:
            # Find wave C
            startPriceWC = self.stockForDateIndex(currentWaveEndIndex)
            endPriceWC = startPriceWC
            looksGood = True
            for endIndex in range(currentWaveEndIndex, self.endIndex, self.step):
                endPriceWC, endIndex = self.getMin(endIndex, endIndex + self.step) if isPositiveTrend else self.getMax(endIndex, endIndex + self.step)
                
                if (startPriceWC >= endPriceWC) != isPositiveTrend:
                    looksGood = False
                    break
            
            if looksGood:
                action = self.endIndex
                newState = self.makeNextState(self.WAVE_C, self.endIndex, isPositiveTrend, longestDurationSoFar, startPriceW1, endPriceW1, startPriceWA)
                cost = self.getCost(startPriceWC, endPriceWC)
                result += [(action, newState, cost)]
                
        return result

stocks = [
    0, 1, 2, 3,               # 1
    2, 1,                     # 2
    2, 3, 4, 5, 5, 4, 6, 7,   # 3
    6, 5,                     # 4
    6, 7, 8, 9,               # 5
    8, 7, 6,                  # A
    7,                        # B
    5, 3                      # C
]

problem = ElliottWaveProblem(0, len(stocks), lambda x:stocks[x], step=1, partialSequence=False)
ucs = algorithm.UniformCostSearch()
ucs.solve(problem)

lastEndIndex = 0
for action in ucs.actions:
    print action - lastEndIndex
    lastEndIndex = action

