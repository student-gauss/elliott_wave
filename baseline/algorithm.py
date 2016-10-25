import heapq, collections, re, sys, time, os, random

class SearchProblem:
    def startState(self): raise NotImplementedError('Override me')
    def isEnd(self, state): raise NotImplementedError('Override me')
    def succAndCost(self, state): raise NotImplementedError('Override me')

class SearchAlgorithm:
    def solve(self, problem): raise NotImplementedError('Override me.')

class UniformCostSearch(SearchAlgorithm):
    def __init__(self):
        self.actions = None
        
    def solve(self, problem):
        self.actions = None
        self.totalCost = None

        frontier = PriorityQueue()
        backpointers = {}

        startState = problem.startState()
        frontier.update(startState, 0)

        while True:
            state, pastCost = frontier.removeMin()
            if state == None: break

            if problem.isEnd(state):
                self.actions = []
                while state != startState:
                    action, prevState = backpointers[state]
                    self.actions.append(action)
                    state = prevState
                self.actions.reverse()
                self.totalCost = pastCost
                return

            for action, newState, cost in problem.succAndCost(state):
                if frontier.update(newState, pastCost + cost):
                    backpointers[newState] = (action, state)

class PriorityQueue:
    def __init__(self):
        self.DONE = -100000
        self.heap = []
        self.priorities = {}

    def update(self, state, newPriority):
        oldPriority = self.priorities.get(state)
        if oldPriority == None or newPriority < oldPriority:
            self.priorities[state] = newPriority
            heapq.heappush(self.heap, (newPriority, state))
            return True
        return False

    def removeMin(self):
        while len(self.heap) > 0:
            priority, state = heapq.heappop(self.heap)
            if self.priorities[state] == self.DONE:
                continue

            self.priorities[state] = self.DONE
            return (state, priority)
        return (None, None)

          
        
