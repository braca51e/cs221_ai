### Search problem (modeling)
import util

class SearchProblem(object):
    def startState(self):
        raise NotImplementedError

    def isEnd(self, state):
        # return boolean if state is end or not
        raise NotImplementedError

    def succAndCost(self, state):
        # return list of (action, newState, cost)
        raise NotImplementedError
        
class TransportationProblem(SearchProblem):
    def __init__(self, N):
        self.N = N

    def startState(self):
        return 1

    def isEnd(self, state):
        return state == self.N

    def succAndCost(self, state):
        result = []
        if state + 1 <= self.N:
            result.append(('walk', state + 1, 1))
        if 2 * state <= self.N:
            result.append(('tram', 2 * state, 2))
        return result

### Search algorithms (inference)

def print_solution(solution):
    cost, path = solution
    print 'totalCost:', cost
    for item in path:
        print item

def backtrackingSearch(problem):
    # returns (totalCost, path)
    def recurse(state):
        if problem.isEnd(state):
            return 0, []
        bestCost = float('inf')
        bestPath = None
        for action, newState, cost in problem.succAndCost(state):
            futureCost, futurePath = recurse(newState)
            totalCost = cost + futureCost
            if totalCost <= bestCost:
                bestCost = totalCost
                bestPath = [(action, newState, cost)] + futurePath
        return bestCost, bestPath
    return recurse(problem.startState())

def dynamicProgramming(problem):
    # returns (totalCost, path)
    cache = {}
    def recurse(state):
        if problem.isEnd(state):
            return 0, []
        if state in cache:
            return cache[state]
        bestCost = float('inf')
        bestPath = None
        for action, newState, cost in problem.succAndCost(state):
            futureCost, futurePath = recurse(newState)
            totalCost = cost + futureCost
            if totalCost <= bestCost:
                bestCost = totalCost
                bestPath = [(action, newState, cost)] + futurePath
        cache[state] = (bestCost, bestPath)
        return bestCost, bestPath
    return recurse(problem.startState())

def uniformCostSearch(problem):
    frontier = util.PriorityQueue()
    frontier.update(problem.startState(), 0)
    while True:
        # Move from frontier to explored
        state, pastCost = frontier.removeMin()
        if problem.isEnd(state):
            # Note: don't compute history
            return (pastCost, [])
        # Push out on the frontier
        for action, newState, cost in problem.succAndCost(state):
            frontier.update(newState, pastCost + cost)

### Main
problem = TransportationProblem(400)
print problem.succAndCost(5)
print problem.succAndCost(25)
#print_solution(backtrackingSearch(problem))
print_solution(dynamicProgramming(problem))
print_solution(uniformCostSearch(problem))

