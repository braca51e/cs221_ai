import os

from util import *

############################################################
# Modeling

class TransportationMDP(object):
    walkCost = 1
    tramCost = 2
    failProb = 0.5
    foundReward = 100

    def __init__(self, N):
        self.N = N

    def startState(self):
        return 1

    def isEnd(self, state):
        return state == self.N

    def actions(self, state):
        results = []
        if state + 1 <= self.N:
            results.append('walk')
        if 2 * state <= self.N:
            results.append('tram')
        return results

    def succProbReward(self, state, action):
        # Return a list of (newState, prob, reward) triples, where:
        # - newState: s' (where I might end up)
        # - prob: T(s, a, s')
        # - reward: Reward(s, a, s')
        results = []
        if action == 'walk':
            reward = -self.walkCost
            if state + 1 == self.N:
                reward += self.foundReward
            results.append((state + 1, 1.0, reward))
        elif action == 'tram':
            results.append((state, self.failProb, -self.tramCost))
            reward = -self.tramCost
            if 2 * state == self.N:
                reward += self.foundReward
            results.append((2 * state, 1 - self.failProb, reward))
        return results

    def discount(self):
        return 1.0

    def states(self):
        return range(1, self.N + 1)

############################################################
# Inference algorithms
def sampleEpisode(mdp, policy):
    # Run policy on MDP once to generate an episode
    # Returns:
    #   - utility: sum of discounted rewards for the episode 
    #   - history: list of states visited
    state = mdp.startState()
    utility = 0.0  # discounted sum of rewards
    curDiscount = 1.0  # gamma^t
    history = [state]
    while True:
        if mdp.isEnd(state):
            break
        action = policy(state)
        transitions = mdp.succProbReward(state, action)
        state, prob, reward = weightedSample(transitions, [x[1] for x in transitions])
        utility += curDiscount * reward
        history.append(state)
        curDiscount *= mdp.discount()
    return history, utility

def estimateValue(mdp, policy):
    N = 100
    utilities = []
    for i in range(N):
        history, utility = sampleEpisode(mdp, policy)
        utilities.append(utility)
        print('Utility %.2f from %s' % (utility, history))
    print('Estimated value: %.2f' % (sum(utilities) / N))

def policyEvaluation(mdp, policy):
    V = {s: 0.0 for s in mdp.states()}  # estimate of V_pi(s)

    # Define estimate of Q_pi(s, a) in terms of estimate of V_pi(s)
    def Q(state, action):
        return sum(
                prob * (reward + mdp.discount() * V[newState])
                for newState, prob, reward in mdp.succProbReward(state, action)
        )

    for t in range(100):
        print('%d: %s' % (t, colorRow(V)))
        V_new = {}  # new estimate of V
        for state in mdp.states():
            if mdp.isEnd(state):
                V_new[state] = 0.0
                continue
            V_new[state] = Q(state, policy(state))
        V = V_new

def valueIteration(mdp):
    V = {s: 0.0 for s in mdp.states()}  # estimate of V_opt(s)
    policy = {}  # estimate of pi_opt(state)

    # Define estimate of Q_opt(s, a) in terms of estimate of V_opt(s)
    def Q(state, action):
        return sum(
                prob * (reward + mdp.discount() * V[newState])
                for newState, prob, reward in mdp.succProbReward(state, action)
        )

    for t in range(100):
        V_new = {}  # new values at iteration t based on t - 1 (V)
        for state in mdp.states():
            if mdp.isEnd(state):
                V_new[state], policy[state] = 0.0, None
            else:
                V_new[state], policy[state] = max((Q(state, action), action) for action in mdp.actions(state))
            print('%d: %.2f %s' % (state, V_new[state], policy[state]))
        print('%d: %s' % (t, colorRow(V)))
        print
        V = V_new


############################################################
# Main
mdp = TransportationMDP(15)
print mdp.succProbReward(3, 'walk')
print mdp.succProbReward(3, 'tram')
print mdp.succProbReward(14, 'walk')

### Policies ###
def alwaysWalk(state):
    return 'walk'

def alwaysTram(state):
    if 'tram' in mdp.actions(state):
        return 'tram'
    return 'walk'

#estimateValue(mdp, alwaysWalk)
#estimateValue(mdp, alwaysTram)
#policyEvaluation(mdp, alwaysWalk)
#policyEvaluation(mdp, alwaysTram)
valueIteration(mdp)
