class HalvingGame(object):
    def __init__(self, N):
        self.N = N

    # state = (player, number)
    def startState(self):
        return (+1, self.N)

    def actions(self, state):
        return ['-', '/']

    def succ(self, state, action):
        player, number = state
        if action == '-':
            return (-player, number - 1)
        elif action == '/':
            return (-player, number // 2)
        else:
            raise ValueError(state)

    def isEnd(self, state):
        player, number = state
        return number == 0

    def utility(self, state):
        if not self.isEnd(state):
            raise ValueError(state)
        player, number = state  # either (+1, 0), (-1, 0)
        return player * float('inf')

    def player(self, state):
        player, number = state
        return player

game = HalvingGame(15)
print(game.succ(game.startState(), '/'))

def simplePolicy(state):
    action = '-'
    print('simplePolicy: state %s, action %s' % (state, action))
    return action

def humanPolicy(state):
    while True:
        print('humanPolicy: choose action for state %s:' % str(state)),
        action = raw_input().strip()
        if action in game.actions(state):
            return action

def minimaxPolicy(state):
    def recurse(state):
        # Return (V_minimax(state), action)
        if game.isEnd(state):
            return (game.utility(state), None)
        candidates = [(recurse(game.succ(state, action))[0], action)
                for action in game.actions(state)]
        player, number = state 
        if player == +1:
            return max(candidates)
        elif player == -1:
            return min(candidates)
        else:
            raise ValueError(state)
    value, action = recurse(state)
    print('minimaxPolicy: state %s, action %s, value %f' % (state, action, value))
    return action

policies = {
        +1: humanPolicy,
        #-1: simplePolicy,
        -1: minimaxPolicy
}

state = game.startState()
while not game.isEnd(state):
    # whose turn?
    player = game.player(state)
    policy = policies[player]
    # What action?
    action = policy(state)
    # What happens next
    state = game.succ(state, action)
print('Utility: %f' % game.utility(state))


