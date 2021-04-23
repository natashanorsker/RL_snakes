"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
from irlc.ex09.mdp import MDP
from irlc.ex09.value_iteration import value_iteration
import matplotlib.pyplot as plt
import numpy as np

# These are the snake rules: If you move to state s, then you are teleported to snake_rules[s].
snake_rules = {
        2: 16,
        4: 8,
        7: 21,
        10: 3,
        12: 25,
        14: 1,
        17: 27,
        19: 5,
        22: 3,
        23: 32,
        24: 44,
        26: 44,
        28: 38,
        30: 18,
        33: 48,
        35: 11,
        36: 34,
        40: 53,
        41: 29,
        42: 9,
        45: 51,
        47: 31,
        50: 25,
        52: 38,
    }

class SnakesMDP(MDP):
    def __init__(self, rules={}):
        self.initial_state = 0
        self.rules = rules
        self.smax = 55 # Maximal state. When a player reaches this state, he/she wins.
        super().__init__(initial_state=0)

    def is_terminal(self, state):
        return state == self.smax

    def Psr(self, state, action): 
        """ Return the dictionary of transitions. They will all have the form so that if:
        > d = Psr(state, action)
        then
        > d = { (s1,1): p1, ...}
        where p1 is the chance to move to board state s1 given we are in current board state `state`.
        Note the probabilites will often (but not in every case!) be 1/6.
        """
        # TODO: 12 lines missing.
        raise NotImplementedError("Implement function body")

    def A(self, state):
        return {0} # Just a single dummy action.

if __name__ == "__main__":
    """ 
    Rules for the snakes and ladder game: 
    The player starts in square s=0, and the game terminates when the player is in square s = 55. 
    When a player reaches the base of a ladder he/she climbs it, and when they reach a snakes mouth of a snake they are translated to the base.
    When a player overshoots the goal state they go backwards from the goal state. 
    
    A few examples:    
    If the player is in position s=0 (start)
    > roll 3: Go to state s=3. 
    > roll 2: Go to state s=16 (using the ladder)

    Or if the player is in state s=54
    > Roll 1: Win the game
    > Roll 2: stay in 54
    > Roll 3: Go to 53
    > Roll 4: Go to 52    
    """
    mdp = SnakesMDP()
    pi, V_norule = value_iteration(mdp, gamma=1 - 1e-5, theta=0.0001, max_iters=10 ** 6, verbose=False)
    width = .4
    def v2bar(V):
        k, x = zip(*V.items())
        return np.asarray(k), np.asarray(x)

    plt.figure(figsize=(10,5))
    plt.grid()
    k,x = v2bar(V_norule)
    plt.bar(k-width/2, x, width=width, label="No rules")

    mdp_rules = SnakesMDP(rules=snake_rules)
    pi, V_rules = value_iteration(mdp_rules, gamma=1 - 1e-5, theta=0.0001, max_iters=10 ** 6, verbose=False)
    k, x = v2bar(V_rules)
    plt.bar(k + width / 2, x, width=width, label="Rules")
    plt.legend()
    plt.xlabel("Current tile")
    plt.ylabel("Moves remaining")
    plt.show()

    from irlc.ex09.mdp import MDP2GymEnv
    from irlc import train
    from irlc.ex09.value_iteration_agent import ValueIterationAgent
    env = MDP2GymEnv(mdp_rules)
    # TODO: 2 lines missing.
    raise NotImplementedError("use the train()-function to simulate a bunch of games.")

    avg = np.mean(z)                # this and --
    pct95 = np.percentile(z, q=95)  # This might be of help (define z above).
    print(f"Average game length was: {avg:.1f} and 5% of the games last longer than {pct95:.1f} moves :-(") 
