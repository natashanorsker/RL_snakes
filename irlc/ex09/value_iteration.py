"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from irlc.ex09.policy_evaluation import qs_
from irlc import savepdf

def value_iteration(mdp, gamma=.99, theta=0.0001, max_iters=10 ** 6, verbose=False):
    V = defaultdict(lambda: 0)  # value function
    for i in range(max_iters):
        delta = 0
        for s in mdp.nonterminal_states:
            # TODO: 2 lines missing.
            raise NotImplementedError("")
        if verbose:
            print(i, delta)
        if delta < theta:
            break
    pi = values2policy(mdp, V, gamma)
    return pi, V

def values2policy(mdp, V, gamma):
    pi = {}
    for s in mdp.nonterminal_states:
        # Create the policy here. pi[s] = a is the action to be taken in state s.
        # You can use the qs_ helper function to simplify things and perhaps
        # re-use ideas from the dp.py problem from week 2.
        # TODO: 2 lines missing.
        raise NotImplementedError("")
    return pi

if __name__ == "__main__":
    import seaborn as sns
    from irlc.ex09.small_gridworld import SmallGridworldMDP, plot_value_function
    env = SmallGridworldMDP()
    policy, v = value_iteration(env, gamma=0.99, theta=1e-6)
    plot_value_function(env, v)

    plt.title("Value function obtained using value iteration to find optimal policy")
    savepdf("value_iteration")
    plt.show()
