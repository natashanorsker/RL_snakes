"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.

References:
  [SB18] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT Press, second edition, 2018. (See sutton2018.pdf). 
"""
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from irlc.ex09.small_gridworld import SmallGridworldMDP,plot_value_function
from irlc import savepdf

def qs_(mdp, s, gamma, v): 
    """
    YMMW, but I found it useful to make a function to compute a dictionary of q-values:
        {a1: q(s,a1), a2: q(s,a2), ..., an: q(s,an)}
    of q(S_t=s, a) for various values of a.

    I.e. the following should work
    > Qs = qs_(mdp, s, gamma, v)
    > Qs[a] # Is the Q-value Q(s,a)

    The mathematical relationship you should implement is the Bellman expectation equation:

    Q(s,a) = E[ r + gamma * v[s'] | s,a]

     """
    return {a: np.sum([sr[1] + gamma * v[sr[0]] * p for sr, p in mdp.Psr(s,a).items()])  for a in mdp.A(s)}
    # raise NotImplementedError("Implement function body")

def policy_evaluation(pi, env, gamma=.99, theta=0.00001):
    v = defaultdict(float)
    delta = theta
    while delta >= theta:
        delta = 0 # Remember to update delta.
        for s in env.nonterminal_states: # this code works and iterate over all non-terminal states. See the MDP class if you are curious about the implementation
            # Implement the main body of the policy evaluation algorithm here
            v_ = v[s]
            # v[s] = sum([pi[s][a] * Qsa for a, Qsa in qs_(env, s, gamma, v).items()])
            Qs = qs_(env, s, gamma, v)
            v[s] = sum([p * Qs[a] for a, p in pi[s].items()])
            # raise NotImplementedError("")
            delta = max(delta, np.abs(v_ - v[s])) # stop condition. v_ is the old value of the value function (see algorithm listing in (SB18))
    return v


if __name__ == "__main__":
    env = SmallGridworldMDP()
    """
    Create the random policy pi0 below. The policy is defined as a nested dict, i.e. 
    
    > pi0[s][a] = (probability to take action a in state s)
     
    """
    pi0 = {s: {a: 1/len(env.A(s)) for a in env.A(s) } for s in env.nonterminal_states }
    V = policy_evaluation(pi0, env, gamma=1)
    plot_value_function(env, V)
    plt.title("Value function using random policy")
    savepdf("policy_eval")
    plt.show()

    expected_v = np.array([0, -14, -20, -22,
                           -14, -18, -20, -20,
                           -20, -20, -18, -14,
                           -22, -20, -14, 0])




