"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import numpy as np
from irlc.ex09.small_gridworld import SmallGridworldMDP
import matplotlib.pyplot as plt
from irlc.ex09.policy_evaluation import policy_evaluation
from irlc.ex01.agent import Agent
from irlc.ex09.policy_evaluation import qs_
from collections import defaultdict

class PolicyIterationAgent(Agent):
    """ this is an old of how we can combine policy iteration into the Agent interface which we will
    use in the subsequent weeks. """
    def __init__(self, env, gamma):
        self.pi, self.v = policy_iteration(env, gamma)
        super().__init__(self)

    def pi(self, s, k=None): 
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")

    def train(self, s, a, r, sp, done=False):
        pass

    def __str__(self):
        return 'PolicyIterationAgent'

def policy_iteration(env, gamma=1.0):
    """
    In this exercise, the policy will simply be a list of actions, such that pi[s] is the action
    we select in state s
    """
    pi = {s: np.random.choice(env.A(s)) for s in env.nonterminal_states}
    policy_stable = False
    V = None
    while not policy_stable:
        # Evaluate the current policy using your code from the previous exercise
        V = policy_evaluation( {s: {pi[s]: 1} for s in env.nonterminal_states}, env, gamma)
        policy_stable = True     # set to False if the policy pi changes
        # Implement the steps for policy improvement here. Start by writing a for-loop over all states
        # and implement the function body.
        # I recommend looking at the property env.nonterminal_states (see MDP class for more information).
        # TODO: 6 lines missing.
        raise NotImplementedError("")
    return pi, V

if __name__ == "__main__":
    env = SmallGridworldMDP()
    pi, v = policy_iteration(env, gamma=0.99)
    expected_v = np.array([ 0, -1, -2, -3,
                           -1, -2, -3, -2,
                           -2, -3, -2, -1,
                           -3, -2, -1,  0])

    from irlc.ex09.small_gridworld import plot_value_function
    plot_value_function(env, v)
    plt.title("Value function using policy iteration to find optimal policy")
    from irlc import savepdf
    savepdf("policy_iteration")
    plt.show()
