"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.

References:
  [SB18] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT Press, second edition, 2018. (See sutton2018.pdf). 
"""
import numpy as np
from irlc.ex01.agent import train
from irlc import main_plot
import matplotlib.pyplot as plt
from irlc.ex09old.mdp import MDPBuilder
from irlc import savepdf
from irlc.ex11.sarsa_agent import SarsaAgent
from irlc.ex11.q_agent import QAgent
from irlc.ex13.tabular_double_q import TabularDoubleQ

class MaximizationBiasEnvironment(MDPBuilder):
    """
    The Maximization Bias old from (SB18, Example 6.7).
    For easy implementation, we fix the number of transitions from state B to terminal state to
    normal_transitions. The code ensure they still have average reward 0.1, i.e. no action will be preferred.
    there are B_actions possible actions from state B in this old (the number is not given in the old).
    """
    def __init__(self, B_actions=10, normal_transitions=100, **kwargs):
        self.terminal_states = [2]
        self.state_A = 0
        self.state_B = 1
        self.LEFT = 0
        self.RIGHT = 1
        self.B_actions = B_actions
        self.n_transitions = normal_transitions
        self.states = [self.state_A, self.state_B, 2]
        self.initial_states = [self.state_A]
        super().__init__(**kwargs)

    def A(self, s):
        # define the actions pace
        if s == self.state_A:
            return [self.LEFT, self.RIGHT]
        elif s == self.state_B: # in state B
            return [n for n in range(self.B_actions)]
        else:
            return [0] # terminal; return a dummy action 0 which does nothing (some code is sensitive to empty action spaces)

    def Psr(self, s, a):
        t = self.terminal_states[0] # terminal state
        if s == self.state_A:
            if a == self.RIGHT: 
                # TODO: 1 lines missing.
                raise NotImplementedError("Implement what the environment does in state A with a RIGHT action")
            else:  
                # TODO: 1 lines missing.
                raise NotImplementedError("Implement what the environment does in state A with a LEFT action")
        else: # s is in state B
            p = 1/self.n_transitions # transition probability
            rewards = [np.random.randn() for _ in range(self.n_transitions)]
            rewards = [r - np.mean(rewards)-0.1 for r in rewards]
            return { (t, r): p for r in rewards}

if __name__ == "__main__":
    """
    The Maximization Bias from (SB18, Example 6.7).
    I have fixed the number of "junk" actions in state B to 10, but it can easily be changed 
    in the environment.

    I don't have an easy way to get the number of 'left'-actions, so instead i plot
    the trajectory length: it is 1 for a right action, and 2 for a left.
    """
    env = MaximizationBiasEnvironment()

    for _ in range(100):
        epsilon = 0.1
        alpha = 0.1
        gamma = 1
        agents = [QAgent(env, epsilon=epsilon, alpha=alpha),
                  SarsaAgent(env, epsilon=epsilon, alpha=alpha),
                  TabularDoubleQ(env, epsilon=epsilon, alpha=alpha)]

        experiments = []
        for agent in agents:
            expn = f"experiments/bias_{str(agent)}"
            train(env, agent, expn, num_episodes=300, max_runs=100)
            experiments.append(expn)

    main_plot(experiments, smoothing_window=10, y_key="Length")
    plt.ylim([1, 2])
    plt.title("Double-Q learning on Maximization-Bias old (Figure 6.5)")
    savepdf("maximization_bias_6_5")
    plt.show()

    main_plot(experiments, smoothing_window=10)
    savepdf("maximization_bias_6_5_reward")
    plt.show()
