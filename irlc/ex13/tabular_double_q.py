"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.

References:
  [SB18] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT Press, second edition, 2018. (See sutton2018.pdf). 
"""
import numpy as np
from irlc.ex01.agent import train
import gym
from irlc import main_plot
import matplotlib.pyplot as plt
from irlc import savepdf
from irlc.utils.common import defaultdict2
from irlc.ex11.sarsa_agent import SarsaAgent
from irlc.ex11.q_agent import QAgent

class TabularDoubleQ(QAgent):
    """
    Implement the tabular version of the double-Q learning agent from
    (SB18, Section 6.7).

    Note we will copy the Q-datastructure from the Agent class.
    """
    def __init__(self, env, gamma=1.0, alpha=0.5, epsilon=0.1):
        super().__init__(env, gamma, epsilon)
        self.alpha = alpha
        # The two Q-value functions. These are of the same type as the regular self.Q function
        # self.Q1 = defaultdict2(self.Q.default_factory)
        # self.Q2 = defaultdict2(self.Q.default_factory)
        from irlc.ex09.rl_agent import TabularQ
        self.Q1 = TabularQ(env)
        self.Q2 = TabularQ(env)
        self.Q = None  # remove self.Q (we will not use it in double Q)

    def pi(self, s, k=None):
        """
        Implement the epsilon-greedy action. The implementation is nearly identical to pi_eps in the Agent class
        which can be used for inspiration, however we should use Q1+Q2 as the Q-value.
        """
        a1, Q1 = self.Q1.get_Qs(s)
        a2, Q2 = self.Q2.get_Qs(s)
        Q = np.asarray(Q1) + np.asarray(Q2)
        # TODO: 1 lines missing.
        raise NotImplementedError("Return epsilon-greedy action using Q")


    def train(self, s, a, r, sp, done=False): 
        """
        Implement the double-Q learning rule, i.e. with probability np.random.rand() < 0.5 switch
        the role of the two Q networks Q1 and Q2. Use the code for the regular Q-agent as inspiration.
        """
        # TODO: 4 lines missing.
        raise NotImplementedError("Implement function body")

    def __str__(self):
        return f"TabularDoubleQ_{self.gamma}_{self.epsilon}_{self.alpha}"

if __name__ == "__main__":
    """ Part 1: Cliffwalking old """
    env = gym.make('CliffWalking-v0')
    epsilon = 0.1
    alpha = 0.5
    for _ in range(20):
        agents = [QAgent(env, epsilon=epsilon, alpha=alpha), SarsaAgent(env, epsilon=epsilon, alpha=alpha),
                  TabularDoubleQ(env, epsilon=epsilon, alpha=alpha)]

        experiments = []
        for agent in agents:
            expn = f"experiments/doubleq_cliffwalk_{str(agent)}"
            train(env, agent, expn, num_episodes=300, max_runs=20)
            experiments.append(expn)

    main_plot(experiments, smoothing_window=10)
    plt.ylim([-100, 0])
    plt.title("Double-Q learning on " + env.spec._env_name)
    savepdf("double_Q_learning_cliff")
    plt.show()
