"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.

References:
  [SB18] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT Press, second edition, 2018. (See sutton2018.pdf). 
"""
from irlc import savepdf
import matplotlib.pyplot as plt
from irlc.ex09.rl_agent import ValueAgent
from irlc.ex01.agent import train
from irlc.ex02.frozen_lake_dp import plot_value_function
import gym

class TDnValueAgent(ValueAgent):
    """ Implement the n-step TD(0) evaluation method from (SB18, Section 7.1)"""
    def __init__(self, env, policy=None, gamma=1, n=1, alpha=0.2, v_init_fun=None):
        self.gamma = gamma
        self.alpha = alpha
        # Variables for TD-n
        self.n = n
        self.S, self.R = None, None  # episode and state buffers.
        super().__init__(env, gamma=gamma, policy=policy, v_init_fun=v_init_fun)


    def train(self, s, a, r, sp, done=False):
        # Recall we are given S_t, A_t, R_{t+1}, S_{t+1} and done is whether t=T+1.
        if self.S is None:  # We are in the initial state. Reset buffer.
            self.R = [None]*(self.n+1)
            self.S = [None]*(self.n+1)
            self.S[0] = s
            self.t = 0
        n, t = self.n, self.t

        self.S[(t+1)%(n+1)] = sp
        self.R[(t+1)%(n+1)] = r
        if done:
            T = t+1
            tau_steps_to_train = range(t - n + 1, T)
        else:
            T = 1e10
            tau_steps_to_train = [t - n + 1]

        for tau in tau_steps_to_train:
            if tau >= 0:
                """ In the notation used by Sutton, compute the expected return G below. It is a good idea to review 
                the code before this section to get an idea about how the data structure is indexed modulo N. """
                # TODO: 3 lines missing.
                raise NotImplementedError("")

                Stau = self.S[tau%(n+1)]
                delta = (G - self.v[Stau])

                if n == 1: # Check the implementation is correct in the case where n=1. !!
                    delta_TD = (r + self.gamma * self.v[sp] - self.v[s])
                    if abs(delta-delta_TD) > 1e-10:
                        raise Exception("n=1 agreement with TD learning failed. You have at least one bug!")

                self.v[Stau] += self.alpha * (G - self.v[Stau])

        self.t += 1
        if done: # terminal state. Reset:
            self.t = 0
            self.T = None
            self.S = None


if __name__ == "__main__":
    envn = "SmallGridworld-v0"
    env = gym.make(envn)
    gamma = 1
    episodes = 1000
    agent = TDnValueAgent(env, gamma=gamma, n=5)
    train(env, agent, num_episodes=episodes)

    plot_value_function(env, agent.v)
    plt.title(f"TDn evaluation of {envn}")
    savepdf("TDn_value_random_smallgrid")
    plt.show()
