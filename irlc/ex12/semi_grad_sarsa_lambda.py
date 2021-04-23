"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.

References:
  [SB18] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT Press, second edition, 2018. (See sutton2018.pdf). 
"""
import gym
import numpy as np
from irlc.ex01.agent import train
from irlc import main_plot, savepdf
import matplotlib.pyplot as plt
from irlc.ex11.semi_grad_sarsa import LinearSemiGradSarsa

class LinearSemiGradSarsaLambda(LinearSemiGradSarsa):
    def __init__(self, env, gamma=0.99, epsilon=0.1, alpha=0.5, lamb=0.9, q_encoder=None):
        """
        Sarsa(Lambda) with linear feature approximators (see (SB18, Section 12.7)).
        """
        super().__init__(env, gamma, alpha=alpha, epsilon=epsilon, q_encoder=q_encoder)
        # self.Q, self.e = None, None # we will not need these since we are using linear function approximators
        self.z = np.zeros(self.Q.d) # Vector to store eligibility trace (same dimension as self.w)
        self.lamb = lamb # lambda in Sarsa(lambda)

    def pi(self, s, k=None):
        if self.t == 0:
            self.a = self.pi_eps(s)
            # if self.a not in self.env.P[s]:
            #     print("bad action")
            self.x = self.Q.x(s,self.a)
            self.Q_old = 0
        return self.a

    def train(self, s, a, r, sp, done=False):
        a_prime = self.pi_eps(sp) if not done else -1
        x_prime = self.Q.x(sp, a_prime) if not done else None

        # if a_prime not in self.env.P[sp] and a_prime is not -1:
        #     print("bad action")
        """
        Update the eligibility trace self.z and the weights self.w here. 
        Note Q-values are approximated as Q = w @ x.
        We use Q_prime = w * x(s', a') to denote the new q-values for (stored for next iteration as in the pseudo code)
        """
        # TODO: 5 lines missing.
        raise NotImplementedError("Update z, w")
        if done: # Reset eligibility trace and time step t as in Sarsa.
            self.t = 0
            self.z = self.z * 0
        else:
            self.Q_old, self.x, self.a = Q_prime, x_prime, a_prime
            self.t += 1

    def __str__(self):
        return f"LinearSarsaLambda_{self.gamma}_{self.epsilon}_{self.alpha}_{self.lamb}"


from irlc.ex11.semi_grad_q import experiment_q, x, episodes
from irlc.ex11.semi_grad_sarsa import experiment_sarsa
experiment_sarsaL = "experiments/mountaincar_sarsaL"
num_of_tilings = 8
alpha = 1 / num_of_tilings / 10 # learning rate

def plot_including_week10(experiments, output):
    exps = ["../ex11/" + e for e in [experiment_q, experiment_sarsa]] + experiments

    main_plot(exps, x_key=x, y_key='Length', smoothing_window=30, resample_ticks=100)
    savepdf(output)
    plt.show()

    # Turn off averaging
    main_plot(exps, x_key=x, y_key='Length', smoothing_window=30, units="Unit", estimator=None, resample_ticks=100)
    savepdf(output+"_individual")
    plt.show()

if __name__ == "__main__":
    env = gym.make("MountainCar500-v0")
    for _ in range(5): # run experiment 10 times
        agent = LinearSemiGradSarsaLambda(env, gamma=1, alpha=alpha, epsilon=0)
        train(env, agent, experiment_sarsaL, num_episodes=episodes, max_runs=10)
    # Make plots (we use an external function so we can re-use it for the semi-gradient n-step controller)
    plot_including_week10([experiment_sarsaL], output="semigrad_sarsaL")
