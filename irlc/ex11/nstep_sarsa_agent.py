"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.

References:
  [SB18] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT Press, second edition, 2018. (See sutton2018.pdf). 
"""
from irlc.ex01.agent import train
import gym
from irlc import main_plot
import matplotlib.pyplot as plt
from irlc.ex11.q_agent import QAgent

class SarsaNAgent(QAgent):
    """ Implement the N-step semi-gradient sarsa agent from (SB18, Section 7.2)"""
    def __init__(self, env, gamma=1, alpha=0.2, epsilon=0.1, n=1):
        # Variables for TD-n
        self.n = n # as in n-step sarse
        # Buffer lists for previous (S_t, R_{t}, A_t) triplets
        self.R, self.S, self.A = [None] * (self.n + 1), [None] * (self.n + 1), [None] * (self.n + 1)
        # self.t = 0
        super().__init__(env, gamma=gamma, alpha=alpha, epsilon=epsilon)

    def pi(self, s, k=None):
        self.t = k  # Save current step in episode for use in train.
        if self.t == 0:
            self.A[self.t] = self.pi_eps(s)
        return self.A[self.t % (self.n+1)]

    def train(self, s, a, r, sp, done=False):
        # Recall we are given S_t, A_t, R_{t+1}, S_{t+1} and done is whether t=T+1.
        n, t = self.n, self.t
        if t == 0:  # We are in the initial state. Reset buffer.
            self.S[0], self.A[0] = s, a
        # Store current observations in buffer.
        self.S[(t+1)%(n+1)] = sp
        self.R[(t+1)%(n+1)] = r
        self.A[(t+1)%(n+1)] = self.pi_eps(sp) if not done else -1
        
        if done:
            T = t+1
            tau_steps_to_train = range(t - n + 1, T)
        else:
            T = 1e10
            tau_steps_to_train = [t - n + 1]
        # Tau represent the current tau-steps which are to be updated. The notation is compatible with that in Sutton.
        for tau in tau_steps_to_train:
            if tau >= 0:
                """
                Compute the return for this tau-step and perform the relevant Q-update. 
                The first step is to compute the expected return G in the below section. 
                """
                # TODO: 4 lines missing.
                raise NotImplementedError("Compute G= (expected return) here.")

                S_tau, A_tau = self.S[tau%(n+1)], self.A[tau%(n+1)]
                delta = (G - self._q(S_tau, A_tau))
                if n == 1: # Check your implementation is correct when n=1 by comparing it with regular Sarsa learning.
                    delta_Sarsa = (r + (0 if done else self.gamma * self._q(sp,A_tau_n)) - self._q(S_tau,A_tau))
                    if abs(delta-delta_Sarsa) > 1e-10:
                        raise Exception("n=1 agreement with Sarsa learning failed. You have at least one bug!")
                self._upd_q(S_tau, A_tau, delta)

    def _q(self, s, a): return self.Q[s,a] # Using these helper methods will come in handy when we work with function approximators, but it is optional.
    def _upd_q(self, s, a, delta): self.Q[s,a] += self.alpha * delta

    def __str__(self):
        return f"SarsaN_{self.gamma}_{self.epsilon}_{self.alpha}_{self.n}"


if __name__ == "__main__":
    envn = 'CliffWalking-v0'
    env = gym.make(envn)
    from irlc.ex11.sarsa_agent import sarsa_exp
    from irlc.ex11.q_agent import q_exp

    agent = SarsaNAgent(env, n=5, epsilon=0.1,alpha=0.5)
    exp = f"experiments/{envn}_{agent}"
    train(env, agent, exp, num_episodes=200, max_runs=10)
    main_plot([q_exp, sarsa_exp, exp], smoothing_window=10)
    plt.ylim([-100,0])
    from irlc import savepdf
    savepdf("n_step_sarsa_cliff")
    plt.show()
