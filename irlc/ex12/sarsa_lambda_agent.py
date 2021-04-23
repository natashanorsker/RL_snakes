"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import gym
from irlc.ex01.agent import train
from irlc import main_plot, savepdf
import matplotlib.pyplot as plt
from irlc.utils.common import defaultdict2
from irlc.ex11.sarsa_agent import SarsaAgent


class SarsaLambdaAgent(SarsaAgent):
    def __init__(self, env, gamma=0.99, epsilon=0.1, alpha=0.5, lamb=0.9):
        """
        Implementation of Sarsa(Lambda) in the tabular version, see
        http://incompleteideas.net/book/first/ebook/node77.html
        for details (and note that as mentioned in the exercise description/lecture Sutton forgets to reset the
        eligibility trace after each episode).
        Note 'lamb' is an abbreveation of lambda, because lambda is a reserved keyword in python.

        The constructor initializes e, the eligibility trace, as a datastructure similar to self.Q. I.e.
        self.e[s][a] is the eligibility trace e(s,a).

        Since Sarsa(Lambda) generalize Sarsa, we have to generate the next action A' from S' in the train method and
        store it for when we take actions. I.e. we can re-use the Sarsa Agents code for acting (self.pi).
        """
        super().__init__(env, gamma=gamma, alpha=alpha, epsilon=epsilon)
        self.lamb = lamb
        # self.e = TabularQ(env)
        self.e = defaultdict2(lambda x: 0)

    def train(self, s, a, r, sp, done=False):
        # TODO: 1 lines missing.
        raise NotImplementedError("a_prime = ... (get action for S'=sp using self.pi_eps; see Sarsa)")
        # TODO: 1 lines missing.
        raise NotImplementedError("delta = ... (The ordinary Sarsa learning signal)")
        # TODO: 1 lines missing.
        raise NotImplementedError("Update the eligibility trace e(s,a) += 1")
        for (s,a), ee in self.e.items():
            # TODO: 2 lines missing.
            raise NotImplementedError("Update Q values and eligibility trace")
        if done: # Clear eligibility trace after each episode (missing in pseudo code) and update variables for Sarsa
            self.e.clear()
        else:
            self.a = a_prime
            self.t += 1

    def __str__(self):
        return f"SarsaLambda_{self.gamma}_{self.epsilon}_{self.alpha}_{self.lamb}"

if __name__ == "__main__":
    envn = 'CliffWalking-v0'
    env = gym.make(envn)

    alpha =0.05
    sarsaLagent = SarsaLambdaAgent(env,gamma=0.99, epsilon=0.1, alpha=alpha, lamb=0.9)
    sarsa = SarsaAgent(env,gamma=0.99,alpha=alpha,epsilon=0.1)
    methods = [("SarsaL", sarsaLagent), ("Sarsa", sarsa)]

    experiments = []
    for k, (name,agent) in enumerate(methods):
        expn = f"experiments/{envn}_{name}"
        train(env, agent, expn, num_episodes=500, max_runs=10)
        experiments.append(expn)
    main_plot(experiments, smoothing_window=10, resample_ticks=200)
    plt.ylim([-100, 0])
    savepdf("cliff_sarsa_lambda")
    plt.show()
