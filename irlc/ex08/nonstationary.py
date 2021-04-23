"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.

References:
  [SB18] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT Press, second edition, 2018. (See sutton2018.pdf). 
"""
import numpy as np
import matplotlib.pyplot as plt
from irlc.ex08.simple_agents import BasicAgent
from irlc.ex08.bandits import StationaryBandit, eval_and_plot
from irlc import savepdf

class NonstationaryBandit(StationaryBandit):
    def __init__(self, k, q_star_mean=0, reward_change_std=0.01):
        self.reward_change_std = reward_change_std
        super().__init__(k, q_star_mean)

    def bandit_step(self, a): 
        """ Implement the non-stationary bandit environment (as described in (SB18)).
        Hint: use reward_change_std * np.random.randn() to generate a single random number with the given std.
         then add one to each coordinate. Remember you have to compute the regret as well, see StationaryBandit for ideas.
         (remember the optimal arm will change when you add noise to q_star) """
        # TODO: 3 lines missing.
        raise NotImplementedError("Implement function body")

    def __str__(self):
        return f"{type(self).__name__}_{self.q_star_mean}_{self.reward_change_std}"


class MovingAverageAgent(BasicAgent):
    """
    The simple bandit from (SB18, Section 2.4), but with moving average alpha
    as described in (SB18, Eqn. (2.3))
    """
    def __init__(self, env, epsilon, alpha): 
        # TODO: 2 lines missing.
        raise NotImplementedError("Implement function body")

    def train(self, s, a, r, sp, done=False): 
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")

    def __str__(self):
        return f"{type(self).__name__}_{self.epsilon}_{self.alpha}"


if __name__ == "__main__":
    plt.figure(figsize=(10, 10))
    epsilon = 0.1
    alphas = [0.15, 0.1, 0.05]

    # TODO: 4 lines missing.
    raise NotImplementedError("")

    labels = [f"Basic agent, epsilon={epsilon}"]
    # TODO: 1 lines missing.
    raise NotImplementedError("")
    use_cache = False # Set this to True to use cache (after code works!)
    eval_and_plot(bandit, agents, steps=10000, num_episodes=200, labels=labels, use_cache=use_cache)
    savepdf("nonstationary_bandits")
    plt.show()
