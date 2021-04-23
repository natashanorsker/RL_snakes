"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.

References:
  [SB18] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT Press, second edition, 2018. (See sutton2018.pdf). 
"""
import numpy as np
import matplotlib.pyplot as plt
from irlc.ex08.bandits import StationaryBandit, eval_and_plot
from irlc import Agent
from irlc import savepdf

class BasicAgent(Agent):
    """
    Simple bandit as described on (SB18, Section 2.4).
    """
    def __init__(self, env, epsilon):
        super().__init__(env)
        self.k = env.action_space.n
        self.epsilon = epsilon

    def pi(self, s, t=None): 
        """ Since this is a bandit, s=None and can be ignored, while k refers to the time step in the current episode """
        if t == 0:
            # At step 0 of episode. Re-initialize data structure. 
            # TODO: 2 lines missing.
            raise NotImplementedError("")
        # compute action here 
        # TODO: 1 lines missing.
        raise NotImplementedError("")

    def train(self, s, a, r, sp, done=False): 
        """ Since this is a bandit, s=sp=None and can be ignored, and done=False and can also be ignored. """
        # TODO: 2 lines missing.
        raise NotImplementedError("Implement function body")

    def __str__(self):
        return f"BasicAgent_{self.epsilon}"

if __name__ == "__main__":
    N = 100000
    S = [np.max( np.random.randn(10) ) for _ in range(100000) ]
    print( np.mean(S), np.std(S)/np.sqrt(N) )

    use_cache = False # Set this to True to use cache (after code works!)
    from irlc.utils.timer import Timer
    timer = Timer(start=True)
    R = 100
    steps = 1000
    env = StationaryBandit(k=10) 
    agents = [BasicAgent(env, epsilon=.1), BasicAgent(env, epsilon=.01), BasicAgent(env, epsilon=0) ]
    eval_and_plot(bandit=env, agents=agents, num_episodes=100, steps=1000, max_episodes=150, use_cache=use_cache)
    savepdf("bandit_epsilon")
    plt.show() 
    print(timer.display())
