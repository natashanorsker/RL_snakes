"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
from irlc import savepdf
import matplotlib.pyplot as plt
from irlc.ex09.rl_agent import ValueAgent
from collections import defaultdict
from irlc.ex01.agent import train
from irlc.ex02.frozen_lake_dp import plot_value_function
import gym
import numpy as np

def get_MC_return_S(episode, gamma, first_visit=True):
    """ Helper method for computing the MC returns.
    Given an episodes in the form [ (s0,a0,r1), (s1,a1,r2), ...]
    this function computes (if first_visit=True) a new list
    [(s, G) , ... ]
    consisting of the unique s_t values in the episode along with their return G_t (computed from their first occurance).

    Alternatively, if first_visit=False, the method return a list of same length of episode
    with all s values and their return.
    """
    ss = [s for s, a, r in episode]
    G = 0
    returns = []
    for t in reversed(range(len(episode))):
        # TODO: 2 lines missing.
        raise NotImplementedError("")
        if s_t not in ss[:t] or not first_visit: 
            # TODO: 1 lines missing.
            raise NotImplementedError("Implement function body")
    return returns

class MCEvaluationAgent(ValueAgent): 
    def __init__(self, env, policy=None, gamma=1, alpha=None, first_visit=True, v_init_fun=None):
        self.episode = [] 
        self.first_visit = first_visit
        self.alpha = alpha
        if self.alpha is None:
            self.returns_sum = defaultdict(float)
            self.returns_count = defaultdict(float)
        super().__init__(env, gamma, policy, v_init_fun=v_init_fun)

    def train(self, s, a, r, sp, done=False): 
        self.episode.append( (s, a, r))
        if done:
            returns = get_MC_return_S(self.episode, self.gamma, self.first_visit)
            for s, G in returns:  
                if self.alpha: 
                    # TODO: 1 lines missing.
                    raise NotImplementedError("Implement function body")
                else: 
                    # TODO: 3 lines missing.
                    raise NotImplementedError("Implement function body")

            self.episode = []

    def __str__(self):
        return f"MCeval_{self.gamma}_{self.alpha}_{self.first_visit}"


if __name__ == "__main__":
    envn = "SmallGridworld-v0"
    from irlc import VideoMonitor
    from irlc.gridworld.gridworld import SuttonCornerGridEnvironment
    env = SuttonCornerGridEnvironment()

    gamma = 1
    episodes = 200
    agent = MCEvaluationAgent(env, gamma=gamma)
    train(env, agent, num_episodes=episodes)
    env = VideoMonitor(env, agent=agent, agent_monitor_keys=("v",))
    env.plot()
    plt.title(f"MC evaluation of {envn} using first-visit")
    savepdf("MC_value_random_smallgrid")
    plt.show()
    env.close()

    env = SuttonCornerGridEnvironment()
    agent_every = MCEvaluationAgent(env, gamma=gamma, first_visit=False)
    train(env, agent_every, num_episodes=episodes)
    env = VideoMonitor(env, agent=agent, agent_monitor_keys=("v",))
    env.plot()
    plt.title(f"MC evaluation of {envn} using every-visit")
    savepdf("MC_value_random_smallgrid_every")
    plt.show()
    env.close()

    print(f"Mean of value functions for first visit {np.mean(list(agent.v.values())):3}") 
    print(f"Mean of value functions for every visit {np.mean(list(agent_every.v.values())):3}") 

    ## Second part:
    repeats = 5000  # increase to e.g. 20'000.
    episodes = 1
    ev, fv = [], []
    env = SuttonCornerGridEnvironment()
    for _ in range(repeats): 
        """
        Instantiate two agents with first_visit=True and first_visit=False.
        Train the agents using the train function for episodes episodes. You might want to pass verbose=False to the 
        'train'-method to suppress output. 
        When done, compute the mean of agent.values() and add it to the lists ev / fv; the mean of these lists
        are the desired result. 
        """
        # TODO: 8 lines missing.
        raise NotImplementedError("Implement function body")

    print(f"First visit: Mean of value functions after {repeats} repeats {np.mean(fv):3}")  
    print(f"Every visit: Mean of value functions after {repeats} repeats {np.mean(ev):3}")  
