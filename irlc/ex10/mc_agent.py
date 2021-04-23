"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import gym
from collections import defaultdict
import matplotlib.pyplot as plt
from irlc import main_plot
from irlc import savepdf
from irlc.ex09.rl_agent import TabularAgent
from irlc import train
from irlc import VideoMonitor

def get_MC_return_SA(episode, gamma, first_visit=True):
    """ Helper method for computing the MC returns.
    Given an episodes in the form [ (s0,a0,r1), (s1,a1,r2), ...]
    this function computes (if first_visit=True) a new list
    [((s,a), G) , ... ]
    consisting of the unique (s_t,a_t) pairs in episode along with their return G_t (computed from their first occurance).

    Alternatively, if first_visit=False, the method return a list of same length of episode
    with all (s,a) pairs and their return.
    """
    sa = [(s, a) for s, a, r in episode]
    G = 0
    returns = []
    for t in reversed(range(len(episode))):
        # TODO: 2 lines missing.
        raise NotImplementedError("")
        if sa_t not in sa[:t] or not first_visit: 
            # TODO: 1 lines missing.
            raise NotImplementedError("Implement function body")
    return returns

class MCAgent(TabularAgent):
    def __init__(self, env, gamma=1.0, epsilon=0.05, alpha=None, first_visit=True):
        if alpha is None:
            self.returns_sum = defaultdict(float)
            self.returns_count = defaultdict(float)
        self.alpha = alpha
        self.first_visit = first_visit
        self.episode = []
        super().__init__(env, gamma, epsilon)

    def pi(self, s,k=None): 
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")

    def train(self, s, a, r, sp, done=False): 
        """
        Consult your implementation of value estimation agent for ideas. Note you can index the Q-values as

        self.Q[s, a] = ...

        see comments in the Agent class for more details, however for now you can consider them as simply a nested structure where
        self.Q[s, a] defaults to 0 unless the Q-value has been updated.
        """
        # TODO: 12 lines missing.
        raise NotImplementedError("Implement function body")

    def __str__(self):
        return f"MC_{self.gamma}_{self.epsilon}_{self.alpha}_{self.first_visit}"

if __name__ == "__main__":
    """ Load environment but make sure it is time-limited. Can you tell why? """
    envn = "SmallGridworld-v0"
    from irlc.gridworld.gridworld import SuttonCornerGridEnvironment, BookGridEnvironment
    env = SuttonCornerGridEnvironment(uniform_initial_state=True)
    # env = BookGridEnvironment(living_reward=-0.05) # Try this if you ar ecurious

    gamma = 1 
    episodes = 20000
    experiment="experiments/mcagent_smallgrid"
    agent = MCAgent(env, gamma=gamma, first_visit=True)
    train(env, agent, experiment_name=experiment, num_episodes=episodes)
    main_plot(experiments=[experiment], resample_ticks=200) 
    plt.title("Smallgrid MC agent value function")
    plt.ylim([-10, 0])
    savepdf("mcagent_smallgrid") 
    plt.show() 

    env = VideoMonitor(env, agent=agent, agent_monitor_keys=("Q",))
    env.plot()
    plt.title(f"MC on-policy control of {envn} using first-visit")

    # plot_value_function(env.env, {s: max([agent.Q[s,a] for a in env.env.P[s]]) for s in env.env.P} )
    savepdf("MC_agent_value_smallgrid")
    plt.show()
