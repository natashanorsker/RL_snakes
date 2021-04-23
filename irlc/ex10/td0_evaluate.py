"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import numpy as np
import matplotlib.pyplot as plt
from irlc.ex09.rl_agent import ValueAgent
from irlc import savepdf
from irlc.ex01.agent import train
import gym_minigrid # For the small gridwordl.

class TD0ValueAgent(ValueAgent):
    def __init__(self, env, policy=None, gamma=0.99, alpha=0.05, v_init_fun=None):
        self.alpha = alpha
        super().__init__(env, gamma=gamma, policy=policy, v_init_fun=v_init_fun)

    def train(self, s, a, r, sp, done=False): 
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")

    def __str__(self):
        return f"TD0Value_{self.gamma}_{self.alpha}"

def value_function_test(env, agent, v_true, episodes=200):
    err = []
    for t in range(episodes):
        train(env, agent, num_episodes=1, verbose=False)
        err.append( np.mean( [(v_true - v0) ** 2 for k, v0 in agent.v.items()] ) )
    return np.asarray(err)

if __name__ == "__main__":
    envn = "SmallGridworld-v0"

    from irlc.gridworld.gridworld import SuttonCornerGridEnvironment 
    from irlc import VideoMonitor
    env = SuttonCornerGridEnvironment() # Make the gridworld environment itself 

    gamma = 1   
    agent = TD0ValueAgent(env, gamma=gamma, alpha=0.05) # Make a TD(0) agent
    train(env, agent, num_episodes=2000) # Train for 2000 episodes 
    env = VideoMonitor(env, agent=agent, agent_monitor_keys=("v",)) # Add a video monitor, the environment will now show an animation 
    train(env,agent,num_episodes=1) # Train for a (single) new episode
    env.plot() # Plot the current state of the environment/agent
    plt.title(f"TD0 evaluation of {envn}")
    savepdf("TD_value_random_smallgrid")
    plt.show() 
