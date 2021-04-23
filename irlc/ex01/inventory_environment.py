"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import gym
import numpy as np
from gym.spaces.discrete import Discrete
from irlc.ex01.agent import Agent, train

class InventoryEnvironment(gym.Env): 
    def __init__(self, N=2):
        self.N = N                               # planning horizon
        self.action_space      = Discrete(3)     # Possible actions {0, 1, 2}
        self.observation_space = Discrete(3)     # Possible observations {0, 1, 2}

    def reset(self):
        self.s = 0                               # reset initial state x0=0
        self.k = 0                               # reset time step k=0
        return self.s                            # always returns the state we reset to

    def step(self, a): 
        # TODO: 2 lines missing.
        raise NotImplementedError("")
        reward = -(a + (self.s + a - w)**2)      # reward = -cost      = -g_k(x_k, u_k, w_k)
        done = self.k == self.N-1                # Have we terminated? (i.e. is k==N-1)
        self.s = s_next                          # update environment state
        # TODO: 2 lines missing.
        raise NotImplementedError("Implement function body")

class RandomAgent(Agent): 
    def pi(self, s, k=None): 
        """ Return action to take in state s at time step k """
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")

    def train(self, s, a, r, sp, done=False):
        """ Called at each step of the simulation to allow the agent to train.
        The agent was in state s, took action a, ended up in state sp (with reward r).
        'done' is a bool which indicates if the environment terminated when transitioning to sp. """
        pass 

def simplified_train(env, agent): 
    s = env.reset()
    J = 0  # Accumulated reward for this rollout
    for k in range(1000): 
        # TODO: 7 lines missing.
        raise NotImplementedError("Implement function body")
    return J 

def run_inventory():
    env = InventoryEnvironment() 
    agent = RandomAgent(env)
    stats, _ = train(env,agent,num_episodes=1,verbose=False)  # Perform one rollout.
    print("Accumulated reward of first episode", stats[0]['Accumulated Reward']) 
    # I recommend inspecting 'stats' in a debugger; why do you think it is a list of length 1?

    stats, _ = train(env, agent, num_episodes=1000,verbose=False)  # do 1000 rollouts 
    avg_reward = np.mean([stat['Accumulated Reward'] for stat in stats])
    print("[RandomAgent class] Average cost of random policy J_pi_random(0)=", -avg_reward) 

    stats, _ = train(env, Agent(env), num_episodes=1000,verbose=False)  # Perform 1000 rollouts using Agent class 
    avg_reward = np.mean([stat['Accumulated Reward'] for stat in stats])
    print("[Agent class] Average cost of random policy J_pi_random(0)=", -avg_reward)  

    """ Second part: Using the simplified training method. I.e. do not use train() below """
    # avg_reward_simplified_train = np.mean([J-values computed using 1000 calls to simplified_train(env, agent)] )
    avg_reward_simplified_train = np.mean( [simplified_train(env, agent) for i in range(1000)]) 
    print("[simplified train] Average cost of random policy J_pi_random(0) =", -avg_reward_simplified_train)  



if __name__ == "__main__":
    run_inventory()
