"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
from irlc.ex01.agent import Agent
from irlc.ex02.dp import DP_stochastic
from irlc import train
import numpy as np


class DynamicalProgrammingAgent(Agent):
    """
    This is an agent which plan using dynamical programming.
    """
    def __init__(self, env, model=None):
        super().__init__(env)
        self.J, self.pi_ = DP_stochastic(model)

    def pi(self, s, k=None):
        if k >= len(self.pi_):
            raise Exception("k >= N; I have not planned this far!")
        # TODO: 2 lines missing.
        raise NotImplementedError("return an action computed using the dp policy. The frozen_lake problem might provide inspiration.")

    def train(self, s, a, r, sp, done=False):  # Do nothing; this is DP so no learning takes place.
        pass


def main():
    from irlc.ex01.inventory_environment import InventoryEnvironment
    from irlc.ex02.inventory import InventoryDPModel

    env = InventoryEnvironment(N=3) 
    inventory = InventoryDPModel(N=3)
    agent = DynamicalProgrammingAgent(env, model=inventory)
    stats, _ = train(env, agent, num_episodes=5000) 

    s = env.reset() # Get initial state
    Er = np.mean([stat['Accumulated Reward'] for stat in stats])
    print("Estimated reward using trained policy and MC rollouts", Er)  
    print("Reward as computed using DP", -agent.J[0][s])  

if __name__ == "__main__":
    main()
